"""
Phase 3 endpoint tests: authentication, validation, endpoints, CORS and
Gemini error handling.

All Gemini interaction is faked (fake chat objects / fake client) - no
real API keys, no real network calls, no quota usage.
"""

import pytest
from fastapi.testclient import TestClient

import main as main_module

from ai.conversation_manager import (
    GeminiQuotaError,
    GeminiServiceError,
    GeminiTimeoutError,
    GeminiTransientError,
)
from ai.session_manager import SessionManager


API_KEY = "test-sandbox-key"  # matches conftest default
H = {"X-API-Key": API_KEY}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeResponse:
    """Mimics the Gemini SDK response object (has .text)."""

    def __init__(self, text):
        self.text = text


class FakeChat:
    """Stands in for a Gemini chat object; records its own history."""

    def __init__(self, client=None):
        self.history = []

    def send_message(self, message):
        self.history.append(message)
        return FakeResponse(f"AI reply to: {message}")


class RaisingChat:
    """Chat fake that always raises the configured error."""

    def __init__(self, error, client=None):
        self.error = error
        self.history = []

    def send_message(self, message):
        raise self.error


class FakeModels:
    def generate_content(self, model, contents):
        return FakeResponse(
            "Recommendation from the OXRO catalog: Chicken Biryani."
        )


class RecordingModels:
    """Records every generate_content call for grounding assertions."""

    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents):
        self.calls.append({"model": model, "contents": contents})
        return FakeResponse("Listed in the OXRO catalog.")


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeModels()


def install_session_manager(chat):
    main_module.app.state.session_manager = SessionManager(
        None,
        chat_factory=lambda cl: chat
    )


@pytest.fixture
def client():
    """
    Runs the real lifespan (dummy keys from conftest), then swaps the
    Gemini client and SessionManager for offline fakes.
    """
    with TestClient(main_module.app) as test_client:
        install_session_manager(FakeChat())
        main_module.app.state.gemini_client = FakeGeminiClient()
        yield test_client


# ---------------------------------------------------------------------------
# Authentication (fail-closed X-API-Key)
# ---------------------------------------------------------------------------

def test_missing_api_key_returns_401(client):
    response = client.post(
        "/chat",
        json={"customer_id": "C1", "message": "Hi"}
    )

    assert response.status_code == 401
    assert response.json()["detail"]["status"] == "unauthorized"


def test_wrong_api_key_returns_401(client):
    response = client.post(
        "/chat",
        headers={"X-API-Key": "wrong-key"},
        json={"customer_id": "C1", "message": "Hi"}
    )

    assert response.status_code == 401
    assert response.json()["detail"]["status"] == "unauthorized"


def test_fail_closed_when_api_key_env_missing(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "Hi"}
    )

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "auth_not_configured"


def test_health_and_ready_remain_unauthenticated(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


# ---------------------------------------------------------------------------
# Request validation (Pydantic)
# ---------------------------------------------------------------------------

def test_blank_message_rejected(client):
    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "   "}
    )

    assert response.status_code == 422


def test_missing_fields_rejected(client):
    response = client.post("/chat", headers=H, json={})

    assert response.status_code == 422


def test_customer_id_too_long_rejected(client):
    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C" * 200, "message": "Hi"}
    )

    assert response.status_code == 422


def test_wrong_type_rejected(client):
    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": 123}
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

def test_chat_greeting_no_gemini_usage(client):
    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_GREET", "message": "hello"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "greeting"
    assert body["matched_foods"] == []
    assert body["reply"]

    # Greetings must not create a Gemini session (no quota usage).
    session_manager = main_module.app.state.session_manager
    assert "C_GREET" not in session_manager.sessions


def test_chat_success_returns_catalogue_match(client):
    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "I want chicken biryani"}
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert body["customer_id"] == "C1"
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert body["detected_preferences"]["food_type"] == "non-vegetarian"
    # The reply comes from the customer's chat session.
    assert "chicken" in body["reply"].lower()

    # Session was created for conversation memory.
    assert "C1" in main_module.app.state.session_manager.sessions


def test_chat_preserves_conversation_history(client):
    client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_HIST", "message": "I want chicken biryani"}
    )
    client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_HIST", "message": "I want veg pizza"}
    )

    chat = main_module.app.state.session_manager.sessions["C_HIST"]["chat"]

    assert len(chat.history) == 2
    assert "chicken biryani" in chat.history[0].lower()
    assert "veg pizza" in chat.history[1].lower()


def test_chat_invalidates_grounded_instructions():
    # Catalogue guardrails are embedded in the grounded message.
    from ai.chat_service import _build_grounded_message

    grounded = _build_grounded_message(
        "I want chicken biryani",
        ["Chicken Biryani"],
        {"food_type": "non-vegetarian"}
    )

    assert "recommend ONLY from this list" in grounded
    assert "Chicken Biryani" in grounded
    assert "Never invent" in grounded


# ---------------------------------------------------------------------------
# Gemini error mapping on /chat
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "error,expected_http,expected_status",
    [
        (GeminiQuotaError("LeakMarker_98127"), 429, "quota_exhausted"),
        (GeminiTransientError("LeakMarker_98127"), 503, "temporary_error"),
        (GeminiTimeoutError("LeakMarker_98127"), 504, "timeout"),
        (GeminiServiceError("LeakMarker_98127"), 502, "service_error"),
    ],
    ids=["quota", "transient", "timeout", "service"],
)
def test_chat_gemini_error_mapping(client, error, expected_http, expected_status):
    install_session_manager(RaisingChat(error))

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_ERR", "message": "I want pizza"}
    )

    assert response.status_code == expected_http
    assert response.json()["detail"]["status"] == expected_status
    # The fake's internal message is not leaked into the response.
    assert "LeakMarker_98127" not in response.text


def test_internal_error_details_not_exposed(client):
    install_session_manager(RaisingChat(GeminiServiceError("secret-stack-trace")))

    body = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_ERR", "message": "I want pizza"}
    ).json()

    assert "secret-stack-trace" not in str(body)


# ---------------------------------------------------------------------------
# POST /recommend-food
# ---------------------------------------------------------------------------

def test_recommend_food_success(client):
    response = client.post(
        "/recommend-food",
        headers=H,
        json={"customer_id": "C1", "message": "I want chicken biryani"}
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert "Chicken Biryani" in body["food_results"]
    assert body["ai_response"]


def test_recommend_food_no_results_maps_to_200(client, monkeypatch):
    monkeypatch.setattr(
        main_module,
        "recommend_food_with_ai",
        lambda *args, **kwargs: {
            "status": "no_results",
            "ai_response": "Nothing matched."
        }
    )

    response = client.post(
        "/recommend-food",
        headers=H,
        json={"customer_id": "C1", "message": "zzzqqq"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_results"
    assert body["food_results"] == []


@pytest.mark.parametrize(
    "error,expected_http,expected_status",
    [
        (GeminiQuotaError("LeakMarker_98127"), 429, "quota_exhausted"),
        (GeminiTransientError("LeakMarker_98127"), 503, "temporary_error"),
        (GeminiTimeoutError("LeakMarker_98127"), 504, "timeout"),
        (GeminiServiceError("LeakMarker_98127"), 502, "service_error"),
    ],
    ids=["quota", "transient", "timeout", "service"],
)
def test_recommend_food_gemini_error_mapping_matches_chat(
    client, monkeypatch, error, expected_http, expected_status
):
    """
    The recommendation module raises the same typed errors as /chat, so
    the HTTP mapping must be identical (429/503/504/502) and internal
    details must never leak.
    """
    def raising(message, gemini_client, max_retries=3):
        raise error

    monkeypatch.setattr(main_module, "recommend_food_with_ai", raising)

    response = client.post(
        "/recommend-food",
        headers=H,
        json={"customer_id": "C1", "message": "I want pizza"}
    )

    assert response.status_code == expected_http
    body = response.json()
    assert body["detail"]["status"] == expected_status
    assert "LeakMarker_98127" not in response.text


def test_recommend_food_internal_failure_maps_to_502(client, monkeypatch):
    def broken(message, gemini_client, max_retries=3):
        raise RuntimeError("secret internal failure")

    monkeypatch.setattr(main_module, "recommend_food_with_ai", broken)

    body = client.post(
        "/recommend-food",
        headers=H,
        json={"customer_id": "C1", "message": "I want pizza"}
    ).json()

    assert body["detail"]["status"] == "service_error"
    assert "secret internal failure" not in str(body)


# ---------------------------------------------------------------------------
# Catalogue grounding (review point 3)
# ---------------------------------------------------------------------------

def test_recommend_food_no_results_never_calls_gemini(client):
    """
    A request that matches nothing must return no_results WITHOUT any
    Gemini call - so no invented recommendations can ever be produced.
    """
    recording = RecordingModels()
    main_module.app.state.gemini_client.models = recording

    response = client.post(
        "/recommend-food",
        headers=H,
        json={"customer_id": "C1", "message": "zzzqqq"}
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "no_results"
    assert body["food_results"] == []
    assert recording.calls == []


def test_recommend_food_prompt_contains_matched_items_and_guardrails(client):
    """
    Gemini must receive the actual matched catalogue items plus explicit
    no-invention / no-availability-claims guardrails.
    """
    recording = RecordingModels()
    main_module.app.state.gemini_client.models = recording

    response = client.post(
        "/recommend-food",
        headers=H,
        json={"customer_id": "C1", "message": "I want chicken biryani"}
    )

    assert response.status_code == 200
    assert response.json()["food_results"] == ["Chicken Biryani"]

    assert len(recording.calls) == 1
    prompt = recording.calls[0]["contents"]

    assert "Chicken Biryani" in prompt
    assert "Recommend only food items from the available list" in prompt
    assert "Never claim that a food item is currently available" in prompt
    assert "Live availability must be confirmed by the backend" in prompt


def test_chat_grounding_no_results_message_disallows_invention(client):
    """
    /chat with a no-match request must still tell Gemini that NOTHING
    matched and to be honest, instead of leaving room for invention.
    """
    from ai.chat_service import _build_grounded_message

    grounded = _build_grounded_message("zzzqqq", [], None)

    assert "none found for this request" in grounded
    assert "honestly tell the customer that nothing matched" in grounded
    assert "Never invent" in grounded


# ---------------------------------------------------------------------------
# POST /session/{customer_id}/end
# ---------------------------------------------------------------------------

def test_end_session_after_chat(client):
    client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_END", "message": "I want chicken biryani"}
    )

    response = client.post("/session/C_END/end", headers=H)

    assert response.status_code == 200
    assert response.json() == {"customer_id": "C_END", "status": "ended"}
    assert "C_END" not in main_module.app.state.session_manager.sessions


def test_end_session_twice_returns_404(client):
    client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_END2", "message": "I want chicken biryani"}
    )
    client.post("/session/C_END2/end", headers=H)

    response = client.post("/session/C_END2/end", headers=H)

    assert response.status_code == 404
    assert response.json()["detail"]["status"] == "session_not_found"


def test_end_unknown_session_returns_404(client):
    response = client.post("/session/NOBODY/end", headers=H)

    assert response.status_code == 404
    assert response.json()["detail"]["status"] == "session_not_found"


# ---------------------------------------------------------------------------
# Degraded mode (no lifespan run -> no Gemini client)
# ---------------------------------------------------------------------------

def test_degraded_mode_chat_returns_not_ready(monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)

    # Plain TestClient: lifespan did NOT run -> no Gemini client.
    bare_client = TestClient(main_module.app)

    response = bare_client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "I want biryani"}
    )

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "not_ready"


def test_greeting_works_when_gemini_is_unavailable(monkeypatch):
    """
    Review point 1: greetings are answered locally and must work even
    when Gemini is unavailable (degraded mode). No session is created.
    Non-greeting messages still correctly report 503 not_ready.
    """
    monkeypatch.setenv("API_KEY", API_KEY)

    # Plain TestClient: lifespan did NOT run -> no Gemini client,
    # no SessionManager.
    bare_client = TestClient(main_module.app)

    response = bare_client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_DEGRADED", "message": "hello"}
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "greeting"
    assert body["matched_foods"] == []
    assert body["reply"]

    # No session was created for the greeting.
    session_manager = getattr(
        main_module.app.state, "session_manager", None
    )
    if session_manager is not None:
        assert "C_DEGRADED" not in session_manager.sessions

    # Food requests still correctly require Gemini.
    food_response = bare_client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_DEGRADED", "message": "I want biryani"}
    )
    assert food_response.status_code == 503
    assert food_response.json()["detail"]["status"] == "not_ready"


# ---------------------------------------------------------------------------
# CORS allow-list
# ---------------------------------------------------------------------------

def test_cors_allows_configured_origin(client):
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key",
        }
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_unlisted_origin(client):
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key",
        }
    )

    assert response.headers.get("access-control-allow-origin") is None

# ---------------------------------------------------------------------------
# Phase 4 Step 2: context-aware follow-ups via POST /chat
# ---------------------------------------------------------------------------

def test_chat_followup_make_it_spicy_uses_stored_context(client):
    client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_FOLLOW", "message": "I want chicken biryani"}
    )

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_FOLLOW", "message": "Make it spicy."}
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    # The refinement keeps the previously matched catalogue item.
    assert body["matched_foods"] == ["Chicken Biryani"]
    # Merged preferences: taste from this turn + earlier preferences.
    assert body["detected_preferences"]["taste"] == "spicy"
    assert body["detected_preferences"]["food_type"] == "non-vegetarian"

    # The grounded FOLLOW-UP message (not an honest no-match) was sent.
    chat = main_module.app.state.session_manager.sessions["C_FOLLOW"]["chat"]
    assert "CUSTOMER FOLLOW-UP MESSAGE" in chat.history[-1]
    assert "Chicken Biryani" in chat.history[-1]


def test_chat_show_more_returns_alternatives_not_repeats(client):
    client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_MORE", "message": "I want chicken biryani"}
    )

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_MORE", "message": "Show more."}
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    foods = body["matched_foods"]

    # Genuine alternatives: non-empty, real catalogue items, and the
    # already-recommended item is never repeated.
    assert len(foods) > 0
    assert "Chicken Biryani" not in foods

    from food_recommendation.recommendation_engine import FOOD_CATEGORIES
    catalogue = {
        food
        for items in FOOD_CATEGORIES.values()
        for food in items
    }
    for food in foods:
        assert food in catalogue

    # Shown list accumulated in context for future continuation turns.
    context = main_module.app.state.session_manager.get_context("C_MORE")
    assert "Chicken Biryani" in context["matched_foods"]
    for food in foods:
        assert food in context["matched_foods"]
