"""
OXRO Sandbox Customer Journey Test Suite (Phase 7 Step 3).

Offline end-to-end journeys through the real POST /chat endpoint:
    - English greeting -> food request -> follow-up
    - Restaurant discovery -> food request
    - Tamil script food conversation
    - Tanglish food conversation
    - Two customers with isolated sessions and contexts
    - End session -> fresh conversation
    - Restaurant backend failure -> recovery
    - Authentication failure for protected chat requests

Safety:
    - FakeChat / FakeGeminiClient only (no real Gemini)
    - backend_client.list_restaurants is always monkeypatched
    - OXRO_BACKEND_BASE_URL is removed (fail-closed, no Render/prod)
    - No Supabase, no external HTTP, no credentials in source
    - Self-contained: no imports from other test modules
"""

from __future__ import annotations

import os
import socket

import pytest
from fastapi.testclient import TestClient

import backend_client
import main as main_module
from ai.restaurant_discovery import UNAVAILABLE_REPLY
from ai.session_manager import SessionManager

API_KEY = "test-sandbox-key"  # matches conftest default
H = {"X-API-Key": API_KEY}

# Shape produced by backend GET /restaurants (restaurantController.list).
SAMPLE_RESTAURANTS = {
    "restaurants": [
        {
            "id": "r_paradise",
            "name": "Paradise Biryani",
            "cuisines": ["Biryani", "Hyderabadi"],
            "rating": 4.5,
            "reviews": 3200,
            "distanceLabel": "2.1 km",
            "etaMinutes": 30,
        },
        {
            "id": "r_saravana",
            "name": "Saravana Bhavan",
            "cuisines": ["South Indian", "Vegetarian"],
            "rating": 4.4,
            "reviews": 5100,
            "distanceLabel": "1.4 km",
            "etaMinutes": 25,
        },
        {
            "id": "r_dominos",
            "name": "Domino's Pizza",
            "cuisines": ["Pizza", "Fast Food"],
            "rating": 4.2,
            "reviews": 8400,
            "distanceLabel": "3.2 km",
            "etaMinutes": 35,
        },
    ]
}


# ---------------------------------------------------------------------------
# Offline fakes (local copies - not imported from other test modules)
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeChat:
    def __init__(self, client=None):
        self.history = []

    def send_message(self, message):
        self.history.append(message)
        return FakeResponse(f"AI reply to: {message}")


class FakeGeminiClient:
    def __init__(self):
        pass


def _ok(payload):
    return {
        "ok": True,
        "data": payload,
        "error": None,
        "message": None,
        "status_code": 200,
    }


def _fail(error, message, status_code=None):
    return {
        "ok": False,
        "data": None,
        "error": error,
        "message": message,
        "status_code": status_code,
    }


def _install_backend(monkeypatch, result):
    """Replace backend_client.list_restaurants with a fixed result."""

    def fake_list():
        return result

    monkeypatch.setattr(backend_client, "list_restaurants", fake_list)


def _post_chat(client, message, customer_id, headers=None):
    return client.post(
        "/chat",
        headers=headers if headers is not None else H,
        json={"customer_id": customer_id, "message": message},
    )


@pytest.fixture
def sandbox_client(monkeypatch):
    """
    Bare TestClient (no lifespan): no real Gemini client is ever built.
    SessionManager + FakeChat are installed so food journeys can run.
    Restaurant backend is always mocked; OXRO_BACKEND_BASE_URL is unset.
    """
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.delenv("OXRO_BACKEND_BASE_URL", raising=False)
    _install_backend(monkeypatch, _ok(SAMPLE_RESTAURANTS))

    client = TestClient(main_module.app)
    main_module.app.state.session_manager = SessionManager(
        None,
        chat_factory=lambda cl: FakeChat(),
    )
    main_module.app.state.gemini_client = FakeGeminiClient()

    yield client

    main_module.app.state.session_manager = None
    main_module.app.state.gemini_client = None


def _session_manager():
    return main_module.app.state.session_manager


# ---------------------------------------------------------------------------
# Journey 1: English greeting, food request, follow-up
# ---------------------------------------------------------------------------

def test_journey_english_greeting_food_and_followup(sandbox_client):
    customer = "C_J_EN"

    greeting = _post_chat(sandbox_client, "hello", customer)
    assert greeting.status_code == 200
    body = greeting.json()
    assert body["status"] == "greeting"
    assert "Hello" in body["reply"]
    assert body["matched_foods"] == []
    # Greetings create no session.
    assert customer not in _session_manager().sessions

    food = _post_chat(sandbox_client, "I want chicken biryani", customer)
    assert food.status_code == 200
    body = food.json()
    assert body["status"] == "success"
    assert body["customer_id"] == customer
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert body["detected_preferences"]["food_type"] == "non-vegetarian"
    assert customer in _session_manager().sessions

    follow = _post_chat(sandbox_client, "Make it spicy.", customer)
    assert follow.status_code == 200
    body = follow.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert body["detected_preferences"]["taste"] == "spicy"
    assert body["detected_preferences"]["food_type"] == "non-vegetarian"

    history = _session_manager().sessions[customer]["chat"].history
    assert len(history) == 2
    assert "chicken biryani" in history[0].lower()
    assert "CUSTOMER FOLLOW-UP MESSAGE" in history[1]
    assert "Chicken Biryani" in history[1]


# ---------------------------------------------------------------------------
# Journey 2: Restaurant discovery followed by food request
# ---------------------------------------------------------------------------

def test_journey_restaurant_discovery_then_food(sandbox_client):
    customer = "C_J_DISC"

    discovery = _post_chat(sandbox_client, "Show me restaurants", customer)
    assert discovery.status_code == 200
    body = discovery.json()
    assert body["status"] == "restaurant_discovery"
    assert body["matched_foods"] == []
    assert "Here are the restaurants:" in body["reply"]
    assert "Paradise Biryani" in body["reply"]
    assert "Saravana Bhavan" in body["reply"]
    assert "Domino's Pizza" in body["reply"]
    assert "r_paradise" in body["reply"]
    # Discovery is local: no session created.
    assert customer not in _session_manager().sessions

    food = _post_chat(sandbox_client, "I want chicken biryani", customer)
    assert food.status_code == 200
    body = food.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert customer in _session_manager().sessions
    # Discovery names must not leak into food matches.
    assert "Paradise Biryani" not in body["matched_foods"]


# ---------------------------------------------------------------------------
# Journey 3: Tamil food conversation
# ---------------------------------------------------------------------------

def test_journey_tamil_script_food_conversation(sandbox_client):
    customer = "C_J_TA"

    # "enakku kozhi biriyani venum" in Tamil script -> Chicken Biryani
    tamil_food = (
        "\u0b8e\u0ba9\u0b95\u0bcd\u0b95\u0bc1 \u0b95\u0bcb\u0bb4\u0bbf "
        "\u0baa\u0bbf\u0bb0\u0bbf\u0baf\u0bbe\u0ba3\u0bbf "
        "\u0bb5\u0bc7\u0ba3\u0bc1\u0bae\u0bcd"
    )
    response = _post_chat(sandbox_client, tamil_food, customer)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert body["detected_preferences"]["food_type"] == "non-vegetarian"

    context = _session_manager().get_context(customer)
    assert "Chicken Biryani" in context["matched_foods"]
    assert context["preferences"]["food_type"] == "non-vegetarian"

    grounded = _session_manager().sessions[customer]["chat"].history[-1]
    assert "Chicken Biryani" in grounded
    assert "Reply in the same language" in grounded


# ---------------------------------------------------------------------------
# Journey 4: Tanglish food conversation
# ---------------------------------------------------------------------------

def test_journey_tanglish_food_conversation(sandbox_client):
    customer = "C_J_TL"

    first = _post_chat(sandbox_client, "enakku koli biriyani venum", customer)
    assert first.status_code == 200
    body = first.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]

    follow = _post_chat(sandbox_client, "karama maathu", customer)
    assert follow.status_code == 200
    body = follow.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert body["detected_preferences"]["taste"] == "spicy"

    history = _session_manager().sessions[customer]["chat"].history
    assert len(history) == 2
    assert "CUSTOMER FOLLOW-UP MESSAGE" in history[1]
    assert "Chicken Biryani" in history[1]
    assert '"taste": "spicy"' in history[1]


# ---------------------------------------------------------------------------
# Journey 5: Two customers - isolated sessions and contexts
# ---------------------------------------------------------------------------

def test_journey_two_customers_isolated_sessions_and_contexts(sandbox_client):
    customer_a = "C_J_ISO_A"
    customer_b = "C_J_ISO_B"

    a1 = _post_chat(sandbox_client, "I want chicken biryani", customer_a)
    b1 = _post_chat(sandbox_client, "I want veg pizza", customer_b)
    assert a1.json()["matched_foods"] == ["Chicken Biryani"]
    assert b1.json()["status"] == "success"
    assert "Vegetable Pizza" in b1.json()["matched_foods"] or any(
        "Pizza" in f for f in b1.json()["matched_foods"]
    )

    a2 = _post_chat(sandbox_client, "Make it spicy.", customer_a)
    assert a2.json()["matched_foods"] == ["Chicken Biryani"]
    assert a2.json()["detected_preferences"]["taste"] == "spicy"

    sm = _session_manager()
    chat_a = sm.sessions[customer_a]["chat"]
    chat_b = sm.sessions[customer_b]["chat"]
    assert chat_a is not chat_b
    assert len(chat_a.history) == 2
    assert len(chat_b.history) == 1

    context_a = sm.get_context(customer_a)
    context_b = sm.get_context(customer_b)
    assert "Chicken Biryani" in context_a["matched_foods"]
    assert "Chicken Biryani" not in context_b["matched_foods"]
    assert context_a["preferences"]["food_type"] == "non-vegetarian"
    assert context_b["preferences"]["food_type"] == "vegetarian"
    assert context_a["preferences"].get("taste") == "spicy"


# ---------------------------------------------------------------------------
# Journey 6: End session then fresh conversation
# ---------------------------------------------------------------------------

def test_journey_end_session_then_fresh_conversation(sandbox_client):
    customer = "C_J_END"

    first = _post_chat(sandbox_client, "I want chicken biryani", customer)
    assert first.json()["status"] == "success"
    assert customer in _session_manager().sessions
    assert "Chicken Biryani" in _session_manager().get_context(customer)[
        "matched_foods"
    ]

    ended = sandbox_client.post(f"/session/{customer}/end", headers=H)
    assert ended.status_code == 200
    assert ended.json() == {"customer_id": customer, "status": "ended"}
    assert customer not in _session_manager().sessions

    # Fresh conversation: no leftover preferences / matched foods.
    fresh = _post_chat(sandbox_client, "I want veg pizza", customer)
    assert fresh.status_code == 200
    body = fresh.json()
    assert body["status"] == "success"
    assert "Chicken Biryani" not in body["matched_foods"]
    assert "Chicken Biryani" not in _session_manager().get_context(customer)[
        "matched_foods"
    ]
    assert body["detected_preferences"]["food_type"] == "vegetarian"

    # New session history starts over (only the new turn).
    history = _session_manager().sessions[customer]["chat"].history
    assert len(history) == 1
    assert "veg pizza" in history[0].lower()


# ---------------------------------------------------------------------------
# Journey 7: Restaurant backend failure then recovery
# ---------------------------------------------------------------------------

def test_journey_restaurant_backend_failure_then_recovery(
    sandbox_client, monkeypatch
):
    customer = "C_J_FAIL"

    _install_backend(
        monkeypatch,
        _fail("connection_error", "connection failed", status_code=None),
    )
    failed = _post_chat(sandbox_client, "Show me restaurants", customer)
    assert failed.status_code == 200
    body = failed.json()
    assert body["status"] == "restaurant_unavailable"
    assert body["reply"] == UNAVAILABLE_REPLY
    assert "Paradise Biryani" not in body["reply"]
    assert customer not in _session_manager().sessions

    # Recovery: backend mock restored to success payload.
    _install_backend(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    recovered = _post_chat(sandbox_client, "Show me restaurants", customer)
    assert recovered.status_code == 200
    body = recovered.json()
    assert body["status"] == "restaurant_discovery"
    assert "Paradise Biryani" in body["reply"]
    assert "Saravana Bhavan" in body["reply"]
    assert "Domino's Pizza" in body["reply"]

    # Food path remains healthy after discovery failure/recovery.
    food = _post_chat(sandbox_client, "I want chicken biryani", customer)
    assert food.json()["status"] == "success"
    assert food.json()["matched_foods"] == ["Chicken Biryani"]


# ---------------------------------------------------------------------------
# Journey 8: Authentication failure for protected chat requests
# ---------------------------------------------------------------------------

def test_journey_auth_failure_for_protected_chat(sandbox_client):
    customer = "C_J_AUTH"

    missing = sandbox_client.post(
        "/chat",
        json={"customer_id": customer, "message": "hello"},
    )
    assert missing.status_code == 401
    assert missing.json()["detail"]["status"] == "unauthorized"

    wrong = sandbox_client.post(
        "/chat",
        headers={"X-API-Key": "wrong-key"},
        json={"customer_id": customer, "message": "hello"},
    )
    assert wrong.status_code == 401
    assert wrong.json()["detail"]["status"] == "unauthorized"

    # Auth failure must not create a session or consume the greeting path.
    assert customer not in _session_manager().sessions

    # Valid key still works after failed attempts.
    ok = _post_chat(sandbox_client, "hello", customer)
    assert ok.status_code == 200
    assert ok.json()["status"] == "greeting"


# ---------------------------------------------------------------------------
# Network / environment safety guards
# ---------------------------------------------------------------------------

def test_sandbox_has_no_backend_base_url(sandbox_client):
    assert os.getenv("OXRO_BACKEND_BASE_URL") is None


def test_sandbox_blocks_real_socket_connections(monkeypatch, sandbox_client):
    """Any unexpected real network attempt must fail the test."""

    def _blocked(*args, **kwargs):
        raise AssertionError("real network request attempted in sandbox")

    monkeypatch.setattr(socket, "create_connection", _blocked)

    # Discovery uses the mock (no socket). Food uses FakeChat (no socket).
    disc = _post_chat(sandbox_client, "Show me restaurants", "C_J_NET")
    assert disc.json()["status"] == "restaurant_discovery"
    food = _post_chat(sandbox_client, "I want chicken biryani", "C_J_NET")
    assert food.json()["status"] == "success"
