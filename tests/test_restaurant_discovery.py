"""
Restaurant discovery tests (Phase 7 Step 2).

All backend interaction is mocked (monkeypatched list_restaurants):
- no real HTTP requests
- no Gemini calls
- no Supabase / Render connections
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend_client
import main as main_module
from ai.restaurant_discovery import (
    EMPTY_REPLY,
    UNAVAILABLE_REPLY,
    is_restaurant_discovery,
    restaurant_discovery_reply,
)

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


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "Show me restaurants.",
        "Which restaurants are available?",
        "List available restaurants.",
        "Suggest some restaurants.",
        "I want to order food.",
        "restaurants",
        "Any restaurants near me?",
        "Where can I find restaurants?",
        "I need restaurants",
        "Place an order",
        "Recommend restaurants",
        "Show me the restaurants",
        "What restaurants do you have?",
        "I am looking for restaurants",
    ],
)
def test_restaurant_discovery_intent_detected(message):
    assert is_restaurant_discovery(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "I want chicken biryani",
        "Suggest vegetarian food",
        "Show me paneer pizza",
        "Make it spicy",
        "Show more options",
        "hello",
        "good morning",
        "How are you?",
        "zzzqqq",
        "What is the weather?",
        "I want to order chicken biryani",  # dish-level, not discovery
        "",
    ],
)
def test_non_discovery_messages_not_detected(message):
    assert is_restaurant_discovery(message) is False


def test_greeting_still_wins_over_discovery():
    # Greetings are checked first in /chat; detector itself must not
    # treat pure greetings as discovery.
    assert is_restaurant_discovery("hello") is False
    assert is_restaurant_discovery("hi") is False


# ---------------------------------------------------------------------------
# Formatting - only real backend fields
# ---------------------------------------------------------------------------

def test_successful_listing_multiple_restaurants(monkeypatch):
    _install_backend(monkeypatch, _ok(SAMPLE_RESTAURANTS))

    result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_discovery"
    assert result["matched_foods"] == []
    assert result["detected_preferences"] is None

    reply = result["reply"]
    assert "Here are the restaurants:" in reply
    # Header must not claim availability (backend has no such field).
    assert "available restaurants" not in reply

    # All three restaurants present with real names and IDs.
    assert "Paradise Biryani" in reply
    assert "r_paradise" in reply
    assert "Saravana Bhavan" in reply
    assert "r_saravana" in reply
    assert "Domino's Pizza" in reply
    assert "r_dominos" in reply

    # Real field values from the backend payload.
    assert "Biryani, Hyderabadi" in reply
    assert "Rating: 4.5 (3200 reviews)" in reply
    assert "Delivery time: ~30 min" in reply
    assert "Distance: 2.1 km" in reply
    assert "South Indian, Vegetarian" in reply
    assert "Rating: 4.4 (5100 reviews)" in reply
    assert "Pizza, Fast Food" in reply


def test_single_restaurant_listing(monkeypatch):
    payload = {"restaurants": [SAMPLE_RESTAURANTS["restaurants"][0]]}
    _install_backend(monkeypatch, _ok(payload))

    result = restaurant_discovery_reply("List restaurants")

    assert result["status"] == "restaurant_discovery"
    assert "Paradise Biryani" in result["reply"]
    assert "Saravana Bhavan" not in result["reply"]
    assert result["reply"].count("1.") == 1


def test_never_invents_availability_field(monkeypatch):
    # Backend payload has no availability/status field - reply must not
    # claim open/closed/available status details that do not exist.
    _install_backend(monkeypatch, _ok(SAMPLE_RESTAURANTS))

    result = restaurant_discovery_reply("Show me restaurants")
    reply_lower = result["reply"].lower()

    assert "open now" not in reply_lower
    assert "currently closed" not in reply_lower
    assert "accepting orders" not in reply_lower
    # Header uses the generic word from the prompt but does not invent
    # per-restaurant status lines.
    assert "status:" not in reply_lower


def test_partial_fields_only_show_present_ones(monkeypatch):
    # Minimal object: only id + name exist; other fields omitted.
    payload = {"restaurants": [{"id": "r_x", "name": "Only Name"}]}
    _install_backend(monkeypatch, _ok(payload))

    result = restaurant_discovery_reply("Show me restaurants")
    reply = result["reply"]

    assert "Only Name" in reply
    assert "r_x" in reply
    assert "Cuisine:" not in reply
    assert "Rating:" not in reply
    assert "Delivery time:" not in reply
    assert "Distance:" not in reply


def test_non_dict_entries_are_skipped(monkeypatch):
    payload = {
        "restaurants": [
            "garbage",
            None,
            SAMPLE_RESTAURANTS["restaurants"][0],
        ]
    }
    _install_backend(monkeypatch, _ok(payload))

    result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_discovery"
    assert "Paradise Biryani" in result["reply"]
    assert "garbage" not in result["reply"]


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------

def test_empty_restaurant_list(monkeypatch):
    _install_backend(monkeypatch, _ok({"restaurants": []}))

    result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_empty"
    assert result["reply"] == EMPTY_REPLY
    assert "Paradise" not in result["reply"]
    assert result["matched_foods"] == []


# ---------------------------------------------------------------------------
# Backend failures -> polite unavailable message, never fake data
# ---------------------------------------------------------------------------

def test_backend_connection_failure(monkeypatch):
    _install_backend(
        monkeypatch,
        _fail("connection_error", "Could not connect to OXRO backend"),
    )

    result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY
    assert result["matched_foods"] == []


def test_backend_http_error(monkeypatch):
    _install_backend(
        monkeypatch,
        _fail("http_error", "OXRO backend returned HTTP 500", status_code=500),
    )

    result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY
    assert "500" not in result["reply"]  # no internal details leaked


def test_backend_base_url_not_configured(monkeypatch):
    _install_backend(
        monkeypatch,
        _fail("base_url_not_configured", "OXRO backend base URL is not configured"),
    )

    result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY


def test_backend_timeout(monkeypatch):
    _install_backend(
        monkeypatch,
        _fail("timeout", "Request to OXRO backend timed out"),
    )

    result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY


def test_backend_invalid_json(monkeypatch):
    _install_backend(
        monkeypatch,
        _fail("invalid_json", "OXRO backend returned invalid JSON", status_code=200),
    )

    result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY


# ---------------------------------------------------------------------------
# Unexpected response structures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_result",
    [
        None,
        "not-a-dict",
        {"ok": True, "data": None},           # data not a dict
        {"ok": True, "data": []},             # data not a dict
        {"ok": True, "data": {"items": []}},  # no restaurants key
        {"ok": True, "data": {"restaurants": "oops"}},  # not a list
        {"ok": True, "data": {"restaurants": [1, 2, 3]}},  # no dicts
        {"ok": False},  # ok False without error handling shape
        {},
    ],
)
def test_unexpected_structure_returns_unavailable(monkeypatch, bad_result):
    _install_backend(monkeypatch, bad_result)

    result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY
    assert result["matched_foods"] == []


def test_backend_unexpected_exception_returns_unavailable(monkeypatch, caplog):
    """list_restaurants raising must be contained - never propagate."""
    import logging

    secretish = "boom with https://secret.example/token=abc123"

    def boom():
        raise RuntimeError(secretish)

    monkeypatch.setattr(backend_client, "list_restaurants", boom)

    with caplog.at_level(logging.ERROR, logger="oxro.restaurant_discovery"):
        result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY
    assert result["matched_foods"] == []

    # Logged by exception type only - message/args (which may contain
    # sensitive details) must not appear in the log record.
    assert any(
        "RuntimeError" in record.getMessage() for record in caplog.records
    )
    assert not any(
        "secret.example" in record.getMessage()
        or "abc123" in record.getMessage()
        for record in caplog.records
    )


def test_backend_exception_at_chat_endpoint(monkeypatch, bare_client):
    """Unexpected list_restaurants exception still yields HTTP 200 reply."""
    def boom():
        raise ValueError("internal-details-must-not-leak")

    monkeypatch.setattr(backend_client, "list_restaurants", boom)

    response = _post_chat(bare_client, "Show me restaurants", customer_id="C_EXC")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_unavailable"
    assert body["reply"] == UNAVAILABLE_REPLY
    assert "internal-details-must-not-leak" not in body["reply"]


# ---------------------------------------------------------------------------
# API endpoint integration (/chat) - mocked backend, no Gemini needed
# ---------------------------------------------------------------------------

@pytest.fixture
def bare_client(monkeypatch):
    """No lifespan: no Gemini, no SessionManager (degraded mode)."""
    monkeypatch.setenv("API_KEY", API_KEY)
    return TestClient(main_module.app)


def _post_chat(client, message, customer_id="C_REST"):
    return client.post(
        "/chat",
        headers=H,
        json={"customer_id": customer_id, "message": message},
    )


def test_chat_restaurant_discovery_success(monkeypatch, bare_client):
    _install_backend(monkeypatch, _ok(SAMPLE_RESTAURANTS))

    response = _post_chat(bare_client, "Show me restaurants")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_discovery"
    assert body["customer_id"] == "C_REST"
    assert body["matched_foods"] == []
    assert "Paradise Biryani" in body["reply"]
    assert "r_paradise" in body["reply"]
    assert "Here are the restaurants:" in body["reply"]
    assert "available restaurants" not in body["reply"]


def test_chat_restaurant_discovery_multiple_intents(monkeypatch, bare_client):
    _install_backend(monkeypatch, _ok(SAMPLE_RESTAURANTS))

    for message in (
        "Which restaurants are available?",
        "List available restaurants.",
        "Suggest some restaurants.",
        "I want to order food.",
    ):
        response = _post_chat(bare_client, message, customer_id="C_INT")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "restaurant_discovery"
        assert "Paradise Biryani" in body["reply"]


def test_chat_restaurant_discovery_empty(monkeypatch, bare_client):
    _install_backend(monkeypatch, _ok({"restaurants": []}))

    response = _post_chat(bare_client, "Show me restaurants")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_empty"
    assert body["reply"] == EMPTY_REPLY


def test_chat_restaurant_discovery_connection_failure(monkeypatch, bare_client):
    _install_backend(
        monkeypatch,
        _fail("connection_error", "Could not connect to OXRO backend"),
    )

    response = _post_chat(bare_client, "Show me restaurants")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_unavailable"
    assert body["reply"] == UNAVAILABLE_REPLY


def test_chat_restaurant_discovery_http_error(monkeypatch, bare_client):
    _install_backend(
        monkeypatch,
        _fail("http_error", "OXRO backend returned HTTP 503", status_code=503),
    )

    response = _post_chat(bare_client, "List restaurants")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_unavailable"
    assert body["reply"] == UNAVAILABLE_REPLY


def test_chat_restaurant_discovery_unexpected_structure(monkeypatch, bare_client):
    _install_backend(monkeypatch, _ok({"restaurants": "not-a-list"}))

    response = _post_chat(bare_client, "Show me restaurants")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_unavailable"
    assert body["reply"] == UNAVAILABLE_REPLY


def test_chat_restaurant_discovery_no_session_created(monkeypatch, bare_client):
    _install_backend(monkeypatch, _ok(SAMPLE_RESTAURANTS))

    response = _post_chat(bare_client, "Show me restaurants", customer_id="C_NOSESS")

    assert response.status_code == 200
    session_manager = getattr(
        main_module.app.state, "session_manager", None
    )
    if session_manager is not None:
        assert "C_NOSESS" not in session_manager.sessions


def test_chat_restaurant_discovery_works_without_gemini(monkeypatch, bare_client):
    """Degraded mode: discovery must not require Gemini / SessionManager."""
    assert getattr(main_module.app.state, "session_manager", None) is None
    assert getattr(main_module.app.state, "gemini_client", None) is None

    _install_backend(monkeypatch, _ok(SAMPLE_RESTAURANTS))

    response = _post_chat(bare_client, "Show me restaurants", customer_id="C_DEG")

    assert response.status_code == 200
    assert response.json()["status"] == "restaurant_discovery"


# ---------------------------------------------------------------------------
# Normal food / greeting conversations continue to work
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


def test_food_message_still_uses_gemini_path(monkeypatch, bare_client):
    """Dish-level requests must NOT be captured by restaurant discovery."""
    from ai.session_manager import SessionManager

    # Install fakes so the food path can run without real Gemini.
    main_module.app.state.session_manager = SessionManager(
        None,
        chat_factory=lambda cl: FakeChat(),
    )
    main_module.app.state.gemini_client = FakeGeminiClient()

    # Discovery backend must NOT be called for dish requests.
    def fail_list():
        raise AssertionError(
            "list_restaurants must not be called for dish-level messages"
        )

    monkeypatch.setattr(backend_client, "list_restaurants", fail_list)

    response = _post_chat(bare_client, "I want chicken biryani", customer_id="C_FOOD")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]


def test_greeting_still_answered_locally(monkeypatch, bare_client):
    _install_backend(
        monkeypatch,
        _fail("connection_error", "should not matter"),
    )

    response = _post_chat(bare_client, "hello", customer_id="C_HI")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "greeting"
    assert body["matched_foods"] == []


def test_food_search_helper_unaffected_by_discovery():
    """is_restaurant_discovery must not swallow normal food queries."""
    from food_recommendation.food_search import smart_food_search

    assert is_restaurant_discovery("I want chicken biryani") is False
    result = smart_food_search("I want chicken biryani")
    foods = [item["food"] for item in result["results"]]
    assert "Chicken Biryani" in foods


def test_discovery_cannot_make_real_network_requests(monkeypatch):
    """
    Prove restaurant discovery tests never open a real network socket:

    1. Fail-closed path: with OXRO_BACKEND_BASE_URL unset, the client
       returns base_url_not_configured without any HTTP attempt.
    2. Mocked path: full discovery succeeds while socket.create_connection
       is blocked - any real dial would raise and fail the test.
    """
    import socket

    # --- Part 1: no base URL -> fail closed, never dials -------------
    monkeypatch.delenv("OXRO_BACKEND_BASE_URL", raising=False)

    closed = backend_client.list_restaurants()
    assert closed["ok"] is False
    assert closed["error"] == "base_url_not_configured"
    assert closed["data"] is None

    # --- Part 2: mocked backend + blocked sockets -> offline success -
    dialed = []

    def _blocked_create_connection(*args, **kwargs):
        dialed.append(args)
        raise OSError("real network disabled in tests")

    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection)
    _install_backend(monkeypatch, _ok(SAMPLE_RESTAURANTS))

    result = restaurant_discovery_reply("Show me restaurants")

    assert result["status"] == "restaurant_discovery"
    assert "Here are the restaurants:" in result["reply"]
    assert dialed == [], "discovery must not open real sockets"

    # Module source still never hardcodes a production host.
    source = open(backend_client.__file__, encoding="utf-8").read()
    assert "oxro-backend.onrender.com" not in source
    assert "render.com" not in source
