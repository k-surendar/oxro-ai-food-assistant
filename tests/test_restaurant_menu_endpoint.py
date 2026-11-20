"""
Restaurant menu endpoint integration tests (Phase 7 Step 3).

POST /chat menu short-circuit:
- all backend list/menu calls are mocked
- no real HTTP, no Gemini, no Supabase / Render
- OXRO_BACKEND_BASE_URL is always unset
"""

from __future__ import annotations

import os
import socket

import pytest
from fastapi.testclient import TestClient

import backend_client
import main as main_module
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

# Shape produced by backend GET /restaurants/{id}/menu.
SAMPLE_MENU = {
    "menu": [
        {
            "id": 900001,
            "name": "Hyderabadi Chicken Biryani",
            "brand": "Paradise Biryani",
            "price": 249,
            "mrp": 299,
            "rating": 4.3,
            "reviews": 980,
            "restaurantId": "r_paradise",
            "restaurantName": "Paradise Biryani",
            "veg": False,
            "egg": False,
            "description": "",
            "imageUrl": None,
        },
        {
            "id": 900003,
            "name": "Veg Dum Biryani",
            "brand": "Paradise Biryani",
            "price": 199,
            "mrp": 249,
            "rating": 4.2,
            "reviews": 410,
            "restaurantId": "r_paradise",
            "restaurantName": "Paradise Biryani",
            "veg": True,
            "egg": False,
            "description": "",
            "imageUrl": None,
        },
    ]
}


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
        self.models = self

    def generate_content(self, model, contents):
        raise AssertionError("Gemini must not be called for menu requests")


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


def _install_list(monkeypatch, result):
    def fake_list():
        return result

    monkeypatch.setattr(backend_client, "list_restaurants", fake_list)


def _install_menu(monkeypatch, result):
    def fake_menu(restaurant_id=None):
        return result

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fake_menu)


def _install_menu_capture(monkeypatch, result):
    seen = {"calls": []}

    def fake_menu(restaurant_id=None):
        seen["calls"].append(restaurant_id)
        return result

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fake_menu)
    return seen


def _post_chat(client, message, customer_id="C_MENU", headers=None):
    return client.post(
        "/chat",
        headers=headers if headers is not None else H,
        json={"customer_id": customer_id, "message": message},
    )


@pytest.fixture
def menu_client(monkeypatch):
    """
    Bare TestClient (no lifespan): no real Gemini client is built.
    SessionManager + FakeChat are installed so non-menu paths can run
    when needed, but menu short-circuits before SessionManager access.
    OXRO_BACKEND_BASE_URL is always unset.
    """
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.delenv("OXRO_BACKEND_BASE_URL", raising=False)
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok(SAMPLE_MENU))

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
# A. Successful menu request
# ---------------------------------------------------------------------------

def test_menu_endpoint_success(menu_client):
    response = _post_chat(
        menu_client,
        "Show me the Paradise Biryani menu",
        customer_id="C_MENU_OK",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_menu"
    assert body["customer_id"] == "C_MENU_OK"
    assert body["matched_foods"] == []
    assert body["detected_preferences"] is None
    assert "Menu for Paradise Biryani:" in body["reply"]
    assert "Hyderabadi Chicken Biryani" in body["reply"]
    assert "Veg Dum Biryani" in body["reply"]
    assert "Price: 249 (MRP: 299)" in body["reply"]


# ---------------------------------------------------------------------------
# B. Menu request without a restaurant name
# ---------------------------------------------------------------------------

def test_menu_endpoint_requires_restaurant_name(menu_client, monkeypatch):
    calls = []

    def fail_menu(restaurant_id=None):
        calls.append(restaurant_id)
        return _ok(SAMPLE_MENU)

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fail_menu)

    response = _post_chat(menu_client, "Show me the menu", customer_id="C_MENU_REQ")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "menu_restaurant_required"
    assert "Which restaurant's menu" in body["reply"]
    assert calls == []


# ---------------------------------------------------------------------------
# C. Empty restaurant menu
# ---------------------------------------------------------------------------

def test_menu_endpoint_empty_menu(menu_client, monkeypatch):
    _install_menu(monkeypatch, _ok({"menu": []}))

    response = _post_chat(
        menu_client,
        "Show me the Paradise Biryani menu",
        customer_id="C_MENU_EMPTY",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "menu_empty"
    assert "couldn't find any menu items" in body["reply"]
    assert "Paradise Biryani" in body["reply"]


# ---------------------------------------------------------------------------
# D. Restaurant not found
# ---------------------------------------------------------------------------

def test_menu_endpoint_restaurant_not_found(menu_client, monkeypatch):
    _install_menu(
        monkeypatch,
        _fail("http_error", "OXRO backend returned HTTP 404", status_code=404),
    )

    response = _post_chat(
        menu_client,
        "Show me the Paradise Biryani menu",
        customer_id="C_MENU_404",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "menu_restaurant_not_found"
    assert "couldn't find that restaurant" in body["reply"]


def test_menu_endpoint_unknown_name_not_found(menu_client, monkeypatch):
    calls = []

    def fail_menu(restaurant_id=None):
        calls.append(restaurant_id)
        return _ok(SAMPLE_MENU)

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fail_menu)

    response = _post_chat(
        menu_client,
        "Show me the menu for Atlantis Food Court",
        customer_id="C_MENU_UNK",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "menu_restaurant_not_found"
    assert calls == []


# ---------------------------------------------------------------------------
# E. Backend unavailable
# ---------------------------------------------------------------------------

def test_menu_endpoint_backend_unavailable(menu_client, monkeypatch):
    _install_menu(
        monkeypatch,
        _fail("connection_error", "Could not connect to OXRO backend"),
    )

    response = _post_chat(
        menu_client,
        "Show me the Paradise Biryani menu",
        customer_id="C_MENU_DOWN",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "menu_unavailable"
    assert "temporarily unavailable" in body["reply"]
    # Internal error details are never exposed.
    assert "connection_error" not in body["reply"]
    assert "Could not connect" not in body["reply"]


def test_menu_endpoint_list_backend_unavailable(menu_client, monkeypatch):
    _install_list(
        monkeypatch,
        _fail("timeout", "Request to OXRO backend timed out"),
    )

    response = _post_chat(
        menu_client,
        "Show me the Paradise Biryani menu",
        customer_id="C_MENU_TO",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "menu_unavailable"
    assert "temporarily unavailable" in body["reply"]


# ---------------------------------------------------------------------------
# F. No Gemini call for menu requests; session stores selection state
# ---------------------------------------------------------------------------

def test_menu_endpoint_stores_menu_state_and_no_gemini(menu_client):
    customer = "C_MENU_NOSESS"

    response = _post_chat(
        menu_client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "restaurant_menu"
    # Phase 7 Step 4: successful menu view stores selection state on a
    # session (created on demand) so the next turn can pick a dish.
    state = _session_manager().get_menu_state(customer)
    assert state is not None
    assert state["restaurant_id"] == "r_paradise"
    assert state["restaurant_name"] == "Paradise Biryani"
    assert len(state["items"]) == 2
    assert state["pending_dish"] is None
    assert state["confirmed_dish"] is None
    # Food context stays empty; no Gemini chat history.
    assert _session_manager().get_context(customer) == {
        "preferences": {},
        "matched_foods": [],
    }
    assert _session_manager().sessions[customer]["chat"].history == []
    # FakeGeminiClient.generate_content raises if ever invoked.
    assert getattr(main_module.app.state, "gemini_client", None) is not None


def test_menu_endpoint_does_not_touch_session_history(menu_client):
    customer = "C_MENU_HIST"

    # Seed a food session first.
    food = _post_chat(
        menu_client,
        "I want chicken biryani",
        customer_id=customer,
    )
    assert food.json()["status"] == "success"
    history_before = list(
        _session_manager().sessions[customer]["chat"].history
    )
    context_before = dict(_session_manager().get_context(customer))

    menu = _post_chat(
        menu_client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )
    assert menu.json()["status"] == "restaurant_menu"

    history_after = _session_manager().sessions[customer]["chat"].history
    context_after = _session_manager().get_context(customer)
    assert history_after == history_before
    assert context_after == context_before
    # Menu reply did not append a grounded Gemini message.
    assert len(history_after) == len(history_before)
    # Selection state was stored without altering food context.
    state = _session_manager().get_menu_state(customer)
    assert state is not None
    assert state["restaurant_id"] == "r_paradise"


# ---------------------------------------------------------------------------
# G. No interference with existing intents
# ---------------------------------------------------------------------------

def test_greeting_path_unchanged(menu_client):
    response = _post_chat(menu_client, "hello", customer_id="C_GREET")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "greeting"
    assert "Hello" in body["reply"]
    assert "C_GREET" not in _session_manager().sessions


def test_restaurant_discovery_path_unchanged(menu_client):
    response = _post_chat(menu_client, "Show me restaurants", customer_id="C_DISC")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_discovery"
    assert "Here are the restaurants:" in body["reply"]
    assert "Paradise Biryani" in body["reply"]
    assert "C_DISC" not in _session_manager().sessions


def test_food_path_unchanged(menu_client):
    response = _post_chat(
        menu_client,
        "I want chicken biryani",
        customer_id="C_FOOD",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert "C_FOOD" in _session_manager().sessions


def test_food_followup_still_uses_session(menu_client):
    _post_chat(menu_client, "I want chicken biryani", customer_id="C_FU")
    response = _post_chat(menu_client, "Make it spicy.", customer_id="C_FU")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert body["detected_preferences"]["taste"] == "spicy"


# ---------------------------------------------------------------------------
# H. Authentication still enforced
# ---------------------------------------------------------------------------

def test_menu_endpoint_requires_api_key(menu_client):
    response = menu_client.post(
        "/chat",
        json={"customer_id": "C_AUTH", "message": "Show me the menu"},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["status"] == "unauthorized"


def test_menu_endpoint_rejects_wrong_api_key(menu_client):
    response = menu_client.post(
        "/chat",
        headers={"X-API-Key": "wrong-key"},
        json={"customer_id": "C_AUTH2", "message": "Show me the menu"},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["status"] == "unauthorized"


# ---------------------------------------------------------------------------
# I. No menu backend call for non-menu requests
# ---------------------------------------------------------------------------

def test_food_request_does_not_call_get_restaurant_menu(menu_client, monkeypatch):
    def boom(restaurant_id=None):
        raise AssertionError(
            "get_restaurant_menu must not be called for dish-level requests"
        )

    monkeypatch.setattr(backend_client, "get_restaurant_menu", boom)

    response = _post_chat(
        menu_client,
        "I want chicken biryani",
        customer_id="C_NOMENU",
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_greeting_does_not_call_menu_backends(menu_client, monkeypatch):
    def boom_list():
        raise AssertionError("list_restaurants must not be called for greetings")

    def boom_menu(restaurant_id=None):
        raise AssertionError("get_restaurant_menu must not be called for greetings")

    monkeypatch.setattr(backend_client, "list_restaurants", boom_list)
    monkeypatch.setattr(backend_client, "get_restaurant_menu", boom_menu)

    response = _post_chat(menu_client, "hello", customer_id="C_G_NOMENU")

    assert response.status_code == 200
    assert response.json()["status"] == "greeting"


def test_discovery_does_not_call_get_restaurant_menu(menu_client, monkeypatch):
    seen = _install_menu_capture(
        monkeypatch,
        AssertionError("get_restaurant_menu must not be called for discovery"),
    )

    # list_restaurants remains mocked by fixture (success).
    response = _post_chat(menu_client, "Show me restaurants", customer_id="C_D_NOMENU")

    assert response.status_code == 200
    assert response.json()["status"] == "restaurant_discovery"
    assert seen["calls"] == []


# ---------------------------------------------------------------------------
# Network safety
# ---------------------------------------------------------------------------

def test_menu_endpoint_no_real_network(menu_client, monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("real network request attempted in sandbox")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok(SAMPLE_MENU))

    response = _post_chat(
        menu_client,
        "Show me the Paradise Biryani menu",
        customer_id="C_NET",
    )

    assert response.status_code == 200
    assert response.json()["status"] == "restaurant_menu"


def test_backend_base_url_unset(menu_client):
    assert os.getenv("OXRO_BACKEND_BASE_URL") is None
