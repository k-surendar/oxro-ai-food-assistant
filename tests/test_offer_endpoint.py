"""
Offer discovery endpoint integration tests (Phase 7 Step 5).

POST /chat offer short-circuit:
- all backend coupon calls are mocked
- no real HTTP, no Gemini, no Supabase / Render
- OXRO_BACKEND_BASE_URL is always unset
- no coupon application / cart / checkout / payment
"""

from __future__ import annotations

import os
import socket
import time

import pytest
from fastapi.testclient import TestClient

import backend_client
import main as main_module
from ai.session_manager import SessionManager
from rate_limiter import SlidingWindowRateLimiter

# Captured before any fixture patches list_coupons.
_ORIGINAL_LIST_COUPONS = backend_client.list_coupons

API_KEY = "test-sandbox-key"  # matches conftest default
H = {"X-API-Key": API_KEY}

RATE_LIMITED_BODY = {
    "detail": {
        "status": "rate_limited",
        "reason": "Too many requests. Please try again later."
    }
}

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

# Shape produced by backend GET /coupons (couponController.list).
SAMPLE_COUPONS = {
    "coupons": [
        {
            "code": "OXRO500",
            "title": "Flat Rs.500 OFF",
            "sub": "On orders above Rs.2,999",
            "off": 500,
            "percent": False,
            "cap": 0,
            "minOrder": 2999,
            "freeShip": False,
        },
        {
            "code": "WELCOME10",
            "title": "10% OFF up to Rs.300",
            "sub": "On orders above Rs.999",
            "off": 10,
            "percent": True,
            "cap": 300,
            "minOrder": 999,
            "freeShip": False,
        },
        {
            "code": "FREESHIP",
            "title": "Free Delivery",
            "sub": "No shipping fee above Rs.499",
            "off": 0,
            "percent": False,
            "cap": 0,
            "minOrder": 499,
            "freeShip": True,
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
        raise AssertionError("Gemini must not be called for offer requests")


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


def _install_coupons(monkeypatch, result):
    def fake_list():
        return result

    monkeypatch.setattr(backend_client, "list_coupons", fake_list)


def _install_list(monkeypatch, result):
    def fake_list():
        return result

    monkeypatch.setattr(backend_client, "list_restaurants", fake_list)


def _install_menu(monkeypatch, result):
    def fake_menu(restaurant_id=None):
        return result

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fake_menu)


def _post_chat(client, message, customer_id="C_OFFER", headers=None):
    return client.post(
        "/chat",
        headers=headers if headers is not None else H,
        json={"customer_id": customer_id, "message": message},
    )


@pytest.fixture
def offer_client(monkeypatch):
    """
    Bare TestClient (no lifespan): no real Gemini client is built.
    SessionManager + FakeChat installed so non-offer paths can run.
    OXRO_BACKEND_BASE_URL is always unset.
    """
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.delenv("OXRO_BACKEND_BASE_URL", raising=False)
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok(SAMPLE_MENU))
    _install_coupons(monkeypatch, _ok(SAMPLE_COUPONS))

    client = TestClient(main_module.app)
    main_module.app.state.session_manager = SessionManager(
        None,
        chat_factory=lambda cl: FakeChat(),
    )
    main_module.app.state.gemini_client = FakeGeminiClient()

    yield client

    main_module.app.state.session_manager = None
    main_module.app.state.gemini_client = None
    main_module.app.state.rate_limiter = None


def _session_manager():
    return main_module.app.state.session_manager


def _make_limiter(limit=2, window=60, clock=None):
    return SlidingWindowRateLimiter(
        limit=limit,
        window_seconds=window,
        clock=clock or time.monotonic,
    )


# ---------------------------------------------------------------------------
# A. Successful offer listing
# ---------------------------------------------------------------------------


def test_offer_endpoint_success(offer_client):
    response = _post_chat(offer_client, "Any offers?", customer_id="C_OK")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "offers_listed"
    assert body["customer_id"] == "C_OK"
    assert body["matched_foods"] == []
    assert body["detected_preferences"] is None
    assert "Here are the current offers:" in body["reply"]
    assert "OXRO500" in body["reply"]
    assert "WELCOME10" in body["reply"]
    assert "FREESHIP" in body["reply"]


def test_offer_endpoint_multiple_intents(offer_client):
    for message in (
        "Show me coupon codes.",
        "Are there any discounts?",
        "What promotions are available?",
        "Show available coupons.",
    ):
        response = _post_chat(
            offer_client, message, customer_id="C_INT"
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "offers_listed"
        assert "OXRO500" in body["reply"]


# ---------------------------------------------------------------------------
# B. Empty coupon list
# ---------------------------------------------------------------------------


def test_offer_endpoint_empty_list(offer_client, monkeypatch):
    _install_coupons(monkeypatch, _ok({"coupons": []}))

    response = _post_chat(offer_client, "Any offers?", customer_id="C_EMPTY")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "offers_empty"
    assert "no offers listed" in body["reply"].lower()
    assert "OXRO500" not in body["reply"]


# ---------------------------------------------------------------------------
# C. Backend unavailable / error / malformed
# ---------------------------------------------------------------------------


def test_offer_endpoint_backend_unavailable(offer_client, monkeypatch):
    _install_coupons(
        monkeypatch,
        _fail("connection_error", "Could not connect to OXRO backend"),
    )

    response = _post_chat(offer_client, "Any offers?", customer_id="C_DOWN")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "offers_unavailable"
    assert "temporarily unavailable" in body["reply"]
    # Internal error details are never exposed.
    assert "connection_error" not in body["reply"]
    assert "Could not connect" not in body["reply"]


def test_offer_endpoint_http_error(offer_client, monkeypatch):
    _install_coupons(
        monkeypatch,
        _fail("http_error", "OXRO backend returned HTTP 503", status_code=503),
    )

    response = _post_chat(offer_client, "Any offers?", customer_id="C_HTTP")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "offers_unavailable"
    assert "temporarily unavailable" in body["reply"]


def test_offer_endpoint_malformed_response(offer_client, monkeypatch):
    # ok=True but payload missing coupons list.
    _install_coupons(monkeypatch, _ok({"offers": []}))

    response = _post_chat(offer_client, "Any offers?", customer_id="C_BAD")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "offers_unavailable"
    assert "temporarily unavailable" in body["reply"]


def test_offer_endpoint_coupons_wrong_type(offer_client, monkeypatch):
    _install_coupons(monkeypatch, _ok({"coupons": {"code": "X"}}))

    response = _post_chat(offer_client, "Any offers?", customer_id="C_TYPE")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "offers_unavailable"


def test_offer_endpoint_base_url_unset_still_polite(
    offer_client, monkeypatch,
):
    # The fixture already unsets OXRO_BACKEND_BASE_URL; assert it and
    # restore the REAL list_coupons (fixture mock replaced it) so the
    # real client fails closed without any network attempt.
    assert os.getenv("OXRO_BACKEND_BASE_URL") is None

    def _blocked(*args, **kwargs):
        raise AssertionError("real network request attempted with no base URL")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(backend_client, "list_coupons", _ORIGINAL_LIST_COUPONS)

    response = _post_chat(
        offer_client, "Any offers?", customer_id="C_NOURL"
    )
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "offers_unavailable"
    assert "temporarily unavailable" in body["reply"]


# ---------------------------------------------------------------------------
# D. No Gemini, no session creation
# ---------------------------------------------------------------------------


def test_offer_endpoint_does_not_call_gemini(offer_client, monkeypatch):
    # FakeGeminiClient.generate_content raises AssertionError if used.
    # Also ensure FakeChat is never asked to send.
    def fail_chat(*args, **kwargs):
        raise AssertionError("Gemini chat must not be used for offers")

    monkeypatch.setattr(FakeChat, "send_message", fail_chat)

    response = _post_chat(offer_client, "Any offers?", customer_id="C_NOGEM")

    assert response.status_code == 200
    assert response.json()["status"] == "offers_listed"


def test_offer_endpoint_does_not_create_session(offer_client):
    customer = "C_NOSESS"

    response = _post_chat(offer_client, "Any offers?", customer_id=customer)

    assert response.status_code == 200
    assert response.json()["status"] == "offers_listed"
    assert customer not in _session_manager().sessions


def test_offer_endpoint_does_not_touch_existing_session_history(
    offer_client,
):
    customer = "C_HIST"

    # Seed a food session first.
    food = _post_chat(
        offer_client, "I want chicken biryani", customer_id=customer
    )
    assert food.json()["status"] == "success"
    history_before = list(_session_manager().sessions[customer]["chat"].history)
    context_before = dict(_session_manager().get_context(customer))
    menu_before = _session_manager().get_menu_state(customer)

    offer = _post_chat(offer_client, "Any offers?", customer_id=customer)
    assert offer.json()["status"] == "offers_listed"

    history_after = _session_manager().sessions[customer]["chat"].history
    context_after = _session_manager().get_context(customer)
    menu_after = _session_manager().get_menu_state(customer)
    assert history_after == history_before
    assert context_after == context_before
    assert menu_after == menu_before


# ---------------------------------------------------------------------------
# E. Authentication and rate limiting
# ---------------------------------------------------------------------------


def test_offer_endpoint_requires_api_key(offer_client):
    response = offer_client.post(
        "/chat",
        json={"customer_id": "C_AUTH", "message": "Any offers?"},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["status"] == "unauthorized"


def test_offer_endpoint_rate_limited(monkeypatch):
    """Offer requests count toward the per-customer rate limit."""
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.delenv("OXRO_BACKEND_BASE_URL", raising=False)
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok(SAMPLE_MENU))
    _install_coupons(monkeypatch, _ok(SAMPLE_COUPONS))

    with TestClient(main_module.app) as client:
        main_module.app.state.session_manager = SessionManager(
            None,
            chat_factory=lambda cl: FakeChat(),
        )
        main_module.app.state.gemini_client = FakeGeminiClient()
        main_module.app.state.rate_limiter = _make_limiter(limit=2)

        customer = "C_RL"
        first = _post_chat(client, "Any offers?", customer_id=customer)
        second = _post_chat(client, "Any offers?", customer_id=customer)
        third = _post_chat(client, "Any offers?", customer_id=customer)

        assert first.status_code == 200
        assert first.json()["status"] == "offers_listed"
        assert second.status_code == 200
        assert second.json()["status"] == "offers_listed"
        assert third.status_code == 429
        assert third.json() == RATE_LIMITED_BODY
        assert int(third.headers["Retry-After"]) >= 1

        main_module.app.state.session_manager = None
        main_module.app.state.gemini_client = None
        main_module.app.state.rate_limiter = None


# ---------------------------------------------------------------------------
# F. Existing intents retain priority
# ---------------------------------------------------------------------------


def test_greeting_path_unchanged(offer_client):
    response = _post_chat(offer_client, "hello", customer_id="C_GREET")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "greeting"
    assert "C_GREET" not in _session_manager().sessions


def test_restaurant_discovery_path_unchanged(offer_client):
    response = _post_chat(
        offer_client, "Show me restaurants", customer_id="C_DISC"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_discovery"
    assert "Paradise Biryani" in body["reply"]
    assert "C_DISC" not in _session_manager().sessions


def test_menu_path_unchanged(offer_client):
    response = _post_chat(
        offer_client,
        "Show me the Paradise Biryani menu",
        customer_id="C_MENU",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_menu"
    assert "Hyderabadi Chicken Biryani" in body["reply"]


def test_dish_selection_confirm_keeps_priority(offer_client):
    customer = "C_SEL"
    menu = _post_chat(
        offer_client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )
    assert menu.json()["status"] == "restaurant_menu"

    propose = _post_chat(
        offer_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )
    assert propose.json()["status"] == "dish_selection_proposed"

    # Confirmation must NOT be captured as an offer request.
    confirm = _post_chat(offer_client, "yes", customer_id=customer)
    assert confirm.status_code == 200
    body = confirm.json()
    assert body["status"] == "dish_selected"
    assert "Selected dish: Hyderabadi Chicken Biryani" in body["reply"]


def test_dish_price_keeps_priority(offer_client):
    customer = "C_PRICE"
    _post_chat(
        offer_client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )
    _post_chat(
        offer_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )

    price = _post_chat(
        offer_client, "What's the price?", customer_id=customer
    )
    assert price.status_code == 200
    body = price.json()
    assert body["status"] == "dish_price"
    assert "Price: 249" in body["reply"]


def test_dish_cancellation_keeps_priority(offer_client):
    customer = "C_CANCEL"
    _post_chat(
        offer_client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )
    _post_chat(
        offer_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )

    cancel = _post_chat(offer_client, "cancel", customer_id=customer)
    assert cancel.status_code == 200
    body = cancel.json()
    assert body["status"] == "selection_cancelled"
    assert _session_manager().get_menu_state(customer)["pending_dish"] is None


def test_offer_request_with_active_menu_still_lists_offers(offer_client):
    """Selection returns None for offer phrases; offer short-circuit runs."""
    customer = "C_MENU_OFFER"
    _post_chat(
        offer_client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )
    assert _session_manager().get_menu_state(customer) is not None

    response = _post_chat(offer_client, "Any offers?", customer_id=customer)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "offers_listed"
    assert "OXRO500" in body["reply"]


# ---------------------------------------------------------------------------
# G. Food queries continue through the existing food path
# ---------------------------------------------------------------------------


def test_food_query_uses_food_path_not_offers(offer_client, monkeypatch):
    def fail_coupons():
        raise AssertionError("list_coupons must not be called for food queries")

    monkeypatch.setattr(backend_client, "list_coupons", fail_coupons)

    response = _post_chat(
        offer_client, "I want chicken biryani", customer_id="C_FOOD"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert "C_FOOD" in _session_manager().sessions


def test_food_followup_still_uses_session(offer_client):
    _post_chat(
        offer_client, "I want chicken biryani", customer_id="C_FU"
    )
    response = _post_chat(
        offer_client, "Make it spicy.", customer_id="C_FU"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert body["detected_preferences"]["taste"] == "spicy"


def test_offer_does_not_call_restaurant_or_menu_apis(
    offer_client, monkeypatch,
):
    def boom_list():
        raise AssertionError("list_restaurants must not be called for offers")

    def boom_menu(restaurant_id=None):
        raise AssertionError(
            "get_restaurant_menu must not be called for offers"
        )

    monkeypatch.setattr(backend_client, "list_restaurants", boom_list)
    monkeypatch.setattr(backend_client, "get_restaurant_menu", boom_menu)
    _install_coupons(monkeypatch, _ok(SAMPLE_COUPONS))

    response = _post_chat(offer_client, "Any offers?", customer_id="C_X")

    assert response.status_code == 200
    assert response.json()["status"] == "offers_listed"


# ---------------------------------------------------------------------------
# H. Network safety
# ---------------------------------------------------------------------------


def test_offer_endpoint_no_real_network(offer_client, monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("real network request attempted in sandbox")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    _install_coupons(monkeypatch, _ok(SAMPLE_COUPONS))

    response = _post_chat(
        offer_client, "Any offers?", customer_id="C_NET"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "offers_listed"


def test_offer_endpoint_base_url_unset(offer_client):
    assert os.getenv("OXRO_BACKEND_BASE_URL") is None


def test_offer_module_source_has_no_production_host():
    from ai import offer_discovery

    source = open(offer_discovery.__file__, encoding="utf-8").read()
    assert "oxro-backend.onrender.com" not in source
    assert "render.com" not in source
