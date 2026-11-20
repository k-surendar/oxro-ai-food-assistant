"""
Dish selection endpoint integration tests (Phase 7 Step 4).

POST /chat selection short-circuit:
- all backend list/menu calls are mocked
- no real HTTP, no Gemini, no Supabase / Render
- OXRO_BACKEND_BASE_URL is always unset
- selection is confirmation only (no cart / order / payment)
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
    ]
}

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
        {
            "id": 900099,
            "name": "Egg Fusion Special",
            "brand": "Paradise Biryani",
            # no price - must never be invented
            "rating": None,
            "reviews": 50,
            "restaurantId": "r_paradise",
            "restaurantName": "Paradise Biryani",
            "veg": False,
            "egg": True,
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
        raise AssertionError("Gemini must not be called for selection requests")


def _ok(payload):
    return {
        "ok": True,
        "data": payload,
        "error": None,
        "message": None,
        "status_code": 200,
    }


def _install_list(monkeypatch, result):
    def fake_list():
        return result

    monkeypatch.setattr(backend_client, "list_restaurants", fake_list)


def _install_menu(monkeypatch, result):
    def fake_menu(restaurant_id=None):
        return result

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fake_menu)


def _post_chat(client, message, customer_id="C_SEL", headers=None):
    return client.post(
        "/chat",
        headers=headers if headers is not None else H,
        json={"customer_id": customer_id, "message": message},
    )


@pytest.fixture
def selection_client(monkeypatch):
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


def _view_menu(client, customer):
    response = _post_chat(
        client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "restaurant_menu"
    return response


# ---------------------------------------------------------------------------
# A. Full journey: menu -> propose -> confirm
# ---------------------------------------------------------------------------

def test_journey_menu_propose_confirm(selection_client):
    customer = "C_JOURNEY"

    menu = _view_menu(selection_client, customer)
    assert "Hyderabadi Chicken Biryani" in menu.json()["reply"]

    state = _session_manager().get_menu_state(customer)
    assert state is not None
    assert state["restaurant_id"] == "r_paradise"
    assert state["restaurant_name"] == "Paradise Biryani"
    assert len(state["items"]) == 3
    assert state["pending_dish"] is None
    assert state["confirmed_dish"] is None

    propose = _post_chat(
        selection_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )
    assert propose.status_code == 200
    body = propose.json()
    assert body["status"] == "dish_selection_proposed"
    assert "Hyderabadi Chicken Biryani" in body["reply"]
    assert "Item ID: 900001" in body["reply"]
    assert "Price: 249 (MRP: 299)" in body["reply"]
    assert body["matched_foods"] == []
    assert body["detected_preferences"] is None

    confirm = _post_chat(
        selection_client,
        "Yes, select it",
        customer_id=customer,
    )
    assert confirm.status_code == 200
    body = confirm.json()
    assert body["status"] == "dish_selected"
    assert "Selected dish: Hyderabadi Chicken Biryani" in body["reply"]
    assert "Restaurant: Paradise Biryani" in body["reply"]
    assert "Item ID: 900001" in body["reply"]
    assert "no order has been placed" in body["reply"]

    state = _session_manager().get_menu_state(customer)
    assert state["confirmed_dish"]["id"] == 900001
    assert state["pending_dish"] is None

    # Selection never touches Gemini chat history.
    history = _session_manager().sessions[customer]["chat"].history
    assert history == []


# ---------------------------------------------------------------------------
# B. Price after propose / confirm
# ---------------------------------------------------------------------------

def test_journey_menu_propose_price(selection_client):
    customer = "C_PRICE"
    _view_menu(selection_client, customer)
    _post_chat(
        selection_client,
        "I want Veg Dum Biryani",
        customer_id=customer,
    )

    response = _post_chat(
        selection_client,
        "What's the price?",
        customer_id=customer,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "dish_price"
    assert "Veg Dum Biryani" in body["reply"]
    assert "199" in body["reply"]


def test_price_unavailable_never_invents_number(selection_client):
    customer = "C_PRICE_NA"
    _view_menu(selection_client, customer)
    _post_chat(
        selection_client,
        "I want Egg Fusion Special",
        customer_id=customer,
    )

    response = _post_chat(
        selection_client,
        "How much is it?",
        customer_id=customer,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "dish_price_unavailable"
    assert "Egg Fusion Special" in body["reply"]
    assert "isn't listed" in body["reply"]


# ---------------------------------------------------------------------------
# C. Ambiguous / not on menu
# ---------------------------------------------------------------------------

def test_ambiguous_dish_lists_candidates(selection_client):
    customer = "C_AMBIG"
    _view_menu(selection_client, customer)

    response = _post_chat(
        selection_client,
        "I want biryani",
        customer_id=customer,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "dish_selection_ambiguous"
    assert "Hyderabadi Chicken Biryani" in body["reply"]
    assert "Veg Dum Biryani" in body["reply"]

    state = _session_manager().get_menu_state(customer)
    assert state["pending_dish"] is None


def test_dish_not_on_menu(selection_client):
    customer = "C_NOTON"
    _view_menu(selection_client, customer)

    response = _post_chat(
        selection_client,
        "I want paneer tikka pizza",
        customer_id=customer,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "dish_not_on_menu"
    assert "Paradise Biryani" in body["reply"]


# ---------------------------------------------------------------------------
# D. No menu state -> selection inactive, food path unchanged
# ---------------------------------------------------------------------------

def test_without_menu_food_path_unchanged(selection_client):
    response = _post_chat(
        selection_client,
        "I want chicken biryani",
        customer_id="C_FOOD_ONLY",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["matched_foods"] == ["Chicken Biryani"]
    assert _session_manager().get_menu_state("C_FOOD_ONLY") is None


def test_without_menu_confirm_does_not_hijack(selection_client):
    # Bare "yes" with no menu context must not become a selection reply.
    response = _post_chat(
        selection_client,
        "yes",
        customer_id="C_YES_NOMENU",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] != "dish_selected"
    assert body["status"] != "menu_selection_required"


def test_followup_after_menu_still_reaches_food_path(selection_client):
    customer = "C_FOLLOW"
    _view_menu(selection_client, customer)

    # Seed a food recommendation first (creates matched_foods context).
    food = _post_chat(
        selection_client,
        "I want chicken biryani",
        customer_id=customer,
    )
    # With menu active, "I want chicken biryani" is a menu pick
    # (unique phrase match) - so seed food BEFORE viewing menu instead.
    if food.json()["status"] != "success":
        # Re-do with a fresh customer: food first, then menu would wipe
        # menu state only - for this test we only need follow-up fallthrough.
        pass

    # Refinement with no pick signal must not be a selection status.
    # Use a customer whose menu is active but message is a pure follow-up.
    follow_customer = "C_FOLLOW2"
    _view_menu(selection_client, follow_customer)
    response = _post_chat(
        selection_client,
        "Make it spicy",
        customer_id=follow_customer,
    )
    assert response.status_code == 200
    body = response.json()
    # Falls through to food path (no Gemini-required selection status).
    assert body["status"] not in {
        "dish_selection_proposed",
        "dish_selected",
        "dish_not_on_menu",
        "dish_selection_ambiguous",
        "dish_price",
        "dish_price_unavailable",
        "menu_selection_required",
        "dish_name_required",
    }


# ---------------------------------------------------------------------------
# E. New menu replaces previous selection state
# ---------------------------------------------------------------------------

def test_new_menu_replaces_previous_state(selection_client, monkeypatch):
    customer = "C_SWITCH"
    _view_menu(selection_client, customer)
    _post_chat(
        selection_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )
    assert (
        _session_manager().get_menu_state(customer)["pending_dish"]
        is not None
    )

    # Different restaurant menu.
    _install_menu(
        monkeypatch,
        _ok(
            {
                "menu": [
                    {
                        "id": 910001,
                        "name": "Masala Dosa",
                        "price": 129,
                        "restaurantId": "r_saravana",
                        "restaurantName": "Saravana Bhavan",
                        "veg": True,
                    }
                ]
            }
        ),
    )

    response = _post_chat(
        selection_client,
        "Show me the Saravana Bhavan menu",
        customer_id=customer,
    )
    assert response.json()["status"] == "restaurant_menu"

    state = _session_manager().get_menu_state(customer)
    assert state["restaurant_id"] == "r_saravana"
    assert state["items"][0]["name"] == "Masala Dosa"
    assert state["pending_dish"] is None
    assert state["confirmed_dish"] is None


# ---------------------------------------------------------------------------
# F. No Gemini / no menu backend on selection turns
# ---------------------------------------------------------------------------

def test_selection_turns_do_not_call_gemini_or_menu_backend(
    selection_client,
    monkeypatch,
):
    customer = "C_NOGEM"
    _view_menu(selection_client, customer)

    def boom_menu(restaurant_id=None):
        raise AssertionError(
            "get_restaurant_menu must not be called for selection turns"
        )

    monkeypatch.setattr(backend_client, "get_restaurant_menu", boom_menu)

    propose = _post_chat(
        selection_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )
    confirm = _post_chat(
        selection_client,
        "yes",
        customer_id=customer,
    )
    price = _post_chat(
        selection_client,
        "What's the price?",
        customer_id=customer,
    )

    assert propose.json()["status"] == "dish_selection_proposed"
    assert confirm.json()["status"] == "dish_selected"
    assert price.json()["status"] == "dish_price"
    # FakeGeminiClient.generate_content raises if ever invoked.


def test_selection_creates_no_gemini_history(selection_client):
    customer = "C_HIST"
    _view_menu(selection_client, customer)
    _post_chat(
        selection_client,
        "I want Veg Dum Biryani",
        customer_id=customer,
    )
    _post_chat(selection_client, "Yes, select it", customer_id=customer)

    history = _session_manager().sessions[customer]["chat"].history
    assert history == []


# ---------------------------------------------------------------------------
# G. Isolation / auth / network safety
# ---------------------------------------------------------------------------

def test_selection_isolated_between_customers(selection_client):
    customer_a = "C_ISO_A"
    customer_b = "C_ISO_B"
    _view_menu(selection_client, customer_a)
    _view_menu(selection_client, customer_b)

    _post_chat(
        selection_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer_a,
    )

    state_a = _session_manager().get_menu_state(customer_a)
    state_b = _session_manager().get_menu_state(customer_b)
    assert state_a["pending_dish"] is not None
    assert state_b["pending_dish"] is None


def test_selection_requires_api_key(selection_client):
    response = selection_client.post(
        "/chat",
        json={"customer_id": "C_AUTH", "message": "yes"},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["status"] == "unauthorized"


def test_selection_endpoint_no_real_network(selection_client, monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("real network request attempted in sandbox")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok(SAMPLE_MENU))

    customer = "C_NET"
    _view_menu(selection_client, customer)
    response = _post_chat(
        selection_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "dish_selection_proposed"


def test_backend_base_url_unset(selection_client):
    assert os.getenv("OXRO_BACKEND_BASE_URL") is None


# ---------------------------------------------------------------------------
# H. Session capacity: menu still shown when storage fails
# ---------------------------------------------------------------------------

def test_menu_shown_even_when_session_capacity_full(
    selection_client,
    monkeypatch,
):
    customer = "C_CAP_SEL"
    sm = _session_manager()
    # Fill capacity with other customers.
    sm.max_sessions = 1
    sm.create_session("OTHER")

    response = _post_chat(
        selection_client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restaurant_menu"
    assert "Menu for Paradise Biryani:" in body["reply"]
    # Selection state was not stored (capacity full).
    assert sm.get_menu_state(customer) is None


# ---------------------------------------------------------------------------
# I. FIX 1 — cancel / change journey (endpoint)
# ---------------------------------------------------------------------------

def test_journey_select_no_cannot_later_confirm(selection_client):
    customer = "C_CANCEL"
    _view_menu(selection_client, customer)
    _post_chat(
        selection_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )

    cancel = _post_chat(selection_client, "No", customer_id=customer)
    assert cancel.status_code == 200
    body = cancel.json()
    assert body["status"] == "selection_cancelled"
    assert "Cancelled" in body["reply"]

    state = _session_manager().get_menu_state(customer)
    assert state["pending_dish"] is None
    assert state["confirmed_dish"] is None

    yes = _post_chat(selection_client, "Yes", customer_id=customer)
    assert yes.status_code == 200
    body = yes.json()
    assert body["status"] != "dish_selected"
    assert body["status"] == "menu_selection_required"
    assert _session_manager().get_menu_state(customer)["confirmed_dish"] is None


def test_journey_change_selection_then_pick_another(selection_client):
    customer = "C_CHANGE"
    _view_menu(selection_client, customer)
    _post_chat(
        selection_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )

    change = _post_chat(
        selection_client,
        "Change my selection",
        customer_id=customer,
    )
    assert change.json()["status"] == "selection_changed"
    assert _session_manager().get_menu_state(customer)["pending_dish"] is None

    again = _post_chat(
        selection_client,
        "I want Veg Dum Biryani",
        customer_id=customer,
    )
    assert again.json()["status"] == "dish_selection_proposed"
    assert "Veg Dum Biryani" in again.json()["reply"]
    assert _session_manager().get_menu_state(customer)["pending_dish"]["id"] == 900003


@pytest.mark.parametrize(
    "message,tag",
    [
        ("Cancel", "CANCEL"),
        ("I don't want it", "DONTWANT"),
        ("Never mind", "NEVERMIND"),
    ],
)
def test_rejection_phrases_cancel_pending(selection_client, message, tag):
    customer = f"C_REJ_{tag}"
    _view_menu(selection_client, customer)
    _post_chat(
        selection_client,
        "I want Veg Dum Biryani",
        customer_id=customer,
    )

    response = _post_chat(selection_client, message, customer_id=customer)
    assert response.json()["status"] == "selection_cancelled"
    assert _session_manager().get_menu_state(customer)["pending_dish"] is None


# ---------------------------------------------------------------------------
# J. FIX 2 — price names a different dish than pending (endpoint)
# ---------------------------------------------------------------------------

def test_price_question_for_different_dish_than_pending(selection_client):
    customer = "C_PRICE_OTHER"
    _view_menu(selection_client, customer)
    _post_chat(
        selection_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )

    response = _post_chat(
        selection_client,
        "What is the price of Veg Dum Biryani?",
        customer_id=customer,
    )
    body = response.json()
    assert body["status"] == "dish_price"
    assert "Veg Dum Biryani" in body["reply"]
    assert "Price: 199" in body["reply"]
    assert "Hyderabadi Chicken Biryani" not in body["reply"]
    assert "Price: 249" not in body["reply"]


def test_price_question_ambiguous_name_clarifies(selection_client):
    customer = "C_PRICE_AMB"
    _view_menu(selection_client, customer)

    response = _post_chat(
        selection_client,
        "What is the price of biryani?",
        customer_id=customer,
    )
    body = response.json()
    assert body["status"] == "dish_selection_ambiguous"
    assert "Hyderabadi Chicken Biryani" in body["reply"]
    assert "Veg Dum Biryani" in body["reply"]


def test_price_question_missing_dish_not_on_menu(selection_client):
    customer = "C_PRICE_MISS"
    _view_menu(selection_client, customer)

    response = _post_chat(
        selection_client,
        "What is the price of Paneer Tikka Pizza?",
        customer_id=customer,
    )
    assert response.json()["status"] == "dish_not_on_menu"


# ---------------------------------------------------------------------------
# K. FIX 3 — displayed vs undisplayed items (endpoint)
# ---------------------------------------------------------------------------

def _long_menu_payload():
    from ai.restaurant_menu import MAX_DISPLAYED_MENU_ITEMS

    dishes = [
        {
            "id": 1000 + i,
            "name": f"Displayed Dish {i}",
            "price": 100 + i,
            "restaurantId": "r_paradise",
            "restaurantName": "Paradise Biryani",
            "veg": True,
        }
        for i in range(MAX_DISPLAYED_MENU_ITEMS)
    ]
    dishes.append(
        {
            "id": 2000,
            "name": "Hidden Special 0",
            "price": 500,
            "restaurantId": "r_paradise",
            "restaurantName": "Paradise Biryani",
            "veg": False,
        }
    )
    return {"menu": dishes}


def test_endpoint_only_displayed_items_stored_and_selectable(
    selection_client,
    monkeypatch,
):
    from ai.restaurant_menu import MAX_DISPLAYED_MENU_ITEMS

    customer = "C_DISPLAY"
    _install_menu(monkeypatch, _ok(_long_menu_payload()))
    menu = _post_chat(
        selection_client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )
    assert menu.json()["status"] == "restaurant_menu"
    # Truncated reply does not show the hidden dish.
    assert "Hidden Special 0" not in menu.json()["reply"]

    state = _session_manager().get_menu_state(customer)
    assert len(state["items"]) == MAX_DISPLAYED_MENU_ITEMS
    assert state["undisplayed_names"] == ["Hidden Special 0"]

    # Displayed dish can be selected and confirmed with full identity.
    propose = _post_chat(
        selection_client,
        "I want Displayed Dish 0",
        customer_id=customer,
    )
    body = propose.json()
    assert body["status"] == "dish_selection_proposed"
    assert "Displayed Dish 0" in body["reply"]
    assert "Item ID: 1000" in body["reply"]
    assert "Paradise Biryani" in body["reply"]
    assert "Price: 100" in body["reply"]

    confirm = _post_chat(selection_client, "Yes, select it", customer_id=customer)
    body = confirm.json()
    assert body["status"] == "dish_selected"
    assert "Displayed Dish 0" in body["reply"]
    assert "Item ID: 1000" in body["reply"]
    assert "Restaurant: Paradise Biryani" in body["reply"]
    assert "Price: 100" in body["reply"]


def test_endpoint_undisplayed_dish_not_selectable(
    selection_client,
    monkeypatch,
):
    customer = "C_HIDDEN"
    _install_menu(monkeypatch, _ok(_long_menu_payload()))
    _post_chat(
        selection_client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )

    response = _post_chat(
        selection_client,
        "I want Hidden Special 0",
        customer_id=customer,
    )
    body = response.json()
    assert body["status"] == "dish_not_displayed"
    assert "Hidden Special 0" in body["reply"]
    assert "not among the currently displayed items" in body["reply"]
    assert "500" not in body["reply"]

    state = _session_manager().get_menu_state(customer)
    assert state["pending_dish"] is None


def test_endpoint_confirm_after_cancel_does_not_confirm(
    selection_client,
):
    customer = "C_CANCEL_CONFIRM"
    _view_menu(selection_client, customer)
    _post_chat(
        selection_client,
        "I want Hyderabadi Chicken Biryani",
        customer_id=customer,
    )
    _post_chat(selection_client, "No", customer_id=customer)

    response = _post_chat(selection_client, "Yes, select it", customer_id=customer)
    body = response.json()
    assert body["status"] == "menu_selection_required"
    assert _session_manager().get_menu_state(customer)["confirmed_dish"] is None


def test_endpoint_undisplayed_price_not_offered(
    selection_client,
    monkeypatch,
):
    customer = "C_HIDDEN_PRICE"
    _install_menu(monkeypatch, _ok(_long_menu_payload()))
    _post_chat(
        selection_client,
        "Show me the Paradise Biryani menu",
        customer_id=customer,
    )

    response = _post_chat(
        selection_client,
        "What is the price of Hidden Special 0?",
        customer_id=customer,
    )
    body = response.json()
    assert body["status"] == "dish_not_displayed"
    assert "500" not in body["reply"]
