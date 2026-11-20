"""
Dish selection unit tests (Phase 7 Step 4).

All session state is in-process SessionManager + FakeChat.
No Gemini, no network, no Supabase / Render / backend HTTP.
"""

from __future__ import annotations

import pytest

from ai.dish_selection import (
    extract_price_dish_query,
    handle_dish_selection,
    is_cancellation,
    is_confirmation,
    is_dish_pick_signal,
    is_price_request,
    match_dish,
)
from ai.session_manager import SessionManager


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeChat:
    def __init__(self, client=None):
        self.history = []

    def send_message(self, message):
        self.history.append(message)
        return FakeResponse(f"echo:{message}")


def make_manager(**kwargs):
    return SessionManager(
        client=None,
        chat_factory=lambda client: FakeChat(),
        **kwargs,
    )


SAMPLE_ITEMS = [
    {
        "id": 900001,
        "name": "Hyderabadi Chicken Biryani",
        "price": 249,
        "mrp": 299,
        "restaurantId": "r_paradise",
        "restaurantName": "Paradise Biryani",
        "veg": False,
    },
    {
        "id": 900003,
        "name": "Veg Dum Biryani",
        "price": 199,
        "mrp": 249,
        "restaurantId": "r_paradise",
        "restaurantName": "Paradise Biryani",
        "veg": True,
    },
    {
        "id": 900099,
        "name": "Egg Fusion Special",
        # no price field - price must never be invented
        "restaurantId": "r_paradise",
        "restaurantName": "Paradise Biryani",
        "egg": True,
    },
]


def menu_state(**overrides):
    state = {
        "restaurant_id": "r_paradise",
        "restaurant_name": "Paradise Biryani",
        "items": [dict(i) for i in SAMPLE_ITEMS],
        "undisplayed_names": [],
        "pending_dish": None,
        "confirmed_dish": None,
    }
    state.update(overrides)
    return state


def seed(customer_id, state, sm=None):
    sm = sm or make_manager()
    sm.set_menu_state(customer_id, state)
    return sm


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "yes",
        "Yes.",
        "yep",
        "OK",
        "Yes, select it",
        "yes, select it.",
        "Confirm",
        "That's right",
        "Go ahead",
        "Sure",
    ],
)
def test_is_confirmation_positive(message):
    assert is_confirmation(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "I want chicken biryani",
        "Make it spicy",
        "Show me the menu",
        "hello",
        "yes I also want pizza",  # not a bare confirm phrase
    ],
)
def test_is_confirmation_negative(message):
    assert is_confirmation(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "What's the price?",
        "How much is it?",
        "What is the price",
        "Show me the price",
        "price?",
    ],
)
def test_is_price_request_positive(message):
    assert is_price_request(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "I want Hyderabadi Chicken Biryani",
        "Make it spicy",
        "Show more",
        "hello",
    ],
)
def test_is_price_request_negative(message):
    assert is_price_request(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "I want Hyderabadi Chicken Biryani",
        "I'd like the Veg Dum Biryani",
        "Give me Egg Fusion Special",
        "Select Veg Dum Biryani",
        "Can I get Hyderabadi Chicken Biryani",
    ],
)
def test_is_dish_pick_signal_positive(message):
    assert is_dish_pick_signal(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Make it spicy",
        "Show more options",
        "hello",
        "Show me the Paradise Biryani menu",
        "What's the price?",
        "yes",
    ],
)
def test_is_dish_pick_signal_negative(message):
    assert is_dish_pick_signal(message) is False


# ---------------------------------------------------------------------------
# match_dish
# ---------------------------------------------------------------------------

def test_match_dish_whole_name_ok():
    item, code, candidates = match_dish(
        "I want Hyderabadi Chicken Biryani",
        SAMPLE_ITEMS,
    )
    assert code == "ok"
    assert candidates == []
    assert item["id"] == 900001


def test_match_dish_unique_phrase_fallback():
    item, code, _ = match_dish("I want chicken biryani", SAMPLE_ITEMS)
    assert code == "ok"
    assert item["id"] == 900001


def test_match_dish_ambiguous_phrase():
    item, code, candidates = match_dish("biryani", SAMPLE_ITEMS)
    assert item is None
    assert code == "ambiguous"
    assert len(candidates) == 2


def test_match_dish_not_found():
    item, code, candidates = match_dish("I want pizza", SAMPLE_ITEMS)
    assert item is None
    assert code == "not_found"
    assert candidates == []


def test_match_dish_required_empty():
    item, code, _ = match_dish("   ", SAMPLE_ITEMS)
    assert item is None
    assert code == "required"


def test_match_dish_does_not_match_inside_longer_token():
    # "Dum" must not whole-name-match "Veg Dum Biryani" as exact name;
    # phrase containment of "dum" hits only one dish -> unique ok.
    item, code, _ = match_dish("dum", SAMPLE_ITEMS)
    assert code == "ok"
    assert item["id"] == 900003


# ---------------------------------------------------------------------------
# handle_dish_selection - no menu state -> fall through
# ---------------------------------------------------------------------------

def test_no_menu_state_returns_none():
    sm = make_manager()
    sm.create_session("C1")
    assert handle_dish_selection("C1", "I want pizza", sm) is None
    assert handle_dish_selection("C1", "yes", sm) is None
    assert handle_dish_selection("C1", "What's the price?", sm) is None


def test_unknown_customer_returns_none():
    sm = make_manager()
    assert handle_dish_selection("NOBODY", "yes", sm) is None


# ---------------------------------------------------------------------------
# Propose selection
# ---------------------------------------------------------------------------

def test_propose_by_full_name():
    sm = seed("C1", menu_state())
    result = handle_dish_selection(
        "C1",
        "I want Hyderabadi Chicken Biryani",
        sm,
    )

    assert result is not None
    assert result["status"] == "dish_selection_proposed"
    assert "Hyderabadi Chicken Biryani" in result["reply"]
    assert "Paradise Biryani" in result["reply"]
    assert "Item ID: 900001" in result["reply"]
    assert "Price: 249 (MRP: 299)" in result["reply"]
    assert "Yes, select it" in result["reply"]
    assert result["matched_foods"] == []
    assert result["detected_preferences"] is None

    state = sm.get_menu_state("C1")
    assert state["pending_dish"]["id"] == 900001
    assert state["confirmed_dish"] is None


def test_propose_by_unique_phrase():
    sm = seed("C1", menu_state())
    result = handle_dish_selection("C1", "I want chicken biryani", sm)

    assert result["status"] == "dish_selection_proposed"
    assert "Hyderabadi Chicken Biryani" in result["reply"]
    assert sm.get_menu_state("C1")["pending_dish"]["id"] == 900001


def test_propose_bare_dish_name_without_pick_signal():
    sm = seed("C1", menu_state())
    result = handle_dish_selection("C1", "Veg Dum Biryani", sm)

    assert result["status"] == "dish_selection_proposed"
    assert sm.get_menu_state("C1")["pending_dish"]["id"] == 900003


def test_propose_clears_previous_confirmation():
    sm = seed(
        "C1",
        menu_state(
            confirmed_dish=dict(SAMPLE_ITEMS[1]),
            pending_dish=None,
        ),
    )
    result = handle_dish_selection(
        "C1",
        "I want Hyderabadi Chicken Biryani",
        sm,
    )

    assert result["status"] == "dish_selection_proposed"
    state = sm.get_menu_state("C1")
    assert state["pending_dish"]["id"] == 900001
    assert state["confirmed_dish"] is None


# ---------------------------------------------------------------------------
# Ambiguous / not on menu / name required
# ---------------------------------------------------------------------------

def test_ambiguous_lists_candidates():
    sm = seed("C1", menu_state())
    result = handle_dish_selection("C1", "I want biryani", sm)

    assert result["status"] == "dish_selection_ambiguous"
    assert "Hyderabadi Chicken Biryani" in result["reply"]
    assert "Veg Dum Biryani" in result["reply"]
    # Ambiguity must not store a pending dish.
    assert sm.get_menu_state("C1")["pending_dish"] is None


def test_dish_not_on_menu_with_pick_signal():
    sm = seed("C1", menu_state())
    result = handle_dish_selection("C1", "I want pizza", sm)

    assert result["status"] == "dish_not_on_menu"
    assert "Paradise Biryani" in result["reply"]
    assert "couldn't find that dish" in result["reply"]
    assert sm.get_menu_state("C1")["pending_dish"] is None


def test_dish_name_required_for_bare_i_want():
    sm = seed("C1", menu_state())
    result = handle_dish_selection("C1", "I want", sm)

    assert result["status"] == "dish_name_required"
    assert "full dish name" in result["reply"]


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------

def test_confirm_pending_dish():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[0])),
    )
    result = handle_dish_selection("C1", "Yes, select it", sm)

    assert result["status"] == "dish_selected"
    assert "Selected dish: Hyderabadi Chicken Biryani" in result["reply"]
    assert "Restaurant: Paradise Biryani" in result["reply"]
    assert "Item ID: 900001" in result["reply"]
    assert "Price: 249 (MRP: 299)" in result["reply"]
    assert "no order has been placed" in result["reply"]

    state = sm.get_menu_state("C1")
    assert state["confirmed_dish"]["id"] == 900001
    assert state["pending_dish"] is None


def test_confirm_without_pending_and_without_confirmed():
    sm = seed("C1", menu_state())
    result = handle_dish_selection("C1", "yes", sm)

    assert result["status"] == "menu_selection_required"
    assert "choose a dish" in result["reply"]


def test_confirm_again_restates_confirmed():
    sm = seed(
        "C1",
        menu_state(confirmed_dish=dict(SAMPLE_ITEMS[1])),
    )
    result = handle_dish_selection("C1", "yes", sm)

    assert result["status"] == "dish_selected"
    assert "Veg Dum Biryani" in result["reply"]


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def test_price_uses_pending_dish():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[0])),
    )
    result = handle_dish_selection("C1", "What's the price?", sm)

    assert result["status"] == "dish_price"
    assert "Hyderabadi Chicken Biryani" in result["reply"]
    assert "Price: 249" in result["reply"]


def test_price_uses_confirmed_dish():
    sm = seed(
        "C1",
        menu_state(confirmed_dish=dict(SAMPLE_ITEMS[1])),
    )
    result = handle_dish_selection("C1", "How much is it?", sm)

    assert result["status"] == "dish_price"
    assert "Veg Dum Biryani" in result["reply"]
    assert "199" in result["reply"]


def test_price_unavailable_when_not_listed():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[2])),
    )
    result = handle_dish_selection("C1", "What's the price?", sm)

    assert result["status"] == "dish_price_unavailable"
    assert "Egg Fusion Special" in result["reply"]
    assert "isn't listed" in result["reply"]
    # Never invent a number.
    assert "249" not in result["reply"]


def test_price_without_any_selected_dish():
    sm = seed("C1", menu_state())
    result = handle_dish_selection("C1", "What's the price?", sm)

    assert result["status"] == "menu_selection_required"


# ---------------------------------------------------------------------------
# Fall-through for non-selection messages while menu is active
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "Make it spicy",
        "Show more options",
        "Suggest something else",
        "What about vegetarian?",
    ],
)
def test_followups_fall_through_with_menu_active(message):
    sm = seed("C1", menu_state())
    assert handle_dish_selection("C1", message, sm) is None


def test_greeting_falls_through_with_menu_active():
    sm = seed("C1", menu_state())
    assert handle_dish_selection("C1", "hello", sm) is None


def test_menu_request_falls_through_with_menu_active():
    sm = seed("C1", menu_state())
    # New menu intent is handled by the menu short-circuit before this
    # module; if it reaches here it must not be treated as a pick.
    assert (
        handle_dish_selection("C1", "Show me the Saravana menu", sm)
        is None
    )


# ---------------------------------------------------------------------------
# Isolation / state integrity
# ---------------------------------------------------------------------------

def test_selection_isolated_between_customers():
    sm = make_manager()
    sm.set_menu_state("A", menu_state())
    sm.set_menu_state("B", menu_state())

    handle_dish_selection("A", "I want Hyderabadi Chicken Biryani", sm)

    assert sm.get_menu_state("A")["pending_dish"] is not None
    assert sm.get_menu_state("B")["pending_dish"] is None


def test_get_menu_state_returns_copy():
    sm = seed("C1", menu_state())
    state = sm.get_menu_state("C1")
    state["items"].clear()
    state["restaurant_name"] = "hacked"

    fresh = sm.get_menu_state("C1")
    assert len(fresh["items"]) == len(SAMPLE_ITEMS)
    assert fresh["restaurant_name"] == "Paradise Biryani"


def test_end_session_clears_menu_state():
    sm = seed("C1", menu_state())
    sm.end_session("C1")
    assert sm.get_menu_state("C1") is None
    assert handle_dish_selection("C1", "yes", sm) is None


def test_update_context_preserves_menu_state():
    sm = seed("C1", menu_state())
    sm.update_context(
        "C1",
        {"preferences": {"taste": "spicy"}, "matched_foods": ["X"]},
    )
    state = sm.get_menu_state("C1")
    assert state is not None
    assert state["restaurant_id"] == "r_paradise"
    assert sm.get_context("C1")["preferences"] == {"taste": "spicy"}


def test_create_session_clears_menu_state():
    sm = seed("C1", menu_state())
    sm.create_session("C1")
    assert sm.get_menu_state("C1") is None


# ---------------------------------------------------------------------------
# FIX 1 — cancellation / rejection / change selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "No",
        "no.",
        "Cancel",
        "cancel it",
        "I don't want it",
        "I do not want it",
        "Never mind",
        "No thanks",
        "Change my selection",
        "change selection",
        "Different dish",
        "Start over",
    ],
)
def test_is_cancellation_positive(message):
    assert is_cancellation(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Yes",
        "I want Hyderabadi Chicken Biryani",
        "What's the price?",
        "Make it spicy",
        "Show me the menu",
        "hello",
        "no more options please",  # not a bare cancel phrase
    ],
)
def test_is_cancellation_negative(message):
    assert is_cancellation(message) is False


def test_cancel_pending_with_no():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[0])),
    )
    result = handle_dish_selection("C1", "No", sm)

    assert result is not None
    assert result["status"] == "selection_cancelled"
    assert "Cancelled" in result["reply"]

    state = sm.get_menu_state("C1")
    assert state["pending_dish"] is None
    assert state["confirmed_dish"] is None


def test_cancelled_selection_cannot_be_confirmed_by_later_yes():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[0])),
    )
    cancel = handle_dish_selection("C1", "No", sm)
    assert cancel["status"] == "selection_cancelled"

    # Subsequent Yes must NOT confirm the cancelled dish.
    yes = handle_dish_selection("C1", "Yes", sm)
    assert yes is not None
    assert yes["status"] != "dish_selected"
    assert yes["status"] == "menu_selection_required"
    assert sm.get_menu_state("C1")["confirmed_dish"] is None
    assert sm.get_menu_state("C1")["pending_dish"] is None


@pytest.mark.parametrize(
    "message",
    [
        "Cancel",
        "I don't want it",
        "Never mind",
        "No",
    ],
)
def test_rejection_phrases_clear_pending(message):
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[1])),
    )
    result = handle_dish_selection("C1", message, sm)

    assert result["status"] == "selection_cancelled"
    assert sm.get_menu_state("C1")["pending_dish"] is None


def test_change_selection_clears_pending_and_allows_reselect():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[0])),
    )
    result = handle_dish_selection("C1", "Change my selection", sm)

    assert result["status"] == "selection_changed"
    assert sm.get_menu_state("C1")["pending_dish"] is None
    assert sm.get_menu_state("C1")["confirmed_dish"] is None

    # Customer can select another displayed item afterwards.
    again = handle_dish_selection("C1", "I want Veg Dum Biryani", sm)
    assert again["status"] == "dish_selection_proposed"
    assert sm.get_menu_state("C1")["pending_dish"]["id"] == 900003


def test_cancel_clears_confirmed_too():
    sm = seed(
        "C1",
        menu_state(confirmed_dish=dict(SAMPLE_ITEMS[0])),
    )
    result = handle_dish_selection("C1", "Cancel", sm)

    assert result["status"] == "selection_cancelled"
    state = sm.get_menu_state("C1")
    assert state["confirmed_dish"] is None
    assert state["pending_dish"] is None

    yes = handle_dish_selection("C1", "yes", sm)
    assert yes["status"] == "menu_selection_required"


def test_cancel_with_nothing_pending_falls_through():
    sm = seed("C1", menu_state())
    assert handle_dish_selection("C1", "No", sm) is None
    assert handle_dish_selection("C1", "Cancel", sm) is None


def test_no_is_not_treated_as_confirmation():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[0])),
    )
    assert is_confirmation("No") is False
    result = handle_dish_selection("C1", "No", sm)
    assert result["status"] == "selection_cancelled"


# ---------------------------------------------------------------------------
# FIX 2 — price question names the dish explicitly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message,expected_query",
    [
        ("What is the price of Veg Dum Biryani?", "veg dum biryani"),
        ("What's the price of Veg Dum Biryani", "veg dum biryani"),
        ("How much is Veg Dum Biryani?", "veg dum biryani"),
        ("Price of Egg Fusion Special", "egg fusion special"),
        ("What's the price?", ""),
        ("How much is it?", ""),
    ],
)
def test_extract_price_dish_query(message, expected_query):
    assert extract_price_dish_query(message) == expected_query


def test_price_of_other_dish_ignores_pending():
    # Pending is Chicken Biryani; customer asks price of Veg Dum Biryani.
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[0])),
    )
    result = handle_dish_selection(
        "C1",
        "What is the price of Veg Dum Biryani?",
        sm,
    )

    assert result["status"] == "dish_price"
    assert "Veg Dum Biryani" in result["reply"]
    assert "Price: 199" in result["reply"]
    # Never return the pending dish's own price line for a different dish.
    assert "Hyderabadi Chicken Biryani" not in result["reply"]
    assert "Price: 249" not in result["reply"]


def test_price_of_exact_displayed_dish_by_name():
    sm = seed("C1", menu_state())
    result = handle_dish_selection(
        "C1",
        "How much is Hyderabadi Chicken Biryani?",
        sm,
    )

    assert result["status"] == "dish_price"
    assert "Hyderabadi Chicken Biryani" in result["reply"]
    assert "249" in result["reply"]


def test_price_of_ambiguous_name_asks_to_clarify():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[0])),
    )
    result = handle_dish_selection(
        "C1",
        "What is the price of biryani?",
        sm,
    )

    assert result["status"] == "dish_selection_ambiguous"
    assert "Hyderabadi Chicken Biryani" in result["reply"]
    assert "Veg Dum Biryani" in result["reply"]
    # No price invented for an ambiguous request.
    assert "249" not in result["reply"]
    assert "199" not in result["reply"]


def test_price_of_missing_dish_not_on_menu():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[0])),
    )
    result = handle_dish_selection(
        "C1",
        "What is the price of Paneer Tikka Pizza?",
        sm,
    )

    assert result["status"] == "dish_not_on_menu"
    assert "249" not in result["reply"]


def test_price_of_dish_without_price_does_not_invent():
    sm = seed("C1", menu_state())
    result = handle_dish_selection(
        "C1",
        "What is the price of Egg Fusion Special?",
        sm,
    )

    assert result["status"] == "dish_price_unavailable"
    assert "Egg Fusion Special" in result["reply"]
    assert "isn't listed" in result["reply"]
    assert "249" not in result["reply"]
    assert "199" not in result["reply"]


def test_bare_price_still_uses_pending():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[1])),
    )
    result = handle_dish_selection("C1", "What's the price?", sm)

    assert result["status"] == "dish_price"
    assert "Veg Dum Biryani" in result["reply"]
    assert "199" in result["reply"]


# ---------------------------------------------------------------------------
# FIX 3 — only displayed menu items are selectable
# ---------------------------------------------------------------------------

def _displayed_and_hidden():
    """12 displayed items + 2 hidden beyond the display limit."""
    from ai.restaurant_menu import MAX_DISPLAYED_MENU_ITEMS

    displayed = [
        {
            "id": 1000 + i,
            "name": f"Displayed Dish {i}",
            "price": 100 + i,
            "restaurantId": "r_paradise",
            "restaurantName": "Paradise Biryani",
        }
        for i in range(MAX_DISPLAYED_MENU_ITEMS)
    ]
    hidden = [
        {
            "id": 2000 + i,
            "name": f"Hidden Special {i}",
            "price": 500 + i,
            "restaurantId": "r_paradise",
            "restaurantName": "Paradise Biryani",
        }
        for i in range(2)
    ]
    return displayed, hidden


def test_select_displayed_dish_ok():
    displayed, _ = _displayed_and_hidden()
    sm = seed("C1", menu_state(items=[dict(i) for i in displayed]))

    result = handle_dish_selection(
        "C1",
        "I want Displayed Dish 0",
        sm,
    )

    assert result["status"] == "dish_selection_proposed"
    assert "Displayed Dish 0" in result["reply"]
    assert "Item ID: 1000" in result["reply"]
    assert "Price: 100" in result["reply"]
    assert sm.get_menu_state("C1")["pending_dish"]["id"] == 1000


def test_undisplayed_dish_is_not_selectable():
    displayed, hidden = _displayed_and_hidden()
    sm = seed(
        "C1",
        menu_state(
            items=[dict(i) for i in displayed],
            undisplayed_names=[h["name"] for h in hidden],
        ),
    )

    result = handle_dish_selection(
        "C1",
        "I want Hidden Special 0",
        sm,
    )

    assert result["status"] == "dish_not_displayed"
    assert "Hidden Special 0" in result["reply"]
    assert "not among the currently displayed items" in result["reply"]
    # Must not silently select a different item.
    assert sm.get_menu_state("C1")["pending_dish"] is None
    # Hidden price must not be offered as a selection price.
    assert "500" not in result["reply"]


def test_undisplayed_bare_name_also_not_selectable():
    displayed, hidden = _displayed_and_hidden()
    sm = seed(
        "C1",
        menu_state(
            items=[dict(i) for i in displayed],
            undisplayed_names=[h["name"] for h in hidden],
        ),
    )

    result = handle_dish_selection("C1", "Hidden Special 1", sm)

    assert result["status"] == "dish_not_displayed"
    assert sm.get_menu_state("C1")["pending_dish"] is None


def test_confirm_displayed_pending_ok():
    displayed, _ = _displayed_and_hidden()
    sm = seed(
        "C1",
        menu_state(
            items=[dict(i) for i in displayed],
            pending_dish=dict(displayed[3]),
        ),
    )

    result = handle_dish_selection("C1", "Yes, select it", sm)

    assert result["status"] == "dish_selected"
    assert "Displayed Dish 3" in result["reply"]
    assert "Item ID: 1003" in result["reply"]
    state = sm.get_menu_state("C1")
    assert state["confirmed_dish"]["id"] == 1003
    assert state["pending_dish"] is None


def test_confirm_cancelled_selection_fails():
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(SAMPLE_ITEMS[0])),
    )
    handle_dish_selection("C1", "No", sm)

    result = handle_dish_selection("C1", "Yes, select it", sm)

    assert result["status"] == "menu_selection_required"
    assert sm.get_menu_state("C1")["confirmed_dish"] is None


def test_confirm_stale_pending_not_in_displayed_items_refused():
    # Pending points at a dish id that is no longer in displayed items.
    stale = {
        "id": 999999,
        "name": "No Longer Displayed",
        "price": 1,
    }
    sm = seed(
        "C1",
        menu_state(pending_dish=dict(stale)),
    )

    result = handle_dish_selection("C1", "Yes", sm)

    assert result["status"] == "dish_not_displayed"
    assert sm.get_menu_state("C1")["pending_dish"] is None
    assert sm.get_menu_state("C1")["confirmed_dish"] is None


def test_price_of_undisplayed_dish_not_selectable():
    displayed, hidden = _displayed_and_hidden()
    sm = seed(
        "C1",
        menu_state(
            items=[dict(i) for i in displayed],
            undisplayed_names=[h["name"] for h in hidden],
            pending_dish=dict(displayed[0]),
        ),
    )

    result = handle_dish_selection(
        "C1",
        "What is the price of Hidden Special 0?",
        sm,
    )

    assert result["status"] == "dish_not_displayed"
    assert "500" not in result["reply"]
    assert "100" not in result["reply"]
