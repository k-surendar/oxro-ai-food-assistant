"""
Restaurant menu discovery tests (Phase 7 - menu step).

All backend interaction is mocked (monkeypatched list_restaurants and
get_restaurant_menu):
- no real HTTP requests
- no Gemini calls
- no Supabase / Render connections
"""

from __future__ import annotations

import os
import socket

import pytest

import backend_client
from ai.restaurant_menu import (
    MAX_DISPLAYED_MENU_ITEMS,
    MENU_AMBIGUOUS_REPLY,
    MENU_NOT_FOUND_REPLY,
    MENU_REQUIRED_REPLY,
    MENU_UNAVAILABLE_REPLY,
    is_menu_request,
    restaurant_menu_reply,
)

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

# Shape produced by backend GET /restaurants/{id}/menu
# (restaurantController.menu + catalogStore dish objects).
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
            "icon": 58460,
            "colors": [4294967295, 4294902784],
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
            "price": 159,
            "mrp": 159,
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
    """Replace backend_client.list_restaurants with a fixed result."""

    def fake_list():
        return result

    monkeypatch.setattr(backend_client, "list_restaurants", fake_list)


def _install_menu(monkeypatch, result):
    """Replace backend_client.get_restaurant_menu with a fixed result."""

    def fake_menu(restaurant_id=None):
        return result

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fake_menu)


def _install_menu_capture(monkeypatch, result):
    """Mock menu fetch and capture the restaurant_id that was requested."""
    seen = {"restaurant_id": None}

    def fake_menu(restaurant_id=None):
        seen["restaurant_id"] = restaurant_id
        return result

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fake_menu)
    return seen


@pytest.fixture(autouse=True)
def _no_backend_base_url(monkeypatch):
    """Fail-closed: never allow a real backend URL in these tests."""
    monkeypatch.delenv("OXRO_BACKEND_BASE_URL", raising=False)


# ---------------------------------------------------------------------------
# Intent detection - positive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "Show me the menu.",
        "Show me the Paradise Biryani menu.",
        "What is available on the menu at Paradise Biryani?",
        "Give me the menu for Saravana Bhavan.",
        "I want to see the restaurant menu.",
        "Paradise Biryani menu.",
        "menu",
        "List the menu for r_paradise",
        "Fetch the Domino's Pizza menu",
        "What's on the menu?",
    ],
)
def test_is_menu_request_positive(message):
    assert is_menu_request(message) is True


# ---------------------------------------------------------------------------
# Intent detection - negative (greetings, discovery, ordinary food)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "Hi.",
        "hello",
        "Show me restaurants.",
        "Find restaurants near me.",
        "Which restaurants are available?",
        "I want chicken biryani.",
        "I want pizza.",
        "Recommend some vegetarian food.",
        "Make it spicy.",
        "",
        None,
    ],
)
def test_is_menu_request_negative(message):
    assert is_menu_request(message) is False


# ---------------------------------------------------------------------------
# Restaurant resolution
# ---------------------------------------------------------------------------

def test_exact_restaurant_name_resolution(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu_capture(monkeypatch, _ok(SAMPLE_MENU))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "restaurant_menu"
    # Name resolution used list_restaurants then fetched that id only via
    # the mocked get_restaurant_menu - assert reply header has the name.
    assert "Menu for Paradise Biryani:" in result["reply"]


def test_explicit_restaurant_id_resolution(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    seen = _install_menu_capture(monkeypatch, _ok(SAMPLE_MENU))

    result = restaurant_menu_reply("Show me the menu for r_paradise")

    assert result["status"] == "restaurant_menu"
    assert seen["restaurant_id"] == "r_paradise"
    assert "Menu for Paradise Biryani:" in result["reply"]


def test_restaurant_name_is_case_insensitive(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    seen = _install_menu_capture(monkeypatch, _ok(SAMPLE_MENU))

    result = restaurant_menu_reply("show me the paradise biryani MENU")

    assert result["status"] == "restaurant_menu"
    assert seen["restaurant_id"] == "r_paradise"


def test_missing_restaurant_name_asks_to_specify(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    calls = []

    def fail_menu(restaurant_id=None):
        calls.append(restaurant_id)
        return _ok(SAMPLE_MENU)

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fail_menu)

    result = restaurant_menu_reply("Show me the menu")

    assert result["status"] == "menu_restaurant_required"
    assert result["reply"] == MENU_REQUIRED_REPLY
    assert calls == []


def test_ambiguous_restaurant_name_asks_to_specify(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    calls = []

    def fail_menu(restaurant_id=None):
        calls.append(restaurant_id)
        return _ok(SAMPLE_MENU)

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fail_menu)

    result = restaurant_menu_reply(
        "Show me the menu for Paradise Biryani and Saravana Bhavan"
    )

    assert result["status"] == "menu_restaurant_ambiguous"
    assert result["reply"] == MENU_AMBIGUOUS_REPLY
    assert calls == []


def test_unknown_restaurant_name_not_found(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    calls = []

    def fail_menu(restaurant_id=None):
        calls.append(restaurant_id)
        return _ok(SAMPLE_MENU)

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fail_menu)

    result = restaurant_menu_reply("Show me the menu for Atlantis Food Court")

    assert result["status"] == "menu_restaurant_not_found"
    assert result["reply"] == MENU_NOT_FOUND_REPLY
    assert calls == []


def test_empty_user_message_asks_to_specify(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))

    result = restaurant_menu_reply(None)

    assert result["status"] == "menu_restaurant_required"
    assert result["reply"] == MENU_REQUIRED_REPLY


# ---------------------------------------------------------------------------
# FIX 1 - whole-name matching (no partial / longer-name false positives)
# ---------------------------------------------------------------------------

def test_show_paradise_biryani_menu_without_the_word_the(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    seen = _install_menu_capture(monkeypatch, _ok(SAMPLE_MENU))

    result = restaurant_menu_reply("Show me Paradise Biryani menu")

    assert result["status"] == "restaurant_menu"
    assert seen["restaurant_id"] == "r_paradise"
    assert "Menu for Paradise Biryani:" in result["reply"]


def test_longer_restaurant_name_does_not_match_shorter_name(monkeypatch):
    """'Paradise Biryani Express' must not select 'Paradise Biryani'."""
    restaurants = {
        "restaurants": [
            {
                "id": "r_paradise",
                "name": "Paradise Biryani",
                "cuisines": ["Biryani"],
                "rating": 4.5,
                "reviews": 3200,
                "distanceLabel": "2.1 km",
                "etaMinutes": 30,
            },
            {
                "id": "r_paradise_express",
                "name": "Paradise Biryani Express",
                "cuisines": ["Biryani", "Fast Food"],
                "rating": 4.1,
                "reviews": 900,
                "distanceLabel": "3.0 km",
                "etaMinutes": 25,
            },
        ]
    }
    _install_list(monkeypatch, _ok(restaurants))
    seen = _install_menu_capture(monkeypatch, _ok(SAMPLE_MENU))

    result = restaurant_menu_reply("Show me Paradise Biryani Express menu")

    assert result["status"] == "restaurant_menu"
    # Whole-name match: only the Express restaurant is selected.
    assert seen["restaurant_id"] == "r_paradise_express"
    assert "Menu for Paradise Biryani Express:" in result["reply"]
    assert "Menu for Paradise Biryani:" not in result["reply"]


def test_partial_name_not_matched_when_only_shorter_exists(monkeypatch):
    """
    Message names a longer phrase; only the shorter restaurant exists.
    Must NOT select the shorter name (no partial-name guessing).
    """
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    calls = []

    def fail_menu(restaurant_id=None):
        calls.append(restaurant_id)
        return _ok(SAMPLE_MENU)

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fail_menu)

    result = restaurant_menu_reply(
        "Show me Paradise Biryani Express menu"
    )

    assert result["status"] == "menu_restaurant_not_found"
    assert result["reply"] == MENU_NOT_FOUND_REPLY
    assert calls == []


def test_name_match_is_case_insensitive_with_punctuation(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    seen = _install_menu_capture(monkeypatch, _ok(SAMPLE_MENU))

    result = restaurant_menu_reply(
        "Show me the PARADISE BIRYANI menu, please!"
    )

    assert result["status"] == "restaurant_menu"
    assert seen["restaurant_id"] == "r_paradise"


def test_multiple_whole_name_matches_are_ambiguous(monkeypatch):
    """Two distinct whole names in one message -> ambiguous, no guess."""
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    calls = []

    def fail_menu(restaurant_id=None):
        calls.append(restaurant_id)
        return _ok(SAMPLE_MENU)

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fail_menu)

    result = restaurant_menu_reply(
        "Show me Paradise Biryani and Saravana Bhavan menu"
    )

    assert result["status"] == "menu_restaurant_ambiguous"
    assert calls == []


def test_name_not_matched_inside_longer_token(monkeypatch):
    """'Biryani' alone must not match 'Paradise Biryani' via token glue."""
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    calls = []

    def fail_menu(restaurant_id=None):
        calls.append(restaurant_id)
        return _ok(SAMPLE_MENU)

    monkeypatch.setattr(backend_client, "get_restaurant_menu", fail_menu)

    # Single-word residual that is not a full restaurant name.
    result = restaurant_menu_reply("Show me Biryani menu")

    assert result["status"] == "menu_restaurant_not_found"
    assert calls == []


# ---------------------------------------------------------------------------
# Menu retrieval - success / empty / not found / backend failures
# ---------------------------------------------------------------------------

def test_successful_menu_retrieval(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok(SAMPLE_MENU))

    result = restaurant_menu_reply("Give me the menu for Paradise Biryani")

    assert result["status"] == "restaurant_menu"
    reply = result["reply"]
    assert "Menu for Paradise Biryani:" in reply
    assert "Hyderabadi Chicken Biryani" in reply
    assert "Veg Dum Biryani" in reply
    assert "Egg Fusion Special" in reply
    assert result["matched_foods"] == []
    assert result["detected_preferences"] is None


def test_empty_menu_response(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok({"menu": []}))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_empty"
    assert "Paradise Biryani" in result["reply"]
    assert "couldn't find any menu items" in result["reply"]


def test_restaurant_not_found_http_404(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _fail("http_error", "OXRO backend returned HTTP 404", status_code=404),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_restaurant_not_found"
    assert result["reply"] == MENU_NOT_FOUND_REPLY


def test_backend_connection_failure(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _fail("connection_error", "Could not connect to OXRO backend"),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"
    assert result["reply"] == MENU_UNAVAILABLE_REPLY


def test_backend_list_connection_failure(monkeypatch):
    _install_list(
        monkeypatch,
        _fail("connection_error", "Could not connect to OXRO backend"),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"
    assert result["reply"] == MENU_UNAVAILABLE_REPLY


def test_backend_timeout_on_menu(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _fail("timeout", "Request to OXRO backend timed out"),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"
    assert result["reply"] == MENU_UNAVAILABLE_REPLY


# ---------------------------------------------------------------------------
# Invalid / incomplete menu data
# ---------------------------------------------------------------------------

def test_menu_data_not_a_dict(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok(["not", "a", "dict"]))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"
    assert result["reply"] == MENU_UNAVAILABLE_REPLY


def test_menu_list_missing(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok({"items": []}))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"


def test_menu_list_not_a_list(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok({"menu": "oops"}))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"


def test_menu_items_all_invalid(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok({"menu": ["bad", 123, None]}))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"


# ---------------------------------------------------------------------------
# FIX 2 - dish records without a valid non-empty name
# ---------------------------------------------------------------------------

def test_menu_items_missing_name_are_skipped(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _ok(
            {
                "menu": [
                    {"price": 10, "rating": 4.0},
                    {"name": "Named Dish", "price": 12},
                ]
            }
        ),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "restaurant_menu"
    assert "Named Dish" in result["reply"]
    # Unnamed record is not presented as a dish (no fake "Dish" label).
    assert "1. Dish" not in result["reply"]
    assert "Price: 10" not in result["reply"]
    assert "Rating: 4.0" not in result["reply"]


def test_menu_items_empty_or_whitespace_name_are_skipped(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _ok(
            {
                "menu": [
                    {"name": "", "price": 5},
                    {"name": "   ", "price": 6},
                    {"name": None, "price": 7},
                    {"name": 42, "price": 8},
                    {"name": "Real Item", "price": 9},
                ]
            }
        ),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "restaurant_menu"
    assert "Real Item" in result["reply"]
    assert "Price: 5" not in result["reply"]
    assert "Price: 6" not in result["reply"]
    assert "Price: 7" not in result["reply"]
    assert "Price: 8" not in result["reply"]
    assert "Price: 9" in result["reply"]
    assert "Dish" not in result["reply"].replace("Real Item", "")


def test_menu_all_unnamed_records_returns_unavailable(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _ok(
            {
                "menu": [
                    {"price": 10},
                    {"name": "", "price": 11},
                    {"name": "   "},
                    {"name": None},
                ]
            }
        ),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"
    assert result["reply"] == MENU_UNAVAILABLE_REPLY
    assert "Dish" not in result["reply"]


def test_menu_mixed_valid_and_unnamed_keeps_only_named(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _ok(
            {
                "menu": [
                    {"name": "Valid A", "price": 10},
                    {"price": 11},
                    {"name": "", "price": 12},
                    "not-a-dict",
                    {"name": "Valid B", "price": 13},
                ]
            }
        ),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "restaurant_menu"
    reply = result["reply"]
    assert "Valid A" in reply
    assert "Valid B" in reply
    assert "Price: 10" in reply
    assert "Price: 13" in reply
    assert "Price: 11" not in reply
    assert "Price: 12" not in reply
    assert "not-a-dict" not in reply
    # Numbered list only contains named dishes (1 and 2).
    assert "1. Valid A" in reply
    assert "2. Valid B" in reply


def test_menu_partial_invalid_items_keeps_valid_ones(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _ok(
            {
                "menu": [
                    {"name": "Keep Me", "price": 10},
                    "bad-item",
                    {"name": "Also Keep", "price": 20},
                ]
            }
        ),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "restaurant_menu"
    assert "Keep Me" in result["reply"]
    assert "Also Keep" in result["reply"]
    assert "bad-item" not in result["reply"]


def test_restaurant_list_structure_invalid(monkeypatch):
    _install_list(monkeypatch, _ok({"not_restaurants": []}))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"
    assert result["reply"] == MENU_UNAVAILABLE_REPLY


# ---------------------------------------------------------------------------
# Formatting - optional fields, no invented data, UI fields hidden
# ---------------------------------------------------------------------------

def test_formatting_price_mrp_rating_and_labels(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok(SAMPLE_MENU))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")
    reply = result["reply"]

    # Price + useful MRP (mrp > price)
    assert "Price: 249 (MRP: 299)" in reply
    assert "Price: 199 (MRP: 249)" in reply
    # Rating only when present
    assert "Rating: 4.3" in reply
    # Veg / egg only when explicitly true
    assert "Vegetarian" in reply
    assert "Egg" in reply
    # UI-only fields never shown
    assert "icon" not in reply.lower()
    assert "colors" not in reply.lower()
    assert "58460" not in reply


def test_formatting_omits_missing_price_and_rating(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _ok(
            {
                "menu": [
                    {
                        "id": 1,
                        "name": "Mystery Dish",
                        "veg": False,
                        "egg": False,
                    }
                ]
            }
        ),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")
    reply = result["reply"]

    assert "Mystery Dish" in reply
    assert "Price:" not in reply
    assert "MRP:" not in reply
    assert "Rating:" not in reply
    assert "Vegetarian" not in reply
    assert "Egg" not in reply


def test_formatting_mrp_equal_to_price_not_shown(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _ok(
            {
                "menu": [
                    {
                        "name": "Same Price Item",
                        "price": 159,
                        "mrp": 159,
                        "rating": 4.0,
                    }
                ]
            }
        ),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")
    reply = result["reply"]

    assert "Price: 159" in reply
    assert "MRP:" not in reply  # not useful when equal to price
    assert "Rating: 4.0" in reply


def test_formatting_does_not_invent_availability_or_discounts(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok(SAMPLE_MENU))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")
    reply = result["reply"].lower()

    for banned in (
        "available",
        "in stock",
        "out of stock",
        "discount",
        "% off",
        "category",
        "ingredient",
        "on sale",
    ):
        assert banned not in reply


def test_formatting_hides_ui_only_icon_and_colors_values(monkeypatch):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(
        monkeypatch,
        _ok(
            {
                "menu": [
                    {
                        "name": "Icon Only Dish",
                        "price": 10,
                        "icon": 999888,
                        "colors": [1, 2, 3],
                    }
                ]
            }
        ),
    )

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")
    reply = result["reply"]

    assert "Icon Only Dish" in reply
    assert "999888" not in reply
    assert "[1, 2, 3]" not in reply


def test_large_menu_is_truncated_with_omitted_count(monkeypatch):
    dishes = [
        {
            "id": i,
            "name": f"Dish {i}",
            "price": 10 + i,
            "restaurantId": "r_paradise",
            "restaurantName": "Paradise Biryani",
        }
        for i in range(MAX_DISPLAYED_MENU_ITEMS + 5)
    ]
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok({"menu": dishes}))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")
    reply = result["reply"]

    assert result["status"] == "restaurant_menu"
    assert "Dish 0" in reply
    # Shown items are indices 0..MAX-1; index MAX is omitted.
    assert f"Dish {MAX_DISPLAYED_MENU_ITEMS - 1}" in reply
    assert f"Dish {MAX_DISPLAYED_MENU_ITEMS}" not in reply
    assert "... and 5 more item(s) not shown." in reply

    # FIX 3: selection context keeps only DISPLAYED items selectable;
    # undisplayed names are stored solely for "not displayed" messaging.
    context = result["menu_context"]
    assert len(context["items"]) == MAX_DISPLAYED_MENU_ITEMS
    assert context["items"][0]["name"] == "Dish 0"
    assert context["items"][-1]["name"] == f"Dish {MAX_DISPLAYED_MENU_ITEMS - 1}"
    assert context["undisplayed_names"] == [
        f"Dish {i}"
        for i in range(MAX_DISPLAYED_MENU_ITEMS, MAX_DISPLAYED_MENU_ITEMS + 5)
    ]
    # Hidden dishes never appear as selectable items.
    assert all(
        item["name"] != f"Dish {MAX_DISPLAYED_MENU_ITEMS}"
        for item in context["items"]
    )


# ---------------------------------------------------------------------------
# Unexpected exception handling (type-only logging, no sensitive leak)
# ---------------------------------------------------------------------------

def test_unexpected_exception_from_list_is_contained(monkeypatch, caplog):
    def boom():
        raise RuntimeError("super-secret-backend-url-should-not-leak")

    monkeypatch.setattr(backend_client, "list_restaurants", boom)

    with caplog.at_level("ERROR", logger="oxro.restaurant_menu"):
        result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"
    assert result["reply"] == MENU_UNAVAILABLE_REPLY
    assert "super-secret" not in result["reply"]
    assert all("super-secret" not in r.message for r in caplog.records)
    assert any("RuntimeError" in r.message for r in caplog.records)


def test_unexpected_exception_from_menu_is_contained(monkeypatch, caplog):
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))

    def boom(restaurant_id=None):
        raise ValueError("token=abc123-should-not-leak")

    monkeypatch.setattr(backend_client, "get_restaurant_menu", boom)

    with caplog.at_level("ERROR", logger="oxro.restaurant_menu"):
        result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "menu_unavailable"
    assert result["reply"] == MENU_UNAVAILABLE_REPLY
    assert "token=" not in result["reply"]
    assert all("token=" not in r.message for r in caplog.records)
    assert any("ValueError" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Network / environment safety guards
# ---------------------------------------------------------------------------

def test_oxro_backend_base_url_is_unset():
    assert os.getenv("OXRO_BACKEND_BASE_URL") is None


def test_no_real_network_requests_during_menu_flow(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("real network request attempted in sandbox")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    _install_list(monkeypatch, _ok(SAMPLE_RESTAURANTS))
    _install_menu(monkeypatch, _ok(SAMPLE_MENU))

    result = restaurant_menu_reply("Show me the Paradise Biryani menu")

    assert result["status"] == "restaurant_menu"
