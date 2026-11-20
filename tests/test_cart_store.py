"""
Local mock cart store tests (Phase 7 Step 6.1).

Pure in-memory: no HTTP, no Gemini, no Supabase, no backend_client,
no external services of any kind.

Coverage:
    - normal add / get / remove / clear operations
    - customer isolation (carts never leak between customers)
    - quantity updates (existing item never duplicates a line)
    - subtotal calculations (integer and float rounding)
    - invalid inputs (ids, names, prices, quantities, customer ids)
    - missing items (predictable item_not_found, store unchanged)
    - empty carts (unknown / cleared carts are safe)
    - capacity limits and defensive-copy behaviour
"""

from __future__ import annotations

import math

from ai.cart_store import (
    MAX_CART_LINES,
    MAX_ITEM_NAME_LENGTH,
    MAX_PRICE,
    MAX_QUANTITY,
    CartStore,
)

# OXRO customer_id charset (same as main.py): [A-Za-z0-9._:-] - no "@".
CUSTOMER_A = "cust-alice"
CUSTOMER_B = "cust-bob"
CUSTOMER_GUEST = "guest"

DISH_1 = "d_biryani"
DISH_1_NAME = "Hyderabadi Chicken Biryani"
DISH_1_PRICE = 299
DISH_2 = "d_pizza"
DISH_2_NAME = "Veg Pizza"
DISH_2_PRICE = 249


def _line(result):
    assert result["ok"] is True
    items = result["cart"]["items"]
    assert len(items) == 1
    return items[0]


# ---------------------------------------------------------------------------
# Normal operations
# ---------------------------------------------------------------------------

def test_add_item_success_envelope():
    store = CartStore()
    result = store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)

    assert result["ok"] is True
    assert result["error"] is None
    assert result["message"] is None
    assert isinstance(result["cart"], dict)


def test_add_item_stores_exactly_four_fields():
    store = CartStore()
    result = store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 2)

    line = _line(result)
    assert set(line.keys()) == {"item_id", "item_name", "price", "quantity"}
    assert line["item_id"] == DISH_1
    assert line["item_name"] == DISH_1_NAME
    assert line["price"] == DISH_1_PRICE
    assert line["quantity"] == 2


def test_add_item_echoes_customer_id_and_subtotal():
    store = CartStore()
    result = store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, 299, 3)

    assert result["cart"]["customer_id"] == CUSTOMER_A
    assert result["cart"]["subtotal"] == 897


def test_get_cart_returns_multiple_lines_in_order():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)
    store.add_item(CUSTOMER_A, DISH_2, DISH_2_NAME, DISH_2_PRICE, 2)

    result = store.get_cart(CUSTOMER_A)

    assert result["ok"] is True
    ids = [item["item_id"] for item in result["cart"]["items"]]
    assert ids == [DISH_1, DISH_2]


def test_remove_item_deletes_only_that_line():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)
    store.add_item(CUSTOMER_A, DISH_2, DISH_2_NAME, DISH_2_PRICE, 1)

    result = store.remove_item(CUSTOMER_A, DISH_1)

    assert result["ok"] is True
    ids = [item["item_id"] for item in result["cart"]["items"]]
    assert ids == [DISH_2]
    assert result["cart"]["subtotal"] == DISH_2_PRICE


def test_remove_last_line_returns_empty_cart_view():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)

    result = store.remove_item(CUSTOMER_A, DISH_1)

    assert result["ok"] is True
    assert result["cart"]["items"] == []
    assert result["cart"]["subtotal"] == 0


def test_clear_cart_empties_a_populated_cart():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 3)
    store.add_item(CUSTOMER_A, DISH_2, DISH_2_NAME, DISH_2_PRICE, 1)

    result = store.clear_cart(CUSTOMER_A)

    assert result["ok"] is True
    assert result["cart"]["items"] == []
    assert result["cart"]["subtotal"] == 0
    assert store.get_cart(CUSTOMER_A)["cart"]["items"] == []


def test_item_name_is_trimmed_before_storage():
    store = CartStore()
    result = store.add_item(CUSTOMER_A, DISH_1, "  Biryani  ", 100, 1)

    assert _line(result)["item_name"] == "Biryani"


# ---------------------------------------------------------------------------
# Customer isolation
# ---------------------------------------------------------------------------

def test_carts_are_isolated_between_customers():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 2)

    result = store.get_cart(CUSTOMER_B)

    assert result["ok"] is True
    assert result["cart"]["items"] == []
    assert result["cart"]["subtotal"] == 0


def test_one_customers_add_never_changes_another_cart():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)
    before = store.get_cart(CUSTOMER_A)["cart"]

    store.add_item(CUSTOMER_B, DISH_2, DISH_2_NAME, DISH_2_PRICE, 5)

    after = store.get_cart(CUSTOMER_A)["cart"]
    assert after == before


def test_remove_in_one_cart_leaves_other_carts_untouched():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)
    store.add_item(CUSTOMER_B, DISH_1, DISH_1_NAME, DISH_1_PRICE, 4)

    store.remove_item(CUSTOMER_A, DISH_1)

    assert store.get_cart(CUSTOMER_A)["cart"]["items"] == []
    other = store.get_cart(CUSTOMER_B)["cart"]
    assert len(other["items"]) == 1
    assert other["items"][0]["quantity"] == 4


def test_clear_in_one_cart_leaves_other_carts_untouched():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)
    store.add_item(CUSTOMER_B, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)

    store.clear_cart(CUSTOMER_A)

    assert store.get_cart(CUSTOMER_A)["cart"]["items"] == []
    assert len(store.get_cart(CUSTOMER_B)["cart"]["items"]) == 1


def test_same_item_id_in_two_customers_stays_separate():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 2)
    store.add_item(CUSTOMER_B, DISH_1, DISH_1_NAME, DISH_1_PRICE, 7)

    a = store.get_cart(CUSTOMER_A)["cart"]["items"][0]
    b = store.get_cart(CUSTOMER_B)["cart"]["items"][0]

    assert a["quantity"] == 2
    assert b["quantity"] == 7


# ---------------------------------------------------------------------------
# Quantity updates (existing item)
# ---------------------------------------------------------------------------

def test_add_existing_item_updates_quantity_not_duplicate():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 2)
    result = store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 3)

    items = result["cart"]["items"]
    assert len(items) == 1
    assert items[0]["quantity"] == 5


def test_add_existing_keeps_original_price():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, 299, 1)
    result = store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, 599, 1)

    assert _line(result)["price"] == 299


def test_add_existing_keeps_original_name():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)
    result = store.add_item(CUSTOMER_A, DISH_1, "Renamed Dish", DISH_1_PRICE, 1)

    assert _line(result)["item_name"] == DISH_1_NAME


def test_quantity_accumulates_across_many_adds():
    store = CartStore()
    for _ in range(4):
        store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)

    line = store.get_cart(CUSTOMER_A)["cart"]["items"][0]
    assert line["quantity"] == 4
    assert store.get_cart(CUSTOMER_A)["cart"]["subtotal"] == DISH_1_PRICE * 4


def test_quantity_exceeded_is_rejected_and_store_unchanged():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 60)

    result = store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 50)

    assert result["ok"] is False
    assert result["error"] == "quantity_exceeded"
    assert result["cart"] is not None
    line = store.get_cart(CUSTOMER_A)["cart"]["items"][0]
    assert line["quantity"] == 60


def test_quantity_at_max_limit_is_accepted():
    store = CartStore()
    result = store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, MAX_QUANTITY)

    assert result["ok"] is True
    assert _line(result)["quantity"] == MAX_QUANTITY


# ---------------------------------------------------------------------------
# Subtotal calculations
# ---------------------------------------------------------------------------

def test_subtotal_empty_cart_is_zero():
    store = CartStore()
    result = store.get_cart(CUSTOMER_A)

    assert result["cart"]["subtotal"] == 0


def test_subtotal_sums_price_times_quantity():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, 299, 2)
    store.add_item(CUSTOMER_A, DISH_2, DISH_2_NAME, 249, 1)

    result = store.get_cart(CUSTOMER_A)

    assert result["cart"]["subtotal"] == 299 * 2 + 249 * 1


def test_subtotal_stays_integer_for_integer_prices():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, 100, 3)

    subtotal = store.get_cart(CUSTOMER_A)["cart"]["subtotal"]

    assert isinstance(subtotal, int)
    assert subtotal == 300


def test_subtotal_rounds_float_totals_to_two_decimals():
    store = CartStore()
    store.add_item(CUSTOMER_A, "d1", "Item One", 0.1, 3)

    subtotal = store.get_cart(CUSTOMER_A)["cart"]["subtotal"]

    assert subtotal == 0.3
    assert math.isclose(subtotal, 0.3, abs_tol=1e-9)


def test_subtotal_with_fractional_prices():
    store = CartStore()
    store.add_item(CUSTOMER_A, "d1", "Item One", 99.5, 2)
    store.add_item(CUSTOMER_A, "d2", "Item Two", 49, 1)

    assert store.get_cart(CUSTOMER_A)["cart"]["subtotal"] == 248.0


def test_subtotal_tracks_quantity_updates():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, 150, 1)
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, 150, 2)

    assert store.get_cart(CUSTOMER_A)["cart"]["subtotal"] == 450


def test_subtotal_shrinks_after_remove():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, 200, 2)
    store.add_item(CUSTOMER_A, DISH_2, DISH_2_NAME, 100, 1)

    result = store.remove_item(CUSTOMER_A, DISH_2)

    assert result["cart"]["subtotal"] == 400


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def _invalid_customers():
    return [
        None,
        123,
        "",
        "   ",
        " cust-a",
        "cust-a ",
        "cust a",
        "cust-a\n",
        "a" * 129,
        "cust<script>",
    ]


def _invalid_item_ids():
    return [
        None,
        42,
        "",
        " d1",
        "d1 ",
        "d 1",
        "d1\n",
        "a" * 129,
        "d1!",
    ]


def test_add_item_rejects_invalid_customer_ids():
    store = CartStore()
    for bad in _invalid_customers():
        result = store.add_item(bad, DISH_1, DISH_1_NAME, 100, 1)
        assert result["ok"] is False, bad
        assert result["error"] == "invalid_customer_id", bad
        assert result["cart"] is None


def test_get_cart_rejects_invalid_customer_ids():
    store = CartStore()
    for bad in _invalid_customers():
        result = store.get_cart(bad)
        assert result["ok"] is False, bad
        assert result["error"] == "invalid_customer_id", bad


def test_add_item_rejects_invalid_item_ids():
    store = CartStore()
    for bad in _invalid_item_ids():
        result = store.add_item(CUSTOMER_A, bad, DISH_1_NAME, 100, 1)
        assert result["ok"] is False, bad
        assert result["error"] == "invalid_item_id", bad


def test_add_item_rejects_invalid_names():
    store = CartStore()
    bad_names = [
        None,
        123,
        ["biryani"],
        "",
        "    ",
        "biryani\nplease",
        "biryani\x00",
        "a" * (MAX_ITEM_NAME_LENGTH + 1),
    ]
    for bad in bad_names:
        result = store.add_item(CUSTOMER_A, DISH_1, bad, 100, 1)
        assert result["ok"] is False, repr(bad)
        assert result["error"] == "invalid_item_name", repr(bad)


def test_add_item_rejects_invalid_prices():
    store = CartStore()
    bad_prices = [
        None,
        "100",
        True,
        False,
        -1,
        -0.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        MAX_PRICE + 1,
        MAX_PRICE + 0.5,
        10**400,  # huge integer: range-checked before isfinite
    ]
    for bad in bad_prices:
        result = store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, bad, 1)
        assert result["ok"] is False, repr(bad)
        assert result["error"] == "invalid_price", repr(bad)


def test_add_item_rejects_invalid_quantities():
    store = CartStore()
    bad_quantities = [
        None,
        "2",
        2.0,
        1.5,
        0,
        -1,
        True,
        False,
        MAX_QUANTITY + 1,
    ]
    for bad in bad_quantities:
        result = store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, 100, bad)
        assert result["ok"] is False, repr(bad)
        assert result["error"] == "invalid_quantity", repr(bad)


def test_invalid_inputs_never_mutate_the_store():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)
    before = store.get_cart(CUSTOMER_A)["cart"]

    store.add_item(CUSTOMER_A, DISH_1, "", 100, 1)      # bad name
    store.add_item(CUSTOMER_A, "bad id", DISH_1_NAME, 100, 1)  # bad id
    store.add_item("", DISH_1, DISH_1_NAME, 100, 1)     # bad customer
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, -5, 1)  # bad price

    assert store.get_cart(CUSTOMER_A)["cart"] == before


def test_zero_and_negative_price_boundary():
    store = CartStore()
    zero = store.add_item(CUSTOMER_A, "d1", "Free Item", 0, 2)

    assert zero["ok"] is True
    assert zero["cart"]["subtotal"] == 0

    negative = store.add_item(CUSTOMER_A, "d2", "Bad Item", -1, 1)
    assert negative["error"] == "invalid_price"


def test_max_price_boundary_accepted():
    store = CartStore()
    result = store.add_item(CUSTOMER_A, "d1", "Premium Item", MAX_PRICE, 1)

    assert result["ok"] is True
    assert result["cart"]["subtotal"] == MAX_PRICE


def test_validation_failures_return_cart_none():
    store = CartStore()
    result = store.add_item(CUSTOMER_A, DISH_1, None, DISH_1_PRICE, 1)

    assert result["cart"] is None


# ---------------------------------------------------------------------------
# Missing items
# ---------------------------------------------------------------------------

def test_remove_unknown_item_returns_item_not_found():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)

    result = store.remove_item(CUSTOMER_A, DISH_2)

    assert result["ok"] is False
    assert result["error"] == "item_not_found"
    assert result["message"]
    assert len(result["cart"]["items"]) == 1


def test_remove_from_unknown_customer_returns_item_not_found():
    store = CartStore()
    result = store.remove_item("cust-nobody", DISH_1)

    assert result["ok"] is False
    assert result["error"] == "item_not_found"
    assert result["cart"]["items"] == []


def test_remove_unknown_item_leaves_store_unchanged():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 2)
    before = store.get_cart(CUSTOMER_A)["cart"]

    store.remove_item(CUSTOMER_A, "d_missing")

    assert store.get_cart(CUSTOMER_A)["cart"] == before


def test_remove_with_invalid_item_id_is_rejected():
    store = CartStore()
    result = store.remove_item(CUSTOMER_A, "bad id")

    assert result["ok"] is False
    assert result["error"] == "invalid_item_id"


def test_remove_with_invalid_customer_id_is_rejected():
    store = CartStore()
    result = store.remove_item("", DISH_1)

    assert result["ok"] is False
    assert result["error"] == "invalid_customer_id"


# ---------------------------------------------------------------------------
# Empty carts
# ---------------------------------------------------------------------------

def test_get_cart_unknown_customer_is_empty():
    store = CartStore()
    result = store.get_cart("cust-nobody")

    assert result["ok"] is True
    assert result["cart"]["customer_id"] == "cust-nobody"
    assert result["cart"]["items"] == []
    assert result["cart"]["subtotal"] == 0


def test_clear_unknown_customer_succeeds():
    store = CartStore()
    result = store.clear_cart("cust-nobody")

    assert result["ok"] is True
    assert result["cart"]["items"] == []
    assert result["cart"]["subtotal"] == 0


def test_clear_cart_is_idempotent():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)
    store.clear_cart(CUSTOMER_A)

    second = store.clear_cart(CUSTOMER_A)

    assert second["ok"] is True
    assert second["cart"]["items"] == []
    assert store.get_cart(CUSTOMER_A)["cart"]["items"] == []


def test_guest_customer_cart_starts_empty():
    store = CartStore()
    result = store.get_cart(CUSTOMER_GUEST)

    assert result["ok"] is True
    assert result["cart"]["items"] == []


# ---------------------------------------------------------------------------
# Capacity limits
# ---------------------------------------------------------------------------

def test_cart_line_limit_rejects_new_lines():
    store = CartStore(max_lines=2)
    store.add_item(CUSTOMER_A, "d1", "Item One", 100, 1)
    store.add_item(CUSTOMER_A, "d2", "Item Two", 100, 1)

    result = store.add_item(CUSTOMER_A, "d3", "Item Three", 100, 1)

    assert result["ok"] is False
    assert result["error"] == "cart_full"
    assert len(result["cart"]["items"]) == 2


def test_cart_line_limit_still_allows_quantity_updates():
    store = CartStore(max_lines=1)
    store.add_item(CUSTOMER_A, "d1", "Item One", 100, 1)

    result = store.add_item(CUSTOMER_A, "d1", "Item One", 100, 1)

    assert result["ok"] is True
    assert result["cart"]["items"][0]["quantity"] == 2


def test_cart_capacity_limit_rejects_new_customers():
    store = CartStore(max_carts=2)
    store.add_item(CUSTOMER_A, "d1", "Item One", 100, 1)
    store.add_item(CUSTOMER_B, "d1", "Item One", 100, 1)

    result = store.add_item("cust-carol", "d1", "Item One", 100, 1)

    assert result["ok"] is False
    assert result["error"] == "cart_capacity_reached"
    assert result["cart"] is None


def test_cart_capacity_allows_existing_customers():
    store = CartStore(max_carts=1)
    store.add_item(CUSTOMER_A, "d1", "Item One", 100, 1)

    result = store.add_item(CUSTOMER_A, "d2", "Item Two", 100, 1)

    assert result["ok"] is True
    assert len(result["cart"]["items"]) == 2


def test_line_limit_matches_default_constant():
    assert MAX_CART_LINES >= 1


# ---------------------------------------------------------------------------
# Defensive copies (callers can never mutate the store)
# ---------------------------------------------------------------------------

def test_returned_items_are_copies():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)

    cart = store.get_cart(CUSTOMER_A)["cart"]
    cart["items"][0]["quantity"] = 999
    cart["items"].clear()

    fresh = store.get_cart(CUSTOMER_A)["cart"]
    assert len(fresh["items"]) == 1
    assert fresh["items"][0]["quantity"] == 1


def test_mutating_one_result_never_affects_the_next_read():
    store = CartStore()
    store.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 2)

    first = store.add_item(CUSTOMER_A, DISH_2, DISH_2_NAME, DISH_2_PRICE, 1)
    first["cart"]["subtotal"] = -1
    first["cart"]["customer_id"] = "tampered"

    second = store.get_cart(CUSTOMER_A)["cart"]
    assert second["subtotal"] == DISH_1_PRICE * 2 + DISH_2_PRICE
    assert second["customer_id"] == CUSTOMER_A


# ---------------------------------------------------------------------------
# Isolation between store instances
# ---------------------------------------------------------------------------

def test_separate_store_instances_share_no_state():
    store_one = CartStore()
    store_two = CartStore()

    store_one.add_item(CUSTOMER_A, DISH_1, DISH_1_NAME, DISH_1_PRICE, 1)

    assert store_two.get_cart(CUSTOMER_A)["cart"]["items"] == []
