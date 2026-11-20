"""
Local in-memory customer cart store (Phase 7 Step 6.1).

Purpose:
    - LOCAL MOCK ONLY: an in-process cart keyed by customer_id, used to
      develop and test customer/cart behaviour without any backend.
    - Stores only dish information supplied by the caller:
      item_id, item_name, price, quantity. Nothing is fetched,
      enriched, guessed or invented.

Security / scope guarantees:
    - No database, no Supabase, no HTTP, no Gemini, no credentials.
    - Does NOT import httpx, backend_client, google.genai or any
      network client. Zero external calls of any kind.
    - customer_id is the isolation key: every operation is scoped to
      exactly one customer and never returns another customer's data.
    - Only caller-supplied values are stored; identifiers must match
      the OXRO identifier charset (no whitespace / control characters).

Result envelope (every operation returns this exact shape):
    success: {"ok": True,  "error": None, "message": None, "cart": <view>}
    failure: {"ok": False, "error": <code>, "message": <text>, "cart": <view|None>}

    - Input-validation failures never mutate the store and return
      cart=None.
    - item_not_found / cart_full / quantity_exceeded failures return the
      caller's OWN cart view (state unchanged) so replies can be built
      without a second read.

Cart view shape:
    {"customer_id": str,
     "items": [{"item_id", "item_name", "price", "quantity"}, ...],
     "subtotal": <int|float>}

Behaviour rules:
    - add_item on an existing item_id UPDATES the quantity (quantities
      sum); there is never a duplicate line. The originally stored
      item_name and price are kept (quantity-only update).
    - remove_item is scoped to the customer's own cart; a missing item
      returns error "item_not_found" (never raises).
    - clear_cart is idempotent: clearing an unknown / already-empty
      cart succeeds with an empty view.
    - get_cart for a valid customer with no cart returns an empty view;
      an invalid customer_id returns error "invalid_customer_id".
    - Returned carts are deep copies: callers cannot mutate the store.
    - subtotal = sum(price * quantity), rounded to 2 decimals when the
      total is a float (integer prices keep integer subtotals).

Limits (validated, fail-closed):
    - customer_id / item_id: 1-128 chars, charset [A-Za-z0-9._:-]
    - item_name: non-blank, <= 200 chars, no control characters
      (surrounding whitespace is trimmed before storage)
    - price: real number (bool rejected), 0 <= price <= 1,000,000,
      finite (NaN / inf rejected)
    - quantity: integer (bool rejected), 1 <= quantity <= 100
    - lines per cart: <= 50 ("cart_full")
    - carts per store: <= 1000 ("cart_capacity_reached")

Concurrency:
    - One store-level threading.Lock guards the cart map; operations
      are short in-memory mutations. Different customers are serialized
      (cheap and safe for a mock).

Scope exclusions (later steps / never): chatbot intent detection,
endpoint integration, customer profiles, checkout, payments, orders.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from typing import Any, Optional

logger = logging.getLogger("oxro.cart_store")

# Validation limits.
MAX_CUSTOMER_ID_LENGTH = 128
MAX_ITEM_ID_LENGTH = 128
MAX_ITEM_NAME_LENGTH = 200
MAX_PRICE = 1_000_000
MAX_QUANTITY = 100
MAX_CART_LINES = 50
MAX_CARTS = 1000

# OXRO identifier charset (same as customer_id / restaurant_id).
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9._:\-]+")


def _validate_identifier(value: Any, max_length: int) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) < 1 or len(value) > max_length:
        return False
    # fullmatch rejects whitespace, control characters and every other
    # character outside the OXRO charset - no silent trimming.
    return _IDENTIFIER_RE.fullmatch(value) is not None


def _validate_name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        return None
    name = value.strip()
    if not name or len(name) > MAX_ITEM_NAME_LENGTH:
        return None
    return name


def _validate_price(value: Any) -> Optional[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # Range first so a huge integer never reaches math.isfinite
    # (which would raise OverflowError); NaN passes both comparisons
    # and is caught by isfinite below; +/- inf fails the range check.
    if value < 0 or value > MAX_PRICE:
        return None
    if not math.isfinite(value):
        return None
    return value


def _validate_quantity(value: Any, max_quantity: int) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 1 or value > max_quantity:
        return None
    return value


def _failure(
    error: str,
    message: str,
    cart: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {"ok": False, "error": error, "message": message, "cart": cart}


def _success(cart: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "error": None, "message": None, "cart": cart}


def _subtotal(lines: list) -> int | float:
    total: int | float = 0
    for line in lines:
        total += line["price"] * line["quantity"]
    if isinstance(total, float):
        return round(total, 2)
    return total


def _cart_view(customer_id: str, lines: list) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "items": [
            {
                "item_id": line["item_id"],
                "item_name": line["item_name"],
                "price": line["price"],
                "quantity": line["quantity"],
            }
            for line in lines
        ],
        "subtotal": _subtotal(lines),
    }


class CartStore:
    """
    In-memory cart map (customer_id -> list of lines) with validation.

    Instances are independent: tests construct their own store, so no
    shared global state and no reset() is needed.
    """

    def __init__(self, max_carts=None, max_lines=None, max_quantity=None):
        self.max_carts = MAX_CARTS if max_carts is None else max(1, int(max_carts))
        self.max_lines = MAX_CART_LINES if max_lines is None else max(1, int(max_lines))
        self.max_quantity = (
            MAX_QUANTITY if max_quantity is None else max(1, int(max_quantity))
        )
        self._carts: dict[str, list] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ reads
    def get_cart(self, customer_id: Any) -> dict[str, Any]:
        if not _validate_identifier(customer_id, MAX_CUSTOMER_ID_LENGTH):
            return _failure(
                "invalid_customer_id",
                "customer_id must be 1-128 characters of letters, "
                "digits, dot, underscore, colon or hyphen.",
            )
        with self._lock:
            lines = self._carts.get(customer_id)
            if not lines:
                return _success(_cart_view(customer_id, []))
            return _success(_cart_view(customer_id, lines))

    # ----------------------------------------------------------- writes
    def add_item(
        self,
        customer_id: Any,
        item_id: Any,
        item_name: Any,
        price: Any,
        quantity: Any,
    ) -> dict[str, Any]:
        if not _validate_identifier(customer_id, MAX_CUSTOMER_ID_LENGTH):
            return _failure(
                "invalid_customer_id",
                "customer_id must be 1-128 characters of letters, "
                "digits, dot, underscore, colon or hyphen.",
            )
        if not _validate_identifier(item_id, MAX_ITEM_ID_LENGTH):
            return _failure(
                "invalid_item_id",
                "item_id must be 1-128 characters of letters, digits, "
                "dot, underscore, colon or hyphen.",
            )
        name = _validate_name(item_name)
        if name is None:
            return _failure(
                "invalid_item_name",
                "item_name must be a non-blank string of at most "
                f"{MAX_ITEM_NAME_LENGTH} characters.",
            )
        clean_price = _validate_price(price)
        if clean_price is None:
            return _failure(
                "invalid_price",
                "price must be a number between 0 and "
                f"{MAX_PRICE}.",
            )
        clean_quantity = _validate_quantity(quantity, self.max_quantity)
        if clean_quantity is None:
            return _failure(
                "invalid_quantity",
                "quantity must be an integer between 1 and "
                f"{self.max_quantity}.",
            )

        with self._lock:
            lines = self._carts.get(customer_id)

            if lines is None:
                if len(self._carts) >= self.max_carts:
                    logger.warning("Cart store at capacity (%s)", self.max_carts)
                    return _failure(
                        "cart_capacity_reached",
                        "The cart service is temporarily at capacity.",
                    )
                lines = []
                self._carts[customer_id] = lines

            for line in lines:
                if line["item_id"] == item_id:
                    total = line["quantity"] + clean_quantity
                    if total > self.max_quantity:
                        return _failure(
                            "quantity_exceeded",
                            f"Quantity for this item cannot exceed "
                            f"{self.max_quantity}.",
                            cart=_cart_view(customer_id, lines),
                        )
                    # Quantity-only update: original name/price kept,
                    # no duplicate line is ever created.
                    line["quantity"] = total
                    return _success(_cart_view(customer_id, lines))

            if len(lines) >= self.max_lines:
                return _failure(
                    "cart_full",
                    f"A cart can hold at most {self.max_lines} items.",
                    cart=_cart_view(customer_id, lines),
                )

            lines.append(
                {
                    "item_id": item_id,
                    "item_name": name,
                    "price": clean_price,
                    "quantity": clean_quantity,
                }
            )
            return _success(_cart_view(customer_id, lines))

    def remove_item(self, customer_id: Any, item_id: Any) -> dict[str, Any]:
        if not _validate_identifier(customer_id, MAX_CUSTOMER_ID_LENGTH):
            return _failure(
                "invalid_customer_id",
                "customer_id must be 1-128 characters of letters, "
                "digits, dot, underscore, colon or hyphen.",
            )
        if not _validate_identifier(item_id, MAX_ITEM_ID_LENGTH):
            return _failure(
                "invalid_item_id",
                "item_id must be 1-128 characters of letters, digits, "
                "dot, underscore, colon or hyphen.",
            )

        with self._lock:
            lines = self._carts.get(customer_id)
            if not lines:
                return _failure(
                    "item_not_found",
                    "That item is not in your cart.",
                    cart=_cart_view(customer_id, []),
                )

            for index, line in enumerate(lines):
                if line["item_id"] == item_id:
                    del lines[index]
                    if not lines:
                        # Never leave empty cart entries behind.
                        del self._carts[customer_id]
                    return _success(
                        _cart_view(customer_id, lines if lines else [])
                    )

            return _failure(
                "item_not_found",
                "That item is not in your cart.",
                cart=_cart_view(customer_id, lines),
            )

    def clear_cart(self, customer_id: Any) -> dict[str, Any]:
        if not _validate_identifier(customer_id, MAX_CUSTOMER_ID_LENGTH):
            return _failure(
                "invalid_customer_id",
                "customer_id must be 1-128 characters of letters, "
                "digits, dot, underscore, colon or hyphen.",
            )

        with self._lock:
            self._carts.pop(customer_id, None)

        # Idempotent: unknown / already-empty carts succeed too.
        return _success(_cart_view(customer_id, []))
