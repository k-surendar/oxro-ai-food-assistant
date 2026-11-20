"""
Offer / coupon discovery for POST /chat (Phase 7 Step 5).

Purpose:
    - Detect offer / coupon / discount requests and answer LOCALLY from
      the OXRO backend via backend_client.list_coupons().
    - Works without Gemini (no quota, no session) - same degraded-mode
      contract as greetings, restaurant discovery and menus.
    - Never invents expiry dates, eligibility rules, usage limits or
      conditions that are not present on the backend coupon objects.

Actual backend GET /coupons response (couponController.list):
    {"coupons": [
        {"code", "title", "sub", "off", "percent", "cap",
         "minOrder", "freeShip"},
        ...
    ]}

    There is NO expiry, eligibility, per-user or stacking field in the
    backend payload; those are therefore never shown or implied.

Error handling (all map to a polite "temporarily unavailable" reply):
    - OXRO_BACKEND_BASE_URL not configured
    - connection / timeout failures
    - HTTP errors
    - missing / malformed payload (no coupons list)
    - empty coupon list (distinct "none right now" reply)
    - unexpected exceptions raised by backend_client.list_coupons
      (logged by exception type only - never message/args)

Pipeline position (main.py, Step 3 wiring - not this module):
    greeting -> restaurant discovery -> restaurant menu ->
    dish selection -> offers -> food path.
    This module only implements detection + reply formatting.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from food_recommendation.language_phrases import normalize_message

logger = logging.getLogger("oxro.offer_discovery")

# Static replies - never fabricated coupon data.
UNAVAILABLE_REPLY = (
    "Sorry, offer information is temporarily unavailable. "
    "Please try again in a moment."
)

EMPTY_REPLY = (
    "There are no offers listed right now. "
    "Please check back a little later."
)

# Offer intent (normalized English; additive only).
# Matches common catalog nouns: offers, coupons, promos, promotions,
# deals, discounts, vouchers - plus common ask/show/list phrasing.
# Deliberately does NOT match bare singular verb "offer"
# ("Do you offer free delivery?"), restaurant discovery, dish search,
# selection confirmations, price questions or cancellation phrases.
_OFFER_PATTERN = re.compile(
    r"(?:"
    r"\b(?:any|some|the|these|those|current|active|available|"
    r"today'?s?|ongoing|latest|new|today)\s+offers?\b"
    r"|\boffers\s+(?:available|list|codes?|today|now|active|"
    r"currently|applicable|listed)\b"
    r"|\b(?:show|list|find|give|get|check|fetch|what|which|are|"
    r"have|got|want|need|looking\s+for)\b(?:\s+\w+){0,3}\s+offers\b"
    r"|\bcoupons?\b"
    r"|\bpromo(?:tions?|codes?|s)?\b"
    r"|\bdiscounts?\b"
    r"|\bdiscount\s+codes?\b"
    r"|\bdeals?\b"
    r"|\bvouchers?\b"
    r")"
)


def is_offer_request(user_message) -> bool:
    """
    True when the message asks about offers, coupons, promotions,
    deals or discounts rather than restaurants, dishes, selection,
    prices or cancellation.
    """
    if not user_message or not isinstance(user_message, str):
        return False

    normalized = normalize_message(user_message)
    if not normalized:
        return False

    return bool(_OFFER_PATTERN.search(normalized))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _discount_line(coupon: dict) -> str | None:
    """
    Structured discount text from off/percent/cap only.
    Returns None when off is missing or not a positive number.
    """
    off = coupon.get("off")
    if not _is_number(off) or off <= 0:
        return None

    percent = coupon.get("percent")
    cap = coupon.get("cap")

    if percent is True:
        line = f"{int(off)}% off"
        if _is_number(cap) and cap > 0:
            line += f" (up to Rs.{int(cap)})"
        return line

    return f"Flat Rs.{int(off)} off"


def _format_coupon(index: int, coupon: dict) -> str:
    """
    Format one coupon using ONLY fields present on the object.
    Missing fields are omitted - never defaulted or invented.
    """
    lines = []

    code = coupon.get("code")
    if _is_non_empty_str(code):
        lines.append(f"{index}. {code.strip()}")
    else:
        lines.append(f"{index}. Coupon")

    title = coupon.get("title")
    if _is_non_empty_str(title):
        lines.append(f"   {title.strip()}")

    sub = coupon.get("sub")
    if _is_non_empty_str(sub):
        lines.append(f"   {sub.strip()}")

    discount_line = _discount_line(coupon)
    if discount_line:
        lines.append(f"   {discount_line}")

    if coupon.get("freeShip") is True:
        lines.append("   Free delivery")

    min_order = coupon.get("minOrder")
    if _is_number(min_order) and min_order > 0:
        lines.append(f"   Minimum order: Rs.{int(min_order)}")

    return "\n".join(lines)


def _format_reply(coupons: list) -> str:
    header = "Here are the current offers:"
    body = "\n\n".join(
        _format_coupon(i, c) for i, c in enumerate(coupons, start=1)
    )
    return f"{header}\n\n{body}"


def _result(status: str, reply: str) -> dict[str, Any]:
    """Same response shape as greeting_reply / restaurant_discovery_reply."""
    return {
        "status": status,
        "reply": reply,
        "detected_preferences": None,
        "matched_foods": [],
    }


def offer_reply(user_message=None) -> dict[str, Any]:
    """
    Fetch coupons from the OXRO backend and build the chat reply.

    Never raises for backend problems: any failure returns a polite
    unavailable message so the customer is never shown fake offers.
    """
    # Imported here so tests can monkeypatch backend_client.list_coupons
    # and so this module stays importable without a configured backend.
    import backend_client

    try:
        result = backend_client.list_coupons()
    except Exception as exc:
        # Contain unexpected exceptions (bugs, mis-patches, etc.).
        # Log ONLY the exception type - never str(exc)/args, which could
        # carry URLs, keys or other sensitive details - and return the
        # standard unavailable reply so no fake data is shown.
        logger.error(
            "Offer discovery: unexpected backend client failure (%s)",
            type(exc).__name__,
        )
        return _result("offers_unavailable", UNAVAILABLE_REPLY)

    if not isinstance(result, dict) or result.get("ok") is not True:
        error = result.get("error") if isinstance(result, dict) else None
        logger.warning(
            "Offer discovery backend failure: %s",
            error or "invalid_result",
        )
        return _result("offers_unavailable", UNAVAILABLE_REPLY)

    data = result.get("data")
    if not isinstance(data, dict):
        logger.warning("Offer discovery: unexpected data structure")
        return _result("offers_unavailable", UNAVAILABLE_REPLY)

    coupons = data.get("coupons")
    if not isinstance(coupons, list):
        logger.warning("Offer discovery: missing coupons list")
        return _result("offers_unavailable", UNAVAILABLE_REPLY)

    if not coupons:
        return _result("offers_empty", EMPTY_REPLY)

    valid = [c for c in coupons if isinstance(c, dict)]

    if not valid:
        logger.warning("Offer discovery: no usable coupon objects")
        return _result("offers_unavailable", UNAVAILABLE_REPLY)

    return _result("offers_listed", _format_reply(valid))
