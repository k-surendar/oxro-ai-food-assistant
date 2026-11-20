"""
Restaurant menu discovery for POST /chat (Phase 7 - menu step).

Purpose:
    - Detect menu-viewing intent ("Show me the menu",
      "Paradise Biryani menu", "Give me the menu for ...", ...)
      and answer LOCALLY from the OXRO backend via backend_client.
    - Resolve a restaurant by explicit ID or exact/unambiguous name
      using list_restaurants(); never guess when missing/ambiguous.
    - Fetch one restaurant's menu with get_restaurant_menu().
    - Works without Gemini (no quota, no session) - same degraded-mode
      contract as greetings and restaurant discovery.
    - Never invents prices, ratings, availability, categories,
      ingredients or discounts: only fields actually present on the
      backend dish objects are displayed.

Actual backend GET /restaurants/{id}/menu response
(restaurantController.menu):
    {"menu": [
        {"id", "name", "brand", "price", "mrp", "rating", "reviews",
         "restaurantId", "restaurantName", "veg", "egg",
         "description", "imageUrl", ...},
        ...
    ]}

    UI-only fields (icon, colors) are never shown in chat replies.

Error handling (all map to a polite, non-fabricated reply):
    - OXRO_BACKEND_BASE_URL not configured / connection / timeout
    - HTTP errors (404 -> restaurant not found; other -> unavailable)
    - empty menu list (distinct "none listed" reply)
    - missing or ambiguous restaurant in the customer message
    - unexpected response structure
    - unexpected exceptions raised by backend_client (logged by
      exception type only - no message/args that could carry
      sensitive details)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from food_recommendation.language_phrases import normalize_message

logger = logging.getLogger("oxro.restaurant_menu")

# Static replies - friendly, never fabricated data.
MENU_UNAVAILABLE_REPLY = (
    "Sorry, menu information is temporarily unavailable. "
    "Please try again in a moment."
)

MENU_REQUIRED_REPLY = (
    "Which restaurant's menu would you like to see? "
    "Please tell me the restaurant name or ID."
)

MENU_AMBIGUOUS_REPLY = (
    "I found more than one restaurant matching that name. "
    "Please specify the full restaurant name or its ID."
)

MENU_NOT_FOUND_REPLY = (
    "Sorry, I couldn't find that restaurant. "
    "Please check the name or choose from the restaurant list."
)

# Cap long menus so chat replies stay readable.
MAX_DISPLAYED_MENU_ITEMS = 12

# Cap the structured selection state stored in session so context stays
# small even for very large backend menus.
MAX_MENU_STATE_ITEMS = 50

# Menu intent: must include the word menu/menus. Narrow enough not to
# swallow greetings, restaurant discovery or dish-level food requests.
_MENU_INTENT_RE = re.compile(
    r"(?:"
    r"\b(?:show|list|give|get|fetch|display|view|see|open|check)\b"
    r"(?:\s+\w+){0,5}\s+menus?\b"
    r"|\bi\s+want\s+to\s+see\b(?:\s+\w+){0,4}\s+menus?\b"
    r"|\bmenus?\b(?:\s+\w+){0,4}\s+(?:for|at|from|of)\b"
    r"|\b(?:for|at|from|of)\s+(?:the\s+)?(?:\w+\s+){0,5}menus?\b"
    r"|\bwhat\s+(?:is|are)\s+(?:available\s+)?on\s+(?:the\s+)?menus?\b"
    r"|\b(?:restaurant|food|dining)\s+menus?\b"
    r"|\b\w+(?:\s+\w+){0,5}\s+menus?\b"
    r"|^menus?$"
    r")"
)

# Scaffold tokens stripped only to tell "no restaurant named" apart from
# "a name was given but nothing matched" (not_found vs required).
_SCAFFOLD_WORDS = frozenset(
    {
        "show",
        "me",
        "us",
        "the",
        "a",
        "an",
        "some",
        "my",
        "full",
        "complete",
        "whole",
        "please",
        "i",
        "want",
        "to",
        "see",
        "list",
        "give",
        "get",
        "fetch",
        "display",
        "view",
        "open",
        "check",
        "menu",
        "menus",
        "restaurant",
        "restaurants",
        "food",
        "dining",
        "for",
        "at",
        "from",
        "of",
        "what",
        "is",
        "are",
        "available",
        "on",
        "and",
        "or",
        "order",
        "ordering",
    }
)


def is_menu_request(user_message) -> bool:
    """
    True when the message asks to view a restaurant menu rather than
    a greeting, restaurant discovery list, or dish-level food search.
    """
    if not user_message or not isinstance(user_message, str):
        return False

    normalized = normalize_message(user_message)
    if not normalized:
        return False

    return bool(_MENU_INTENT_RE.search(normalized))


def _result(
    status: str,
    reply: str,
    menu_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Same response shape as greeting_reply / restaurant_discovery_reply.

    On successful menu display only, menu_context carries the structured
    restaurant + dish list for session storage (Phase 7 Step 4 selection).
    """
    result: dict[str, Any] = {
        "status": status,
        "reply": reply,
        "detected_preferences": None,
        "matched_foods": [],
    }
    if menu_context is not None:
        result["menu_context"] = menu_context
    return result


def _is_number(value: Any) -> bool:
    """True for int/float but not bool (bool is an int subclass)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dish_display_name(dish: Any) -> Optional[str]:
    """Non-empty dish name, or None when missing/blank/non-string."""
    if not isinstance(dish, dict):
        return None
    name = dish.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _format_dish(index: int, dish: dict) -> str:
    """
    Format one dish using ONLY fields present on the object.
    Requires a valid non-empty name (callers filter first).
    Missing fields are omitted - never defaulted or invented.
    UI-only fields (icon, colors) are never shown.
    """
    name = _dish_display_name(dish)
    if name is None:
        # Defensive: unnamed records must never be presented as dishes.
        raise ValueError("dish without a valid name cannot be formatted")

    lines = [f"{index}. {name}"]

    price = dish.get("price")
    mrp = dish.get("mrp")
    if _is_number(price):
        price_line = f"   Price: {price}"
        # MRP only when supplied and useful (strictly above price).
        if _is_number(mrp) and mrp > price:
            price_line += f" (MRP: {mrp})"
        lines.append(price_line)

    rating = dish.get("rating")
    if _is_number(rating):
        lines.append(f"   Rating: {rating}")

    # Veg / egg labels only when the field is explicitly true.
    if dish.get("veg") is True:
        lines.append("   Vegetarian")
    if dish.get("egg") is True:
        lines.append("   Egg")

    return "\n".join(lines)


def _state_item(dish: dict) -> dict[str, Any]:
    """
    Extract only selection-relevant fields from one named dish.
    Missing / wrong-typed fields are omitted - never invented.
    """
    item: dict[str, Any] = {
        "id": dish.get("id"),
        "name": _dish_display_name(dish),
    }

    price = dish.get("price")
    if _is_number(price):
        item["price"] = price

    mrp = dish.get("mrp")
    if _is_number(mrp):
        item["mrp"] = mrp

    rating = dish.get("rating")
    if _is_number(rating):
        item["rating"] = rating

    if dish.get("veg") is True:
        item["veg"] = True
    if dish.get("egg") is True:
        item["egg"] = True

    restaurant_id = dish.get("restaurantId")
    if isinstance(restaurant_id, str) and restaurant_id.strip():
        item["restaurantId"] = restaurant_id.strip()

    restaurant_name = dish.get("restaurantName")
    if isinstance(restaurant_name, str) and restaurant_name.strip():
        item["restaurantName"] = restaurant_name.strip()

    return item


def _build_menu_context(
    restaurant_id: str,
    restaurant_name: Optional[str],
    dishes: list,
) -> dict[str, Any]:
    """
    Structured state stored in session after a successful menu view.

    Only dishes actually DISPLAYED to the customer (the first
    MAX_DISPLAYED_MENU_ITEMS) are selectable. Undisplayed names are
    kept solely so the assistant can explain that a requested dish is
    not among the currently displayed items - they are never proposed,
    confirmed or priced.
    """
    displayed = dishes[:MAX_DISPLAYED_MENU_ITEMS]
    undisplayed = dishes[MAX_DISPLAYED_MENU_ITEMS:MAX_MENU_STATE_ITEMS]

    items = [_state_item(d) for d in displayed]
    undisplayed_names: list[str] = []
    for dish in undisplayed:
        name = _dish_display_name(dish)
        if name:
            undisplayed_names.append(name)

    return {
        "restaurant_id": restaurant_id,
        "restaurant_name": restaurant_name,
        "items": items,
        "undisplayed_names": undisplayed_names,
        "pending_dish": None,
        "confirmed_dish": None,
    }


def _format_menu_reply(restaurant_name: Optional[str], dishes: list) -> str:
    header = (
        f"Menu for {restaurant_name}:"
        if restaurant_name
        else "Menu:"
    )
    shown = dishes[:MAX_DISPLAYED_MENU_ITEMS]
    omitted = len(dishes) - len(shown)
    body = "\n\n".join(
        _format_dish(i, d) for i, d in enumerate(shown, start=1)
    )
    reply = f"{header}\n\n{body}"
    if omitted > 0:
        reply += f"\n\n... and {omitted} more item(s) not shown."
    return reply


def _residual_specifier(normalized: str) -> str:
    """Leftover tokens after stripping menu-request scaffolding words."""
    tokens = re.findall(r"[a-z0-9_:\-]+", normalized)
    return " ".join(t for t in tokens if t not in _SCAFFOLD_WORDS)


def _id_in_message(restaurant_id: str, message: str) -> bool:
    """
    Match a raw restaurant ID in the original message (case-insensitive).
    normalize_message turns underscores into spaces, so IDs are checked
    against the raw text with word boundaries.
    """
    if not restaurant_id or not message:
        return False
    pattern = (
        r"(?<![a-z0-9])" + re.escape(restaurant_id.lower()) + r"(?![a-z0-9])"
    )
    return bool(re.search(pattern, message.lower()))


def _name_in_message(name: str, normalized_message: str) -> bool:
    """
    Whole-name match of a restaurant name inside the normalized message.

    Case-insensitive (both sides already normalized). Uses token
    boundaries so a shorter name never matches inside a longer token
    (e.g. "Biryani" must not match "BiryaniExpress").

    After a full-name hit, the next token (if any) must be end-of-text,
    punctuation-only, or a known menu-request scaffold word. A following
    content word means the phrase could be a longer different restaurant
    (e.g. "Paradise Biryani Express" must not select "Paradise Biryani").
    """
    if not name or not normalized_message:
        return False

    norm_name = normalize_message(name).strip()
    if not norm_name:
        return False

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(norm_name)
        + r"(?![a-z0-9])"
    )
    for match in re.finditer(pattern, normalized_message):
        after = normalized_message[match.end():].lstrip(" \t")
        if not after:
            return True
        next_token = re.match(r"[a-z0-9']+", after)
        if not next_token:
            # Only punctuation remains after the name.
            return True
        if next_token.group(0) in _SCAFFOLD_WORDS:
            return True
        # Content word follows (possible longer restaurant name): reject.
    return False


def _resolve_restaurant(
    message: str,
    restaurants: list,
) -> tuple[Optional[dict], str]:
    """
    Resolve one restaurant from the customer message.

    Returns (restaurant, code) where code is one of:
        "ok"         - exactly one match (id or unambiguous whole name)
        "ambiguous"  - multiple distinct restaurants matched
        "required"   - no restaurant mentioned; ask the customer
        "not_found"  - a name/ID was mentioned but nothing matched

    Never guesses: only whole-name (or explicit ID) matches are used.
    A partial name that could refer to another restaurant is rejected.
    """
    normalized = normalize_message(message)
    id_hits: list[dict] = []
    name_hits: list[dict] = []

    for restaurant in restaurants:
        rid = restaurant.get("id")
        if isinstance(rid, str) and rid.strip():
            if _id_in_message(rid.strip(), message):
                id_hits.append(restaurant)

        name = restaurant.get("name")
        if isinstance(name, str) and name.strip():
            if _name_in_message(name, normalized):
                name_hits.append(restaurant)

    matches: dict[str, dict] = {}
    for restaurant in id_hits + name_hits:
        key = str(restaurant.get("id") or id(restaurant))
        matches[key] = restaurant

    if len(matches) == 1:
        return next(iter(matches.values())), "ok"
    if len(matches) > 1:
        return None, "ambiguous"

    # No match: distinguish "did not say which" vs "said an unknown name".
    if _residual_specifier(normalized):
        return None, "not_found"
    return None, "required"


def _load_restaurants() -> tuple[Optional[list], Optional[str]]:
    """
    Fetch and validate the restaurant list.

    Returns (restaurants, None) on success or (None, status) on failure
    where status is menu_unavailable.
    """
    # Imported here so tests can monkeypatch backend_client.list_restaurants
    # and so this module stays importable without a configured backend.
    import backend_client

    try:
        result = backend_client.list_restaurants()
    except Exception as exc:
        logger.error(
            "Restaurant menu: unexpected list failure (%s)",
            type(exc).__name__,
        )
        return None, "menu_unavailable"

    if not isinstance(result, dict) or result.get("ok") is not True:
        error = result.get("error") if isinstance(result, dict) else None
        logger.warning(
            "Restaurant menu list failure: %s",
            error or "invalid_result",
        )
        return None, "menu_unavailable"

    data = result.get("data")
    if not isinstance(data, dict):
        logger.warning("Restaurant menu: unexpected list data structure")
        return None, "menu_unavailable"

    restaurants = data.get("restaurants")
    if not isinstance(restaurants, list):
        logger.warning("Restaurant menu: missing restaurants list")
        return None, "menu_unavailable"

    valid = [r for r in restaurants if isinstance(r, dict)]
    return valid, None


def restaurant_menu_reply(user_message=None) -> dict[str, Any]:
    """
    Resolve a restaurant, fetch its menu, and build the chat reply.

    Never raises for backend problems: any failure returns a polite
    message so the customer is never shown fake menu data.
    """
    if not user_message or not isinstance(user_message, str):
        return _result("menu_restaurant_required", MENU_REQUIRED_REPLY)

    restaurants, fail_status = _load_restaurants()
    if fail_status is not None:
        return _result(fail_status, MENU_UNAVAILABLE_REPLY)

    restaurant, code = _resolve_restaurant(user_message, restaurants)
    if code == "ambiguous":
        return _result("menu_restaurant_ambiguous", MENU_AMBIGUOUS_REPLY)
    if code == "required":
        return _result("menu_restaurant_required", MENU_REQUIRED_REPLY)
    if code == "not_found":
        return _result("menu_restaurant_not_found", MENU_NOT_FOUND_REPLY)

    restaurant_id = restaurant.get("id")
    if not isinstance(restaurant_id, str) or not restaurant_id.strip():
        return _result("menu_restaurant_not_found", MENU_NOT_FOUND_REPLY)

    restaurant_name: Optional[str] = None
    raw_name = restaurant.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        restaurant_name = raw_name.strip()

    # Imported here so tests can monkeypatch backend_client.get_restaurant_menu
    import backend_client

    try:
        result = backend_client.get_restaurant_menu(restaurant_id)
    except Exception as exc:
        logger.error(
            "Restaurant menu: unexpected menu failure (%s)",
            type(exc).__name__,
        )
        return _result("menu_unavailable", MENU_UNAVAILABLE_REPLY)

    if not isinstance(result, dict) or result.get("ok") is not True:
        error = result.get("error") if isinstance(result, dict) else None
        status_code = (
            result.get("status_code") if isinstance(result, dict) else None
        )
        if status_code == 404:
            return _result("menu_restaurant_not_found", MENU_NOT_FOUND_REPLY)
        logger.warning(
            "Restaurant menu backend failure: %s",
            error or "invalid_result",
        )
        return _result("menu_unavailable", MENU_UNAVAILABLE_REPLY)

    payload = result.get("data")
    if not isinstance(payload, dict):
        logger.warning("Restaurant menu: unexpected menu data structure")
        return _result("menu_unavailable", MENU_UNAVAILABLE_REPLY)

    dishes = payload.get("menu")
    if not isinstance(dishes, list):
        logger.warning("Restaurant menu: missing menu list")
        return _result("menu_unavailable", MENU_UNAVAILABLE_REPLY)

    if not dishes:
        where = f" for {restaurant_name}" if restaurant_name else ""
        return _result(
            "menu_empty",
            f"I couldn't find any menu items{where} right now.",
        )

    # Only records with a valid non-empty name are shown as dishes.
    # Unnamed / blank / non-dict records are skipped, never labeled "Dish".
    valid_dishes = [
        d for d in dishes if _dish_display_name(d) is not None
    ]
    if not valid_dishes:
        logger.warning("Restaurant menu: no usable named menu items")
        return _result("menu_unavailable", MENU_UNAVAILABLE_REPLY)

    if not restaurant_name:
        for dish in valid_dishes:
            fallback = dish.get("restaurantName")
            if isinstance(fallback, str) and fallback.strip():
                restaurant_name = fallback.strip()
                break

    reply = _format_menu_reply(restaurant_name, valid_dishes)
    menu_context = _build_menu_context(
        restaurant_id,
        restaurant_name,
        valid_dishes,
    )
    return _result("restaurant_menu", reply, menu_context=menu_context)
