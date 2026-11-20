"""
Restaurant discovery for POST /chat (Phase 7 Step 2).

Purpose:
    - Detect restaurant-discovery intent ("Show me restaurants",
      "Which restaurants are available?", "I want to order food", ...)
      and answer LOCALLY from the OXRO backend via backend_client.
    - Works without Gemini (no quota, no session) - same degraded-mode
      contract as greetings.
    - Never invents restaurant names, ratings, delivery times or
      availability: only fields actually present in the backend
      response are displayed.

Actual backend GET /restaurants response (restaurantController.list):
    {"restaurants": [
        {"id", "name", "cuisines", "rating", "reviews",
         "distanceLabel", "etaMinutes"},
        ...
    ]}

    There is NO availability/status field in the backend payload;
    availability is therefore never shown.

Error handling (all map to a polite "temporarily unavailable" reply):
    - OXRO_BACKEND_BASE_URL not configured
    - connection / timeout failures
    - HTTP errors
    - empty restaurant list (distinct "none right now" reply)
    - unexpected response structure
    - unexpected exceptions raised by backend_client.list_restaurants
      (logged by exception type only - no message/args that could
      carry sensitive details)
"""

import logging
import re

from food_recommendation.language_phrases import normalize_message

logger = logging.getLogger("oxro.restaurant_discovery")

# Unavailable / empty replies - static, friendly, never fabricated data.
UNAVAILABLE_REPLY = (
    "Sorry, restaurant information is temporarily unavailable. "
    "Please try again in a moment."
)

EMPTY_REPLY = (
    "I couldn't find any restaurants right now. "
    "Please check back a little later."
)

# Discovery intents (normalized English; additive only - food-search
# messages such as "I want chicken biryani" must NOT match).
#   1. show/list/find/suggest/... + restaurants
#   2. restaurants + available/near/around/open
#   3. which/what/where + restaurants
#   4. any/some/good/best/nearby/local/top + restaurants
#   5. bare "restaurant(s)" as the whole message
#   6. explicit ordering-from-restaurant phrasing without a dish name
_DISCOVERY_PATTERN = re.compile(
    r"(?:"
    r"\b(?:show|list|find|give|get|suggest|recommend|display|fetch|bring)\b"
    r"(?:\s+(?:me|us|some|the|all|any|a|few|good|best|nearby|available))?"
    r"(?:\s+\w+){0,2}\s+restaurants?\b"
    r"|\brestaurants?\b(?:\s+\w+){0,2}\s+"
    r"(?:available|near(?:\s+me)?|around(?:\s+me)?|open)\b"
    r"|\b(?:which|what|where)\s+(?:are\s+)?(?:the\s+)?"
    r"(?:good\s+|best\s+|available\s+)?restaurants?\b"
    r"|\b(?:any|some|good|best|nearby|local|top)\s+restaurants?\b"
    r"|^restaurants?$"
    r"|\b(?:want|need|looking\s+for)\b(?:\s+\w+){0,2}\s+restaurants?\b"
    r"|\border\s+(?:food|something|anything)\b"
    r"|\bplace\s+(?:an\s+)?order\b"
    r"|\bwant\s+to\s+order\s+food\b"
    r"|\bready\s+to\s+order\b"
    r")"
)


def is_restaurant_discovery(user_message):
    """
    True when the message is a restaurant-discovery request rather than
    a dish-level food search or a greeting.
    """
    if not user_message:
        return False

    normalized = normalize_message(user_message)
    if not normalized:
        return False

    return bool(_DISCOVERY_PATTERN.search(normalized))


def _format_restaurant(index, restaurant):
    """
    Format one restaurant using ONLY fields present on the object.
    Missing fields are omitted - never defaulted or invented.
    """
    lines = []

    name = restaurant.get("name")
    if isinstance(name, str) and name.strip():
        lines.append(f"{index}. {name.strip()}")
    else:
        lines.append(f"{index}. Restaurant")

    rid = restaurant.get("id")
    if isinstance(rid, str) and rid.strip():
        lines.append(f"   ID: {rid.strip()}")

    cuisines = restaurant.get("cuisines")
    if isinstance(cuisines, list) and cuisines:
        parts = [
            str(c).strip() for c in cuisines
            if isinstance(c, str) and c.strip()
        ]
        if parts:
            lines.append(f"   Cuisine: {', '.join(parts)}")
    elif isinstance(cuisines, str) and cuisines.strip():
        lines.append(f"   Cuisine: {cuisines.strip()}")

    rating = restaurant.get("rating")
    reviews = restaurant.get("reviews")
    if isinstance(rating, (int, float)) and not isinstance(rating, bool):
        if isinstance(reviews, (int, float)) and not isinstance(reviews, bool):
            lines.append(f"   Rating: {rating} ({reviews} reviews)")
        else:
            lines.append(f"   Rating: {rating}")

    eta = restaurant.get("etaMinutes")
    if isinstance(eta, (int, float)) and not isinstance(eta, bool):
        lines.append(f"   Delivery time: ~{eta} min")

    distance = restaurant.get("distanceLabel")
    if isinstance(distance, str) and distance.strip():
        lines.append(f"   Distance: {distance.strip()}")

    return "\n".join(lines)


def _format_reply(restaurants):
    # Header must not claim availability: the backend payload has no
    # availability/status field (see module docstring).
    header = "Here are the restaurants:"
    body = "\n\n".join(
        _format_restaurant(i, r)
        for i, r in enumerate(restaurants, start=1)
    )
    return f"{header}\n\n{body}"


def _result(status, reply):
    """Same response shape as greeting_reply / handle_food_message."""
    return {
        "status": status,
        "reply": reply,
        "detected_preferences": None,
        "matched_foods": [],
    }


def restaurant_discovery_reply(user_message=None):
    """
    Fetch restaurants from the OXRO backend and build the chat reply.

    Never raises for backend problems: any failure returns a polite
    unavailable message so the customer is never shown fake data.
    """
    # Imported here so tests can monkeypatch backend_client.list_restaurants
    # and so this module stays importable without a configured backend.
    import backend_client

    try:
        result = backend_client.list_restaurants()
    except Exception as exc:
        # Contain unexpected exceptions (bugs, mis-patches, etc.).
        # Log ONLY the exception type - never str(exc)/args, which could
        # carry URLs, keys or other sensitive details - and return the
        # standard unavailable reply so no fake data is shown.
        logger.error(
            "Restaurant discovery: unexpected backend client failure (%s)",
            type(exc).__name__,
        )
        return _result("restaurant_unavailable", UNAVAILABLE_REPLY)

    if not isinstance(result, dict) or result.get("ok") is not True:
        error = result.get("error") if isinstance(result, dict) else None
        logger.warning(
            "Restaurant discovery backend failure: %s",
            error or "invalid_result",
        )
        return _result("restaurant_unavailable", UNAVAILABLE_REPLY)

    data = result.get("data")
    if not isinstance(data, dict):
        logger.warning("Restaurant discovery: unexpected data structure")
        return _result("restaurant_unavailable", UNAVAILABLE_REPLY)

    restaurants = data.get("restaurants")
    if not isinstance(restaurants, list):
        logger.warning("Restaurant discovery: missing restaurants list")
        return _result("restaurant_unavailable", UNAVAILABLE_REPLY)

    if not restaurants:
        return _result("restaurant_empty", EMPTY_REPLY)

    valid = [
        r for r in restaurants
        if isinstance(r, dict)
    ]

    if not valid:
        logger.warning("Restaurant discovery: no usable restaurant objects")
        return _result("restaurant_unavailable", UNAVAILABLE_REPLY)

    return _result("restaurant_discovery", _format_reply(valid))
