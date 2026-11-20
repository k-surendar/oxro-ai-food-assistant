"""
Menu-based dish selection for POST /chat (Phase 7 Step 4).

Purpose:
    - After a restaurant menu has been viewed, let the customer pick a
      dish by name, confirm the selection, and ask for its price.
    - Selection is CONFIRMATION ONLY: dish name, restaurant, item ID
      and price as stored from the backend menu payload. There is NO
      cart, order, checkout or payment behaviour in this module.
    - Runs LOCALLY from session-stored menu state - no Gemini, no
      extra backend calls, no invented prices/ratings/availability.

Matching rules:
    - Only dishes DISPLAYED in the menu reply are selectable,
      confirmable and priceable (display limit enforced in session
      state; undisplayed names are never selected).
    - Whole-name match first (token boundaries; a longer phrase that
      could be a different dish never matches a shorter stored name).
    - Fallback: unique phrase containment - the residual query appears
      in exactly one stored dish name (e.g. "chicken biryani" ->
      "Hyderabadi Chicken Biryani").
    - Multiple hits -> ambiguous with candidate names; never guess.
    - Zero hits on a dish-pick message while a menu is active ->
      dish_not_on_menu (do not silently fall through to the local
      catalogue path for that pick).
    - Undisplayed menu names -> dish_not_displayed (explain, never
      silently substitute another dish).
    - "No" / "Cancel" / "Change my selection" clear pending (and
      confirmed) state so a later "Yes" cannot confirm a cancelled dish.

Pipeline position (main.py):
    greeting -> restaurant discovery -> restaurant menu ->
    dish selection -> food path.
    Returns None for messages this module does not handle so existing
    food recommendations and follow-ups stay unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from food_recommendation.language_phrases import normalize_message

logger = logging.getLogger("oxro.dish_selection")


# Static replies - confirmation only, never fabricated data.
MENU_SELECTION_REQUIRED_REPLY = (
    "Please view a restaurant menu first, then tell me the dish "
    "you want to select."
)

DISH_NAME_REQUIRED_REPLY = (
    "Please tell me the full dish name from the menu you'd like "
    "to select."
)

DISH_NOT_ON_MENU_REPLY_TEMPLATE = (
    "I couldn't find that dish on the {restaurant} menu. "
    "Please choose a dish from the menu I showed you."
)

DISH_AMBIGUOUS_REPLY_TEMPLATE = (
    "I found more than one dish matching that: {candidates}. "
    "Please use the full dish name."
)

DISH_PRICE_UNAVAILABLE_REPLY_TEMPLATE = (
    "Price isn't listed for {name} right now."
)

DISH_NOT_DISPLAYED_REPLY_TEMPLATE = (
    "{name} is not among the currently displayed items for "
    "{restaurant}. Please choose a dish from the menu I showed you."
)

SELECTION_CANCELLED_REPLY = (
    "Cancelled - no dish is selected. "
    "Tell me which dish from the displayed menu you'd like instead."
)

SELECTION_CHANGED_REPLY = (
    "Your previous selection was cleared. "
    "Please choose a dish from the displayed menu."
)

# Bare / short confirmation phrases (whole message after normalize).
_CONFIRM_PHRASES = frozenset(
    {
        "yes",
        "yep",
        "yeah",
        "y",
        "ok",
        "okay",
        "confirm",
        "confirm it",
        "yes please",
        "please confirm",
        "select it",
        "yes select it",
        "yes select",
        "yes that's right",
        "that's right",
        "correct",
        "go ahead",
        "sure",
        "do it",
        "i confirm",
    }
)

# Confirmation may also appear as a short leading phrase with trailing
# punctuation-only leftovers already stripped by normalize_message.
_CONFIRM_LEAD_RE = re.compile(
    r"^(?:yes|yep|yeah|ok|okay|sure|confirm|correct)(?:\s+(?:please|"
    r"select\s+it|confirm\s+it|that's\s+right))?$"
)

# Rejection / cancellation of a pending (or confirmed) selection.
_CANCEL_PHRASES = frozenset(
    {
        "no",
        "nope",
        "nah",
        "n",
        "cancel",
        "cancel it",
        "cancel selection",
        "cancel my selection",
        "never mind",
        "nevermind",
        "stop",
        "i don't want it",
        "i do not want it",
        "don't want it",
        "do not want it",
        "i don't want it anymore",
        "not that one",
        "no thanks",
        "no thank you",
        "reject",
        "change my selection",
        "change selection",
        "change my dish",
        "change dish",
        "change it",
        "different dish",
        "another dish",
        "let me change",
        "i want to change",
        "start over",
    }
)

_CANCEL_RE = re.compile(
    r"^(?:"
    r"no|nope|nah|n|cancel(?:\s+(?:it|selection|my\s+selection))?"
    r"|never\s*mind|nevermind|stop"
    r"|i\s+(?:do\s+not|don't)\s+want\s+it(?:\s+anymore)?"
    r"|do\s+not\s+want\s+it|don't\s+want\s+it"
    r"|not\s+that\s+one"
    r"|no\s+thanks|no\s+thank\s+you"
    r"|reject"
    r"|change\s+(?:my\s+)?(?:selection|dish|it)"
    r"|different\s+dish|another\s+dish"
    r"|let\s+me\s+change|i\s+want\s+to\s+change"
    r"|start\s+over"
    r")$"
)

# Price questions about the pending / confirmed dish.
_PRICE_RE = re.compile(
    r"(?:"
    r"\b(?:what(?:'s|\s+is))\s+(?:the\s+)?price\b"
    r"|\bhow\s+much\b"
    r"|\b(?:what(?:'s|\s+is))\s+(?:the\s+)?(?:cost|mrp)\b"
    r"|\bprice(?:\s+of\s+it)?\??$"
    r"|\bshow\s+(?:me\s+)?(?:the\s+)?price\b"
    r")"
)

# Explicit dish name inside a price question:
#   "price of X" / "cost of X" / "how much is X" / "how much for X"
_PRICE_NAME_RE = re.compile(
    r"(?:"
    r"(?:what(?:'s|\s+is)|whats)\s+(?:the\s+)?(?:price|cost|mrp)"
    r"\s+(?:of|for)\s+(.+)"
    r"|\b(?:price|cost|mrp)\s+(?:of|for)\s+(.+)"
    r"|\bhow\s+much\s+(?:is|for|does\s+it\s+cost\s+for)\s+(?:the\s+)?(.+)"
    r"|\bhow\s+much\s+is\s+(?:the\s+)?(.+)"
    r")"
)

# Dish-pick signals: the customer wants a specific item selected.
_PICK_SIGNAL_RE = re.compile(
    r"(?:"
    r"^i(?:\s+(?:want|need)|'d\s+like|\s+would\s+like)\b"
    r"|\b(?:give|get|bring|fetch)\s+me\b"
    r"|\blet\s+me\s+have\b"
    r"|\bi(?:'ll|\s+will|\s+would)\s+have\b"
    r"|\b(?:select|choose|pick|order\s+the)\b"
    r"|\bcan\s+i\s+(?:get|have|order)\b"
    r")"
)

# Scaffold stripped from a dish-pick message to leave the dish query.
_PICK_SCAFFOLD = frozenset(
    {
        "i",
        "want",
        "to",
        "the",
        "a",
        "an",
        "some",
        "please",
        "give",
        "me",
        "get",
        "bring",
        "fetch",
        "let",
        "have",
        "will",
        "would",
        "like",
        "select",
        "choose",
        "pick",
        "order",
        "one",
        "dish",
        "item",
        "from",
        "menu",
        "for",
        "at",
        "can",
        "s",
        "that",
        "this",
        "it",
        "just",
        "new",
        "full",
    }
)

# Leading tokens that may remain before a price question's dish name.
_PRICE_SCAFFOLD = frozenset(
    {
        "what",
        "is",
        "the",
        "whats",
        "how",
        "much",
        "price",
        "cost",
        "mrp",
        "of",
        "for",
        "show",
        "me",
        "please",
        "a",
        "an",
        "this",
        "that",
        "it",
        "its",
    }
)


def _result(status: str, reply: str) -> dict[str, Any]:
    """Same response shape as greeting_reply / restaurant_menu_reply."""
    return {
        "status": status,
        "reply": reply,
        "detected_preferences": None,
        "matched_foods": [],
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalized(message: Optional[str]) -> str:
    if not message or not isinstance(message, str):
        return ""
    return normalize_message(message)


def is_confirmation(message: Optional[str]) -> bool:
    """True for a short confirm phrase ("Yes, select it.")."""
    normalized = _normalized(message)
    if not normalized:
        return False
    if normalized in _CONFIRM_PHRASES:
        return True
    return bool(_CONFIRM_LEAD_RE.match(normalized))


def is_price_request(message: Optional[str]) -> bool:
    """True when the message asks for a dish's price."""
    normalized = _normalized(message)
    if not normalized:
        return False
    return bool(_PRICE_RE.search(normalized))


def is_cancellation(message: Optional[str]) -> bool:
    """
    True for rejection / cancellation / change-selection phrases.
    Whole-message match only so ordinary sentences never cancel.
    """
    normalized = _normalized(message)
    if not normalized:
        return False
    if normalized in _CANCEL_PHRASES:
        return True
    return bool(_CANCEL_RE.match(normalized))


def extract_price_dish_query(message: Optional[str]) -> str:
    """
    Dish name asked about in a price question, or "" when the question
    refers only to the pending/confirmed dish ("What's the price?").
    """
    normalized = _normalized(message)
    if not normalized:
        return ""
    match = _PRICE_NAME_RE.search(normalized)
    if not match:
        return ""
    raw = next((g for g in match.groups() if g), "") or ""
    raw = raw.strip().strip("?!. ")
    # Strip price-question scaffolding left around the dish name.
    tokens = [t for t in raw.split() if t not in _PRICE_SCAFFOLD]
    return " ".join(tokens).strip()


def is_dish_pick_signal(message: Optional[str]) -> bool:
    """True when the message looks like a request to pick a dish."""
    normalized = _normalized(message)
    if not normalized:
        return False
    return bool(_PICK_SIGNAL_RE.search(normalized))


def _residual_tokens(normalized: str, scaffold: frozenset) -> list[str]:
    return [t for t in normalized.split() if t not in scaffold]


def _whole_name_in_message(name: str, normalized_message: str) -> bool:
    """
    Whole-name match of a dish name inside the normalized message.

    Same boundary rules as restaurant names: token boundaries, and a
    following content word rejects the hit (so a longer different dish
    phrase never selects a shorter stored name).
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
            return True
        # Content word follows (possible longer dish name): reject here
        # and keep scanning; containment fallback may still apply.
        continue
    return False


def _phrase_in_dish_name(phrase: str, dish_name: str) -> bool:
    """True when the normalized phrase appears inside the dish name."""
    if not phrase or not dish_name:
        return False
    return phrase in dish_name


def match_dish(
    query: str,
    items: list,
) -> tuple[Optional[dict], str, list]:
    """
    Resolve one dish from stored menu items.

    Returns (item, code, candidates) where code is one of:
        "ok"         - exactly one match (whole name or unique phrase)
        "ambiguous"  - multiple distinct dishes matched
        "not_found"  - nothing matched
        "required"   - empty query

    Never guesses: multiple hits always return the candidate list.
    """
    normalized_query = normalize_message(query or "").strip()
    if not normalized_query:
        return None, "required", []

    valid_items = [
        item for item in (items or [])
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item["name"].strip()
    ]

    whole_hits: dict[str, dict] = {}
    for item in valid_items:
        if _whole_name_in_message(item["name"], normalized_query):
            key = str(item.get("id") if item.get("id") is not None else id(item))
            whole_hits[key] = item

    if len(whole_hits) == 1:
        return next(iter(whole_hits.values())), "ok", []
    if len(whole_hits) > 1:
        return None, "ambiguous", list(whole_hits.values())

    # Fallback: residual phrase (query minus pick scaffolds) contained
    # in exactly one dish name.
    residual = " ".join(_residual_tokens(normalized_query, _PICK_SCAFFOLD))
    if not residual:
        residual = normalized_query

    phrase_hits: dict[str, dict] = {}
    for item in valid_items:
        dish_norm = normalize_message(item["name"])
        if _phrase_in_dish_name(residual, dish_norm):
            key = str(item.get("id") if item.get("id") is not None else id(item))
            phrase_hits[key] = item

    if len(phrase_hits) == 1:
        return next(iter(phrase_hits.values())), "ok", []
    if len(phrase_hits) > 1:
        return None, "ambiguous", list(phrase_hits.values())

    return None, "not_found", []


def _restaurant_label(menu_state: dict) -> str:
    name = menu_state.get("restaurant_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    rid = menu_state.get("restaurant_id")
    if isinstance(rid, str) and rid.strip():
        return rid.strip()
    return "this restaurant"


def _price_line(dish: dict) -> Optional[str]:
    price = dish.get("price")
    if not _is_number(price):
        return None
    line = f"Price: {price}"
    mrp = dish.get("mrp")
    if _is_number(mrp) and mrp > price:
        line += f" (MRP: {mrp})"
    return line


def _format_dish_summary(dish: dict, restaurant: str) -> str:
    """Confirmation text: name, restaurant, item id, price - nothing else."""
    name = dish.get("name") or "the selected dish"
    lines = [f"Selected dish: {name}", f"Restaurant: {restaurant}"]

    dish_id = dish.get("id")
    if dish_id is not None:
        lines.append(f"Item ID: {dish_id}")

    price_line = _price_line(dish)
    if price_line:
        lines.append(price_line)
    else:
        lines.append("Price: not listed")

    lines.append(
        "This is a selection confirmation only - no order has been placed."
    )
    return "\n".join(lines)


def _format_proposed(dish: dict, restaurant: str) -> str:
    name = dish.get("name") or "the dish"
    lines = [f"I found: {name}", f"Restaurant: {restaurant}"]

    dish_id = dish.get("id")
    if dish_id is not None:
        lines.append(f"Item ID: {dish_id}")

    price_line = _price_line(dish)
    if price_line:
        lines.append(price_line)
    else:
        lines.append("Price: not listed")

    lines.append('Reply "Yes, select it" to confirm your selection.')
    return "\n".join(lines)


def _copy_dish(dish: dict) -> dict:
    return {k: dish[k] for k in dish}


def _find_item_by_id(items: list, item_id: Any) -> Optional[dict]:
    if item_id is None:
        return None
    for item in items or []:
        if isinstance(item, dict) and item.get("id") == item_id:
            return item
    return None


def _undisplayed_names(menu_state: dict) -> list[str]:
    raw = menu_state.get("undisplayed_names")
    if not isinstance(raw, list):
        return []
    return [
        name for name in raw
        if isinstance(name, str) and name.strip()
    ]


def _name_in_query(name: str, query: str) -> bool:
    """Whole-name or unique-phrase style check of name inside query."""
    if not name or not query:
        return False
    if _whole_name_in_message(name, query):
        return True
    return _phrase_in_dish_name(query, normalize_message(name))


def _match_undisplayed_name(query: str, names: list) -> Optional[str]:
    """
    Return the unique undisplayed dish name matching query, or None.
    Used only to explain "not among displayed items" - never to select.
    """
    normalized_query = normalize_message(query or "").strip()
    if not normalized_query or not names:
        return None

    hits = [
        name for name in names
        if _name_in_query(name, normalized_query)
        or _phrase_in_dish_name(normalized_query, normalize_message(name))
    ]
    # Deduplicate while preserving order.
    unique: list[str] = []
    for name in hits:
        key = normalize_message(name)
        if key not in {normalize_message(u) for u in unique}:
            unique.append(name)
    if len(unique) == 1:
        return unique[0]
    return None


def _is_displayed(dish: dict, items: list) -> bool:
    """True when dish id still exists among selectable displayed items."""
    if not isinstance(dish, dict):
        return False
    live = _find_item_by_id(items, dish.get("id"))
    return live is not None


def handle_dish_selection(
    customer_id: str,
    message: str,
    session_manager,
) -> Optional[dict[str, Any]]:
    """
    Try to handle the message as menu dish selection / confirmation /
    cancellation / price query.

    Returns a result dict when handled, or None when the message should
    fall through to the existing food path (no menu state, refinement
    follow-ups, unrelated messages).

    Never raises for bad session data; logs by exception type only.
    """
    try:
        menu_state = session_manager.get_menu_state(customer_id)
    except Exception as exc:
        logger.error(
            "Dish selection: menu state read failure (%s)",
            type(exc).__name__,
        )
        return None

    if not isinstance(menu_state, dict) or not menu_state.get("items"):
        # No active menu context: leave greetings / food / discovery
        # behaviour completely unchanged.
        return None

    normalized = _normalized(message)
    if not normalized:
        return None

    items = menu_state.get("items") if isinstance(menu_state.get("items"), list) else []
    restaurant = _restaurant_label(menu_state)
    pending = menu_state.get("pending_dish")
    confirmed = menu_state.get("confirmed_dish")
    if not isinstance(pending, dict):
        pending = None
    if not isinstance(confirmed, dict):
        confirmed = None

    # ---- cancellation / rejection / change selection (FIX 1) --------
    # Checked before confirmation so "No" never confirms a pending dish.
    if is_cancellation(message):
        if pending is None and confirmed is None:
            # Nothing to cancel: not a selection concern.
            return None

        new_state = dict(menu_state)
        new_state["pending_dish"] = None
        new_state["confirmed_dish"] = None
        try:
            session_manager.set_menu_state(customer_id, new_state)
        except Exception as exc:
            logger.error(
                "Dish selection: menu state write failure (%s)",
                type(exc).__name__,
            )

        reply = (
            SELECTION_CHANGED_REPLY
            if "change" in normalized or "different" in normalized
            or "another" in normalized or "start over" in normalized
            else SELECTION_CANCELLED_REPLY
        )
        status = (
            "selection_changed"
            if reply == SELECTION_CHANGED_REPLY
            else "selection_cancelled"
        )
        return _result(status, reply)

    # ---- price query (FIX 2: explicit name wins over pending) ------
    if is_price_request(message):
        named_query = extract_price_dish_query(message)

        if named_query:
            item, code, candidates = match_dish(named_query, items)
            if code == "ok" and item is not None:
                target = item
            elif code == "ambiguous":
                names = [
                    c.get("name") for c in candidates
                    if isinstance(c, dict)
                    and isinstance(c.get("name"), str)
                    and c["name"].strip()
                ]
                return _result(
                    "dish_selection_ambiguous",
                    DISH_AMBIGUOUS_REPLY_TEMPLATE.format(
                        candidates=", ".join(names)
                    ),
                )
            else:
                hidden = _match_undisplayed_name(
                    named_query,
                    _undisplayed_names(menu_state),
                )
                if hidden:
                    return _result(
                        "dish_not_displayed",
                        DISH_NOT_DISPLAYED_REPLY_TEMPLATE.format(
                            name=hidden,
                            restaurant=restaurant,
                        ),
                    )
                return _result(
                    "dish_not_on_menu",
                    DISH_NOT_ON_MENU_REPLY_TEMPLATE.format(
                        restaurant=restaurant
                    ),
                )
        else:
            target = pending or confirmed
            if target is None:
                return _result(
                    "menu_selection_required",
                    "Please tell me which dish to price after viewing "
                    "the menu.",
                )
            # Only price dishes still present in the displayed menu.
            if not _is_displayed(target, items):
                return _result(
                    "dish_not_displayed",
                    DISH_NOT_DISPLAYED_REPLY_TEMPLATE.format(
                        name=target.get("name") or "That dish",
                        restaurant=restaurant,
                    ),
                )
            target = _find_item_by_id(items, target.get("id")) or target

        price_line = _price_line(target)
        name = target.get("name") or "that dish"
        if price_line is None:
            return _result(
                "dish_price_unavailable",
                DISH_PRICE_UNAVAILABLE_REPLY_TEMPLATE.format(name=name),
            )
        return _result(
            "dish_price",
            f"{name} - {price_line} at {restaurant}.",
        )

    # ---- confirmation ----------------------------------------------
    if is_confirmation(message):
        if pending is None:
            if confirmed is not None:
                if not _is_displayed(confirmed, items):
                    return _result(
                        "dish_not_displayed",
                        DISH_NOT_DISPLAYED_REPLY_TEMPLATE.format(
                            name=confirmed.get("name") or "That dish",
                            restaurant=restaurant,
                        ),
                    )
                live = _find_item_by_id(
                    items, confirmed.get("id")
                ) or confirmed
                return _result(
                    "dish_selected",
                    _format_dish_summary(live, restaurant),
                )
            return _result(
                "menu_selection_required",
                "Please choose a dish from the menu first, then confirm it.",
            )

        # Cancelled / stale pending must never confirm (FIX 1 + FIX 3).
        if not _is_displayed(pending, items):
            new_state = dict(menu_state)
            new_state["pending_dish"] = None
            try:
                session_manager.set_menu_state(customer_id, new_state)
            except Exception as exc:
                logger.error(
                    "Dish selection: menu state write failure (%s)",
                    type(exc).__name__,
                )
            return _result(
                "dish_not_displayed",
                DISH_NOT_DISPLAYED_REPLY_TEMPLATE.format(
                    name=pending.get("name") or "That dish",
                    restaurant=restaurant,
                ),
            )

        live = _find_item_by_id(items, pending.get("id")) or pending
        new_state = dict(menu_state)
        new_state["confirmed_dish"] = _copy_dish(live)
        new_state["pending_dish"] = None
        try:
            session_manager.set_menu_state(customer_id, new_state)
        except Exception as exc:
            logger.error(
                "Dish selection: menu state write failure (%s)",
                type(exc).__name__,
            )
            # Still confirm from the in-memory pending dish.
        return _result(
            "dish_selected",
            _format_dish_summary(live, restaurant),
        )

    # ---- dish pick --------------------------------------------------
    pick_signal = is_dish_pick_signal(message)

    # Also treat a bare whole-name mention of a stored dish as a pick
    # ("Hyderabadi Chicken Biryani" with no scaffold words).
    item, code, candidates = match_dish(message, items)

    if code == "ok" and item is not None:
        new_state = dict(menu_state)
        new_state["pending_dish"] = _copy_dish(item)
        # A new explicit pick replaces any previous confirmation.
        new_state["confirmed_dish"] = None
        try:
            session_manager.set_menu_state(customer_id, new_state)
        except Exception as exc:
            logger.error(
                "Dish selection: menu state write failure (%s)",
                type(exc).__name__,
            )
        return _result(
            "dish_selection_proposed",
            _format_proposed(item, restaurant),
        )

    if code == "ambiguous":
        names = []
        for c in candidates:
            name = c.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return _result(
            "dish_selection_ambiguous",
            DISH_AMBIGUOUS_REPLY_TEMPLATE.format(
                candidates=", ".join(names)
            ),
        )

    # not_found / required
    if pick_signal:
        residual = " ".join(
            _residual_tokens(normalized, _PICK_SCAFFOLD)
        )
        if not residual:
            return _result("dish_name_required", DISH_NAME_REQUIRED_REPLY)

        # FIX 3: name exists on the menu but was not displayed.
        hidden = _match_undisplayed_name(
            residual,
            _undisplayed_names(menu_state),
        )
        if hidden:
            return _result(
                "dish_not_displayed",
                DISH_NOT_DISPLAYED_REPLY_TEMPLATE.format(
                    name=hidden,
                    restaurant=restaurant,
                ),
            )

        return _result(
            "dish_not_on_menu",
            DISH_NOT_ON_MENU_REPLY_TEMPLATE.format(restaurant=restaurant),
        )

    # Bare dish-like phrase with no pick signal and no match: if the
    # message is mostly food words it may still be a pick attempt when
    # residual tokens exist after stripping price/confirm scaffolds.
    residual = " ".join(_residual_tokens(normalized, _PICK_SCAFFOLD))
    if residual and residual != normalized and code == "not_found":
        hidden = _match_undisplayed_name(
            residual,
            _undisplayed_names(menu_state),
        )
        if hidden:
            return _result(
                "dish_not_displayed",
                DISH_NOT_DISPLAYED_REPLY_TEMPLATE.format(
                    name=hidden,
                    restaurant=restaurant,
                ),
            )
        # e.g. "Veg Pizza special" after scaffold-ish leading words
        # without an explicit pick verb - only treat as pick when a
        # majority of tokens look like a name hit attempt (containment
        # already failed). Fall through instead: safer for food path.
        return None

    # Bare undisplayed whole-name mention without pick scaffold words.
    if code == "not_found":
        hidden = _match_undisplayed_name(
            normalized,
            _undisplayed_names(menu_state),
        )
        if hidden:
            return _result(
                "dish_not_displayed",
                DISH_NOT_DISPLAYED_REPLY_TEMPLATE.format(
                    name=hidden,
                    restaurant=restaurant,
                ),
            )

    # Not a selection message (refinement follow-ups, new searches, etc.)
    return None
