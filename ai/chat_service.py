"""
Chat orchestration for the POST /chat endpoint.

Purpose:
    - Answer greetings locally so they never consume Gemini quota AND
      never require the Gemini client to be initialized (greetings work
      even in degraded mode; they also never create a session).
    - Run smart_food_search against the OXRO catalogue.
    - Embed the matched catalogue items (and detected preferences) into
      the message sent to the customer's Gemini chat session, so the
      reply is grounded in the OXRO catalogue: Gemini is instructed to
      recommend ONLY matched items and never invent food, restaurants,
      prices, availability or order status.
    - Conversation history is preserved per customer via SessionManager.

Phase 4 - context-aware follow-ups:
    - Each customer's session stores conversation context (merged
      preferences + previously matched catalogue items).
    - New explicit preferences override stored ones (e.g. switching
      from non-vegetarian to vegetarian); unset values inherit.
    - When a follow-up message produces no new direct search matches
      (e.g. "Make it spicy", "Show me cheaper options"), the previously
      matched catalogue items are embedded in the grounded follow-up
      message so Gemini can respond conversationally WITHOUT inventing
      new items.
    - Continuation requests ("Show more", "Suggest something else")
      search the catalogue for genuine alternatives that were NOT
      previously recommended; only when no unshown alternatives exist
      does the follow-up path apply. Never repeats shown items when
      alternatives exist.
    - The OXRO catalogue contains no price information: Gemini is
      explicitly instructed never to claim an item is cheaper or more
      expensive and to explain this limitation honestly.
    - Results are ranked with rank_food_results before the grounded
      prompt is built.

Error handling:
    - Lets the typed Gemini errors from conversation_manager propagate;
      the FastAPI layer maps them to proper HTTP responses (429/503/504/502).
"""

import json
import re

from ai.greetings import detect_greeting, greeting_response
from food_recommendation.food_search import (
    GENERAL_FALLBACK_CATEGORIES,
    NONVEG_FALLBACK_CATEGORIES,
    VEG_FALLBACK_CATEGORIES,
    _collect_from_categories,
    filter_food_type,
    smart_food_search,
)
from food_recommendation.language_phrases import (
    TANGLISH_CONTINUATION_PATTERN,
    normalize_message,
)
from food_recommendation.ranking_engine import rank_food_results
from food_recommendation.recommendation_engine import recommend_by_category


# Upper bound for alternatives surfaced by a continuation request, to
# keep the grounded prompt small.
MAX_FOLLOWUP_ALTERNATIVES = 20

# Continuation requests ask for MORE/OTHER items rather than refining
# the current ones ("Show more", "Suggest something else"). Refinement
# follow-ups ("Make it spicy", "Show me cheaper options") do NOT match.
# Phase 5: Tanglish continuation phrases are handled by the additive
# TANGLISH_CONTINUATION_PATTERN (see language_phrases.py).
CONTINUATION_PATTERN = re.compile(
    r"\bsomething\s+else\b"
    r"|\banything\s+else\b"
    r"|\banything\s+more\b"
    r"|\bmore\s+options?\b"
    r"|\bother\s+options?\b"
    r"|\bmore\s+food\b"
    r"|\b(show|give|get|suggest|recommend)\s+"
    r"(me\s+)?(some\s+)?(more|others?|another|different|else)\b"
)


def is_greeting(user_message):
    """True when the message can be answered locally (no Gemini)."""
    return detect_greeting(user_message)


def greeting_reply(user_message):
    """
    Local greeting response. Never touches Gemini, never creates a
    session, and therefore works even when Gemini is unavailable.
    """
    return {
        "status": "greeting",
        "reply": greeting_response(user_message),
        "detected_preferences": None,
        "matched_foods": [],
    }


def _is_continuation_request(user_message):
    """
    True when the customer asks for MORE/OTHER items ("Show more",
    "Suggest something else", "Innum kaatunga", "Vera edavadhu")
    rather than refining the current ones.
    """
    normalized = normalize_message(user_message)

    return bool(
        CONTINUATION_PATTERN.search(normalized)
        or TANGLISH_CONTINUATION_PATTERN.search(normalized)
    )


def _merge_preferences(previous, current):
    """
    Merge freshly detected preferences over stored ones.

    Rule: a new NON-None value overrides the stored value; a None
    value inherits the stored one. This makes explicit switches work
    ("What about vegetarian?" replaces non-vegetarian) while refinement
    follow-ups keep earlier context ("Make it spicy" keeps non-veg).
    """
    merged = dict(previous or {})

    for key, value in (current or {}).items():
        if value is not None:
            merged[key] = value

    return merged


def _prune_preferences(preferences):
    """Drop None entries so stored context holds only explicit values."""
    return {
        key: value
        for key, value in (preferences or {}).items()
        if value is not None
    }


def _search_alternatives_from_context(preferences, exclude_foods):
    """
    Find genuine catalogue alternatives for a continuation request
    ("Show more", "Suggest something else"), drawn ONLY from the real
    OXRO catalogue and excluding already-recommended items.

    Deterministic: category items first (stable catalogue order), then
    dietary/general fallback categories; filtered by the merged dietary
    preference; deduplicated; capped at MAX_FOLLOWUP_ALTERNATIVES.
    """
    exclude = {food.lower() for food in (exclude_foods or [])}

    candidates = []

    category = (preferences or {}).get("category")
    if category:
        for food in recommend_by_category(category, limit=20):
            candidates.append({"food": food, "category": category})

    food_type = (preferences or {}).get("food_type")

    if food_type == "vegetarian":
        candidates += _collect_from_categories(
            VEG_FALLBACK_CATEGORIES,
            per_category_limit=5
        )
    elif food_type == "non-vegetarian":
        candidates += _collect_from_categories(
            NONVEG_FALLBACK_CATEGORIES,
            per_category_limit=5
        )
    else:
        candidates += _collect_from_categories(
            GENERAL_FALLBACK_CATEGORIES,
            per_category_limit=3
        )

    # Respect the merged dietary preference, drop duplicates and
    # already-recommended items.
    candidates = filter_food_type(candidates, food_type)

    alternatives = []
    seen = set()

    for item in candidates:
        name_lower = item["food"].lower()

        if name_lower in exclude or name_lower in seen:
            continue

        seen.add(name_lower)
        alternatives.append(item)

        if len(alternatives) >= MAX_FOLLOWUP_ALTERNATIVES:
            break

    return alternatives


def _build_grounded_message(user_message, matched_foods, preferences):
    """
    Compose the message sent to the customer's chat session:
    the customer's message plus the catalogue context and guardrails.
    """
    if matched_foods:
        catalogue_section = (
            "MATCHED OXRO CATALOGUE ITEMS (recommend ONLY from this list):\n"
            + json.dumps(matched_foods)
        )
    else:
        catalogue_section = (
            "MATCHED OXRO CATALOGUE ITEMS: none found for this request."
        )

    return f"""
CUSTOMER MESSAGE:
{user_message}

{catalogue_section}

DETECTED PREFERENCES:
{json.dumps(preferences or {}, ensure_ascii=False)}

INSTRUCTIONS:
- Recommend only food items from the matched OXRO catalogue list above.
- If no items matched, honestly tell the customer that nothing matched
  and ask them to try a different request.
- Never invent food items, restaurant names, prices, ratings,
  availability or order status.
- The OXRO catalogue contains no price information. Never claim an item
  is cheaper or more expensive; if asked about price, explain honestly
  that prices are not available and must be confirmed in the app.
- Never claim an item is currently available, in stock, or orderable.
  Live availability must be confirmed by the backend.
- Reply naturally to the customer's message, keeping conversation context.
- Reply in the same language the customer used in their message
  (English, Tamil, Tanglish or a mix).
"""


def _build_followup_message(user_message, previous_foods, preferences):
    """
    Compose the grounded follow-up message for a refinement request
    that produced no new direct search matches. The previously matched
    catalogue items are provided so Gemini can respond using ONLY
    those real items - it must never invent new ones.
    """
    return f"""
CUSTOMER FOLLOW-UP MESSAGE:
{user_message}

CONVERSATION CONTEXT:
Earlier in this conversation, these OXRO catalogue items matched the
customer's request:
{json.dumps(previous_foods)}

MERGED PREFERENCES (across the conversation so far):
{json.dumps(preferences or {}, ensure_ascii=False)}

INSTRUCTIONS:
- The customer is refining or continuing their earlier request. Apply
  the follow-up to the earlier matched OXRO catalogue items above.
- Refer ONLY to the earlier matched catalogue items. Never invent food
  items, restaurant names, prices, ratings, availability or order
  status.
- If the customer asks for more or different items and no further OXRO
  catalogue options exist beyond those listed above, honestly say so
  instead of repeating or inventing items.
- The OXRO catalogue contains no price information. Never claim any
  item is cheaper or more expensive; if the customer asks about price,
  honestly explain that price information is not available and must be
  confirmed in the app.
- Never claim an item is currently available, in stock, or orderable.
  Live availability must be confirmed by the backend.
- Reply naturally, keeping conversation context.
- Reply in the same language the customer used in their message
  (English, Tamil, Tanglish or a mix).
"""


def _send_grounded(customer_id, grounded_message, session_manager):
    """Send through the customer's own chat session (context preserved)."""
    return session_manager.send_customer_message(
        customer_id,
        grounded_message
    )


def _finish_matched_turn(
    customer_id,
    user_message,
    result_items,
    merged_preferences,
    previous_foods,
    session_manager
):
    """
    Complete one turn that has matched catalogue items (used by BOTH
    the direct-match branch and the alternatives branch so ranking,
    grounding, context accumulation and the response shape stay
    identical):
        - rank the items with rank_food_results (before the prompt),
        - build and send the grounded message,
        - accumulate the shown list (deduplicated, newest appended)
          and store it with the merged preferences; update_context
          caps the list keeping the most recent entries.
    """
    ranked_foods = rank_food_results(
        user_message,
        result_items,
        merged_preferences
    )

    grounded_message = _build_grounded_message(
        user_message,
        ranked_foods,
        merged_preferences
    )

    reply = _send_grounded(customer_id, grounded_message, session_manager)

    shown_set = {food.lower() for food in previous_foods}
    combined_foods = list(previous_foods)

    for food in ranked_foods:
        if food.lower() not in shown_set:
            shown_set.add(food.lower())
            combined_foods.append(food)

    session_manager.update_context(
        customer_id,
        {
            "preferences": _prune_preferences(merged_preferences),
            "matched_foods": combined_foods
        }
    )

    return {
        "status": "success",
        "reply": reply,
        "detected_preferences": merged_preferences,
        "matched_foods": ranked_foods,
    }


def handle_food_message(customer_id, user_message, session_manager):
    """
    Catalogue-grounded food path of /chat (non-greeting messages only).

    Requires an initialized SessionManager (the API layer checks this
    and returns 503 not_ready when Gemini is unavailable).

    Pipeline:
        1. Search the catalogue for this message.
        2. Merge the freshly detected preferences over the customer's
           stored preferences (new explicit values override).
        3a. Matches found ->
              rank them, send a grounded message with the matched items
              and store merged preferences + shown items in context
              (shown list accumulates, so repeats are avoidable).
        3b. No matches, continuation request ("Show more",
            "Suggest something else"), stored context exists ->
              search the real catalogue for alternatives NOT previously
              recommended; if found, treat as a normal grounded turn.
        3c. No matches, stored context exists (refinement follow-up
            like "Make it spicy" / "Show me cheaper options") ->
              grounded follow-up message using ONLY the previously
              matched items; stored shown-list is preserved (only the
              merged preferences are refreshed).
        3d. No matches, no stored context ->
              honest "nothing matched" behaviour (unchanged).

    Returns:
        {
            "status": "success",
            "reply": str,
            "detected_preferences": dict (merged),
            "matched_foods": [str, ...]
        }

    Raises:
        GeminiQuotaError / GeminiTransientError /
        GeminiTimeoutError / GeminiServiceError
        (mapped to HTTP responses by the API layer).
    """
    search_result = smart_food_search(user_message)
    new_preferences = search_result["preferences"]

    stored_context = session_manager.get_context(customer_id)
    merged_preferences = _merge_preferences(
        stored_context["preferences"],
        new_preferences
    )
    previous_foods = stored_context["matched_foods"]

    result_items = list(search_result["results"])

    # Continuation requests must not repeat already-recommended items
    # whenever other catalogue options exist (requirement 5), and every
    # surfaced item must respect the MERGED dietary preference (e.g.
    # the general fallback of "Suggest something else" after a
    # non-vegetarian turn must not surface vegetarian items).
    if result_items and previous_foods and _is_continuation_request(user_message):
        result_items = filter_food_type(
            result_items,
            merged_preferences.get("food_type")
        )

        shown = {food.lower() for food in previous_foods}
        result_items = [
            item for item in result_items
            if item["food"].lower() not in shown
        ]

    # ---- 3a: direct matches -> normal grounded turn ----------------
    if result_items:
        return _finish_matched_turn(
            customer_id,
            user_message,
            result_items,
            merged_preferences,
            previous_foods,
            session_manager
        )

    # ---- 3b: continuation without direct matches -> search the real
    #      catalogue for unshown alternatives -----------------------
    if previous_foods and _is_continuation_request(user_message):
        alternatives = _search_alternatives_from_context(
            merged_preferences,
            previous_foods
        )

        if alternatives:
            return _finish_matched_turn(
                customer_id,
                user_message,
                alternatives,
                merged_preferences,
                previous_foods,
                session_manager
            )

    # ---- 3c: refinement follow-up -> grounded message using ONLY the
    #      previously matched items (shown-list preserved) -----------
    if previous_foods:
        grounded_message = _build_followup_message(
            user_message,
            previous_foods,
            merged_preferences
        )

        reply = _send_grounded(customer_id, grounded_message, session_manager)

        # Keep the shown list unchanged; refresh merged preferences so
        # refinements ("make it spicy") are remembered.
        session_manager.update_context(
            customer_id,
            {
                "preferences": _prune_preferences(merged_preferences),
                "matched_foods": previous_foods
            }
        )

        return {
            "status": "success",
            "reply": reply,
            "detected_preferences": merged_preferences,
            "matched_foods": previous_foods,
        }

    # ---- 3d: no matches, no context -> honest no-match (unchanged) --
    grounded_message = _build_grounded_message(
        user_message,
        [],
        merged_preferences
    )

    reply = _send_grounded(customer_id, grounded_message, session_manager)

    return {
        "status": "success",
        "reply": reply,
        "detected_preferences": merged_preferences,
        "matched_foods": [],
    }
