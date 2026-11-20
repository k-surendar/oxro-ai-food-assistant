"""
Phase 4 Step 2 tests: context-aware chat pipeline.

Covers:
    - preference merging with correct overrides
    - contextual follow-ups ("Make it spicy", "Show me cheaper options")
    - continuation requests ("Show more", "Suggest something else")
      with no repeats of previously recommended items
    - food_type override ("What about vegetarian?")
    - honest price limitation (catalogue has no prices)
    - context isolation, expiry and end-session behaviour
    - ranking applied before the grounded prompt

All tests use the real SessionManager with a FakeChat - no Gemini,
no network, no API keys.
"""

import time

import pytest

import ai.chat_service as chat_service_module
from ai.chat_service import (
    _is_continuation_request,
    _merge_preferences,
    _search_alternatives_from_context,
    handle_food_message,
)
from ai.session_manager import SessionManager
from food_recommendation.food_search import (
    NONVEG_FALLBACK_CATEGORIES,
    _collect_from_categories,
)
from food_recommendation.recommendation_engine import (
    FOOD_CATEGORIES,
    recommend_by_category,
)


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
        **kwargs
    )


def last_grounded_message(sm, customer_id):
    return sm.sessions[customer_id]["chat"].history[-1]


def all_catalogue_foods():
    return {
        food
        for items in FOOD_CATEGORIES.values()
        for food in items
    }


# ---------- preference merging (requirement: overrides) ----------

def test_merge_preferences_new_explicit_overrides_old():
    merged = _merge_preferences(
        {"food_type": "non-vegetarian", "taste": "spicy", "budget": None},
        {"food_type": "vegetarian", "taste": None, "budget": "budget-friendly"}
    )

    assert merged["food_type"] == "vegetarian"   # explicit switch wins
    assert merged["taste"] == "spicy"            # None inherits stored
    assert merged["budget"] == "budget-friendly"


def test_merge_preferences_none_inherits_and_empty_stored():
    merged = _merge_preferences({}, {"taste": "spicy"})

    assert merged == {"taste": "spicy"}


def test_is_continuation_request_variants():
    assert _is_continuation_request("Show more.")
    assert _is_continuation_request("Suggest something else.")
    assert _is_continuation_request("Give me more options")

    # Refinement follow-ups are NOT continuation requests.
    assert not _is_continuation_request("Make it spicy.")
    assert not _is_continuation_request("Show me cheaper options.")
    assert not _is_continuation_request("What about vegetarian?")


# ---------- first turn: grounding + context storage ----------

def test_first_turn_grounds_reply_and_stores_context():
    sm = make_manager()

    result = handle_food_message("C1", "I want chicken biryani", sm)

    assert result["status"] == "success"
    assert result["matched_foods"] == ["Chicken Biryani"]
    assert result["detected_preferences"]["food_type"] == "non-vegetarian"

    grounded = last_grounded_message(sm, "C1")
    assert "Chicken Biryani" in grounded
    assert "recommend ONLY from this list" in grounded

    context = sm.get_context("C1")
    assert context["matched_foods"] == ["Chicken Biryani"]
    assert context["preferences"]["food_type"] == "non-vegetarian"
    assert context["preferences"]["category"] == "Biryani"


# ---------- refinement follow-ups (branch 3c) ----------

def test_make_it_spicy_uses_previous_context():
    sm = make_manager()
    handle_food_message("C1", "I want chicken biryani", sm)

    result = handle_food_message("C1", "Make it spicy.", sm)

    assert result["status"] == "success"
    # Response carries the previously matched items (no new search).
    assert result["matched_foods"] == ["Chicken Biryani"]

    grounded = last_grounded_message(sm, "C1")
    assert "CUSTOMER FOLLOW-UP MESSAGE" in grounded
    assert "Chicken Biryani" in grounded
    # Merged preferences travel with the follow-up.
    assert '"taste": "spicy"' in grounded
    assert '"food_type": "non-vegetarian"' in grounded

    # Shown list preserved; merged preferences refreshed.
    context = sm.get_context("C1")
    assert context["matched_foods"] == ["Chicken Biryani"]
    assert context["preferences"]["taste"] == "spicy"
    assert context["preferences"]["food_type"] == "non-vegetarian"


def test_cheaper_options_explains_price_limitation_honestly():
    sm = make_manager()
    handle_food_message("C1", "I want chicken biryani", sm)

    result = handle_food_message("C1", "Show me cheaper options.", sm)

    assert result["status"] == "success"
    assert result["matched_foods"] == ["Chicken Biryani"]

    grounded = last_grounded_message(sm, "C1")
    # Whitespace-normalized so line wraps do not affect assertions.
    normalized = " ".join(grounded.split())
    # Honest price limitation: never claim cheaper without price data.
    assert "no price information" in normalized
    assert "Never claim any item is cheaper or more expensive" in normalized

    # Budget preference merged and remembered.
    context = sm.get_context("C1")
    assert context["preferences"]["budget"] == "budget-friendly"


# ---------- continuation requests (branch 3a exclusion + 3b) ----------

NONVEG_WORDS = (
    "chicken", "mutton", "fish", "prawn",
    "beef", "egg", "seafood", "pepperoni"
)


def test_show_more_returns_new_alternatives_without_repeats():
    sm = make_manager()
    handle_food_message("C1", "I want chicken biryani", sm)

    result = handle_food_message("C1", "Show more.", sm)

    assert result["status"] == "success"

    foods = result["matched_foods"]
    assert len(foods) > 0
    # Never repeats the already-recommended item.
    assert "Chicken Biryani" not in foods
    # Grounded: every alternative comes from the real OXRO catalogue.
    catalogue = all_catalogue_foods()
    for food in foods:
        assert food in catalogue
    # Every alternative respects the MERGED dietary preference
    # (non-vegetarian was stored in context).
    for food in foods:
        assert any(word in food.lower() for word in NONVEG_WORDS)


def test_show_more_accumulates_shown_list_in_context():
    sm = make_manager()
    handle_food_message("C1", "I want chicken biryani", sm)
    result = handle_food_message("C1", "Show more.", sm)

    context = sm.get_context("C1")

    assert "Chicken Biryani" in context["matched_foods"]
    for food in result["matched_foods"]:
        assert food in context["matched_foods"]
    # No duplicates accumulated.
    lowered = [food.lower() for food in context["matched_foods"]]
    assert len(lowered) == len(set(lowered))


def test_suggest_something_else_excludes_previous_items():
    sm = make_manager()
    handle_food_message("C1", "I want chicken biryani", sm)

    result = handle_food_message("C1", "Suggest something else.", sm)

    assert result["status"] == "success"
    foods = result["matched_foods"]

    assert len(foods) > 0
    assert "Chicken Biryani" not in foods

    catalogue = all_catalogue_foods()
    for food in foods:
        assert food in catalogue
    # The merged non-vegetarian preference is respected even though the
    # direct-match fallback for this message is the general one.
    for food in foods:
        assert any(word in food.lower() for word in NONVEG_WORDS)


def test_multi_turn_show_more_excludes_all_previously_recommended():
    """
    Multi-turn continuation: initial recommendation -> "Show more" ->
    "Show more". Previously recommended foods are always excluded
    whenever alternatives exist; when the alternatives are exhausted,
    the response must fall back to the earlier items WITHOUT inventing
    anything new.
    """
    sm = make_manager()
    handle_food_message("C1", "I want chicken biryani", sm)

    first = handle_food_message("C1", "Show more.", sm)
    catalogue = all_catalogue_foods()

    shown = {"chicken biryani"}
    assert len(first["matched_foods"]) > 0
    for food in first["matched_foods"]:
        assert food.lower() not in shown
        assert food in catalogue
        shown.add(food.lower())

    second = handle_food_message("C1", "Show more.", sm)
    second_foods = second["matched_foods"]

    new_foods = [
        food for food in second_foods
        if food.lower() not in shown
    ]

    if new_foods:
        # Alternatives branch: when ANY genuinely new item is returned,
        # EVERY item in the response must be absent from the previously
        # shown set (no repeats may be mixed in), and every returned
        # item must be a genuine OXRO catalogue entry.
        for food in second_foods:
            assert food.lower() not in shown
            assert food in catalogue
    else:
        # Alternatives exhausted: the response contains ONLY previously
        # shown catalogue items and NOTHING new is invented.
        assert set(food.lower() for food in second_foods) <= shown
        for food in second_foods:
            assert food in catalogue

        grounded = last_grounded_message(sm, "C1")
        normalized = " ".join(grounded.split())
        assert "CUSTOMER FOLLOW-UP MESSAGE" in normalized
        assert "no further OXRO catalogue options exist" in normalized

    # Context accumulation stays deduplicated.
    context = sm.get_context("C1")["matched_foods"]
    lowered = [food.lower() for food in context]
    assert len(lowered) == len(set(lowered))


def test_alternatives_search_excluding_everything_returns_empty():
    # Exclude every candidate the alternatives search could produce for
    # these preferences - it must return [] and never invent items.
    exclude = []

    for food in recommend_by_category("Biryani", limit=20):
        exclude.append(food)

    for item in _collect_from_categories(
        NONVEG_FALLBACK_CATEGORIES,
        per_category_limit=5
    ):
        exclude.append(item["food"])

    alternatives = _search_alternatives_from_context(
        {"category": "Biryani", "food_type": "non-vegetarian"},
        exclude
    )

    assert alternatives == []


# ---------- preference override through the pipeline ----------

def test_what_about_vegetarian_overrides_food_type():
    sm = make_manager()
    handle_food_message("C1", "I want chicken biryani", sm)

    result = handle_food_message("C1", "What about vegetarian?", sm)

    assert result["status"] == "success"
    # The explicit switch must replace the stored preference...
    assert result["detected_preferences"]["food_type"] == "vegetarian"
    # ...and the returned items respect it (name-based filter).
    nonveg_words = (
        "chicken", "mutton", "fish", "prawn",
        "beef", "egg", "seafood", "pepperoni"
    )
    for food in result["matched_foods"]:
        assert not any(word in food.lower() for word in nonveg_words)

    context = sm.get_context("C1")
    assert context["preferences"]["food_type"] == "vegetarian"


# ---------- honest no-match (branch 3d, unchanged behaviour) ----------

def test_no_context_no_match_is_honest_and_stores_nothing():
    sm = make_manager()

    result = handle_food_message("C1", "zzzqqq", sm)

    assert result["status"] == "success"
    assert result["matched_foods"] == []

    grounded = last_grounded_message(sm, "C1")
    assert "none found for this request" in grounded
    assert "honestly tell the customer that nothing matched" in grounded


# ---------- isolation / expiry / end-session ----------

def test_context_is_isolated_between_customers():
    sm = make_manager()

    handle_food_message("C_A", "I want chicken biryani", sm)
    handle_food_message("C_B", "I want veg pizza", sm)

    context_a = sm.get_context("C_A")
    context_b = sm.get_context("C_B")

    assert context_a["preferences"]["food_type"] == "non-vegetarian"
    assert context_b["preferences"]["food_type"] == "vegetarian"
    assert "Chicken Biryani" in context_a["matched_foods"]
    assert "Chicken Biryani" not in context_b["matched_foods"]

    # Follow-ups never leak across customers.
    result = handle_food_message("C_B", "Make it spicy.", sm)
    # C_B has veg pizza context, not biryani.
    assert result["matched_foods"] != ["Chicken Biryani"]


def test_followup_after_session_expiry_behaves_as_fresh():
    sm = make_manager(session_ttl_seconds=1)
    handle_food_message("C1", "I want chicken biryani", sm)

    time.sleep(1.2)

    # Context is gone with the expired session: the refinement must be
    # treated as a fresh unmatched request (honest no-match).
    result = handle_food_message("C1", "Make it spicy.", sm)

    assert result["matched_foods"] == []
    grounded = last_grounded_message(sm, "C1")
    assert "none found for this request" in grounded

    # Fresh session, fresh empty context.
    assert sm.get_context("C1") == {"preferences": {}, "matched_foods": []}


def test_followup_after_end_session_behaves_as_fresh():
    sm = make_manager()
    handle_food_message("C1", "I want chicken biryani", sm)

    sm.end_session("C1")

    result = handle_food_message("C1", "Make it spicy.", sm)

    assert result["matched_foods"] == []
    grounded = last_grounded_message(sm, "C1")
    assert "none found for this request" in grounded
    assert sm.get_context("C1") == {"preferences": {}, "matched_foods": []}


# ---------- ranking integration (requirement 7) ----------

def test_rank_food_results_applied_to_direct_and_alternative_matches(monkeypatch):
    """
    Requirement: ranking is applied consistently to BOTH the direct
    matches branch and the alternatives branch, always before the
    grounded prompt is built.
    """
    calls = []
    original = chat_service_module.rank_food_results

    def spy(user_message, results, preferences):
        calls.append({
            "message": user_message,
            "n_results": len(results),
            "preferences": preferences,
        })
        return original(user_message, results, preferences)

    monkeypatch.setattr(chat_service_module, "rank_food_results", spy)

    sm = make_manager()
    handle_food_message("C1", "I want chicken biryani", sm)
    handle_food_message("C1", "Show more.", sm)

    # One ranking call for the direct matches, one for the alternatives.
    assert len(calls) == 2
    assert calls[0]["n_results"] >= 1
    assert calls[1]["n_results"] >= 1
    assert calls[1]["preferences"]["food_type"] == "non-vegetarian"
