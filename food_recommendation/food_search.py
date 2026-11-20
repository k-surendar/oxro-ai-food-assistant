"""
Smart food search over the OXRO catalogue.

Migrated from the OXRO notebook (final 5th iteration of food_search.py),
with review-pass additions for broad requests.

Search pipeline:

    1. Extract meaningful keywords from the message (STOPWORDS filtered).
    2. Search the catalogue per keyword and keep only items matching
       ALL extracted keywords.
    3. Fall back to the detected category when no exact match exists.
    4. NEW: fall back to dietary categories when a food_type
       (vegetarian / non-vegetarian) was detected but no items matched
       (covers broad requests like "Suggest vegetarian food").
    5. NEW: fall back to a small deterministic set of popular items when
       the message is a genuine broad food request with no detectable
       preference (covers "Suggest some food"). Junk input ("zzzqqq")
       still returns no results.
    6. Apply the dietary (veg/non-veg) filter.
    7. Prioritise exact food-name matches.

IMPORTANT LIMITATION - NAME-BASED DIETARY FILTERING (review point 5):
    The vegetarian / non-vegetarian filter works on FOOD NAMES ONLY.
    It is NOT ingredient-level verification. Examples:
        - "Grilled Chicken Salad" is excluded from vegetarian results
          because the name contains "chicken" (name-based, correct here).
        - A hypothetically non-vegetarian dish whose name contains no
          meat word (e.g. "Fried Rice" that includes egg) would be
          treated as vegetarian. Ingredient-level classification requires
          structured menu data from the OXRO backend (Phase 4+).

Phase 5 (multilingual):
    Messages are Unicode-normalized and Tanglish/Tamil food words are
    canonicalized to English catalogue words before keyword extraction
    (see language_phrases.py). Tanglish fillers are dropped, negation
    words keep their meaning ("biryani venam" = don't want biryani).
"""

from food_recommendation.language_phrases import (
    NEGATION_WORDS,
    TANGLISH_FILLER_WORDS,
    TANGLISH_REQUEST_SIGNAL_WORDS,
    canonicalize_food_words,
    normalize_message,
)
from food_recommendation.preference_classifier import classify_preferences
from food_recommendation.recommendation_engine import (
    FOOD_CATEGORIES,
    search_food,
)


NONVEG_KEYWORDS = [
    "chicken", "mutton", "fish", "prawn",
    "beef", "egg", "seafood", "pepperoni"
]

STOPWORDS = {
    "i", "want", "some", "suggest", "show",
    "me", "give", "please", "food", "under",
    "the", "a", "an", "for", "and", "with",
    "healthy", "spicy", "mild", "less", "oily",
    "non", "vegetarian", "veg"
}

# Phase 5: Tanglish/Tamil filler words. Negation words ("illa",
# "venam", ...) are deliberately NOT fillers - they carry meaning and
# are handled by the negation logic below.
STOPWORDS.update(TANGLISH_FILLER_WORDS)

# Categories used for broad dietary fallbacks (review point 1).
VEG_FALLBACK_CATEGORIES = [
    "Vegetarian", "Healthy Food", "South Indian"
]

NONVEG_FALLBACK_CATEGORIES = [
    "Chicken", "Mutton", "Seafood", "Biryani"
]

# Deterministic general fallback for broad requests with no detected
# preference ("Suggest some food").
GENERAL_FALLBACK_CATEGORIES = [
    "Biryani", "South Indian", "Chinese", "Pizza", "Burgers"
]

# Words indicating the customer is genuinely asking about food, used to
# gate the general fallback so gibberish still returns no results.
REQUEST_SIGNAL_WORDS = {
    "food", "eat", "eating", "hungry", "hunger",
    "suggest", "recommend", "recommendation",
    "dishes", "dish", "meal", "meals", "dinner", "lunch"
}

# Phase 5: Tanglish/Tamil equivalents of the same signal words.
REQUEST_SIGNAL_WORDS.update(TANGLISH_REQUEST_SIGNAL_WORDS)


def _has_request_signal(message):
    """True when the message contains a broad food-request signal word."""
    words = set(message.replace(",", " ").split())
    return bool(words & REQUEST_SIGNAL_WORDS)


def _extract_keywords(message):
    """
    Tokenize a canonicalized message and drop:
        - negated words ("biryani venam" removes BOTH tokens, so
          "enakku biryani venam, dosa venum" searches for dosa only),
        - stopwords/short tokens.
    Negation words themselves ("illa", "venam", ...) are never treated
    as food keywords.
    """
    kept = []

    for token in message.split():
        if token in NEGATION_WORDS:
            if kept:
                # The word just before the negation is being negated.
                kept.pop()
            continue

        kept.append(token)

    return [
        word for word in kept
        if word not in STOPWORDS and len(word) > 2
    ]


def _collect_from_categories(categories, per_category_limit):
    """
    Build {"food", "category"} items from the given categories in
    stable catalogue order (deterministic).
    """
    collected = []

    for category in categories:
        for food in FOOD_CATEGORIES.get(category, [])[:per_category_limit]:
            collected.append({"food": food, "category": category})

    return collected


def filter_food_type(results, food_type):
    """
    Filter catalogue results according to the detected food_type.

    NOTE: this is name-based filtering (see module docstring) - it is not
    ingredient-level verification.
    """
    if not food_type:
        return results

    filtered = []

    for item in results:
        name = item["food"].lower()

        is_nonveg = any(
            word in name for word in NONVEG_KEYWORDS
        )

        if food_type == "vegetarian" and not is_nonveg:
            filtered.append(item)

        elif food_type == "non-vegetarian" and is_nonveg:
            filtered.append(item)

    return filtered


def smart_food_search(user_message):
    """
    Full search pipeline. Returns:
        {
            "query": original message,
            "preferences": classifier output,
            "results": [{"food": ..., "category": ...}, ...]
        }
    """
    preferences = classify_preferences(user_message)

    # Unicode-safe normalization, then Tanglish/Tamil food words are
    # canonicalized to their English catalogue words.
    message = canonicalize_food_words(normalize_message(user_message))

    # Extract meaningful food keywords (with negation handling)
    keywords = _extract_keywords(message)

    # Collect possible food matches
    candidates = {}

    for keyword in keywords:
        for item in search_food(keyword):
            candidates[item["food"].lower()] = item

    # Keep results matching ALL extracted keywords
    results = []

    for item in candidates.values():
        food_name = item["food"].lower()

        if all(keyword in food_name for keyword in keywords):
            results.append(item)

    # 1) Use category recommendations if no exact matches exist
    if not results and preferences.get("category"):
        from food_recommendation.recommendation_engine import recommend_by_category

        category_items = recommend_by_category(
            preferences["category"],
            limit=20
        )

        results = [
            {"food": food, "category": preferences["category"]}
            for food in category_items
        ]

    # 2) Dietary fallback: food_type detected but nothing matched
    #    (e.g. "Suggest vegetarian food" has no searchable keywords)
    if not results and preferences.get("food_type") == "vegetarian":
        results = _collect_from_categories(
            VEG_FALLBACK_CATEGORIES,
            per_category_limit=5
        )

    elif not results and preferences.get("food_type") == "non-vegetarian":
        results = _collect_from_categories(
            NONVEG_FALLBACK_CATEGORIES,
            per_category_limit=5
        )

    # 3) General fallback: genuine broad food request with no detectable
    #    preference (e.g. "Suggest some food"). Junk input still yields
    #    no results.
    if not results and _has_request_signal(message):
        results = _collect_from_categories(
            GENERAL_FALLBACK_CATEGORIES,
            per_category_limit=2
        )

    # Apply dietary preference filter
    results = filter_food_type(
        results,
        preferences.get("food_type")
    )

    # Deduplicate by food name (an item can exist in more than one
    # catalogue category, e.g. "Mutton Biryani" in Mutton and Biryani).
    deduped = {}
    for item in results:
        deduped.setdefault(item["food"].lower(), item)
    results = list(deduped.values())

    # Prioritize exact food-name matches
    results.sort(
        key=lambda item: (
            item["food"].lower() != message.strip(),
            item["food"].lower()
        )
    )

    return {
        "query": user_message,
        "preferences": preferences,
        "results": results
    }
