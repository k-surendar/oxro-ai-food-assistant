"""
Rule-based customer preference classifier.

Migrated from the OXRO notebook (final regex-based iteration).
Logic preserved unchanged, including the word-boundary regex fix that makes
"I don't want spicy food" classify as mild instead of spicy.

Phase 5: understands English, Tamil script, Tanglish and mixed-language
messages. Tamil/Tanglish phrases are ADDITIVE checks - every existing
English behaviour is preserved. Detection is phrase/stem based (see
language_phrases.py); the messages are Unicode-normalized first.
Dietary detection remains preference-level only - it never claims
ingredient-level dietary safety.
"""

import re

from food_recommendation.language_phrases import (
    BUDGET_ENTRIES,
    BUDGET_PHRASES,
    HEALTHY_ENTRIES,
    MILD_PHRASES,
    NEGATION_WORDS,
    NONVEG_ENTRIES,
    SWEET_ENTRIES,
    SPICY_ENTRIES,
    VEG_ENTRIES,
    canonicalize_food_words,
    contains_phrase,
    matches_any_entry,
    normalize_message,
)


def classify_preferences(user_message):
    """
    Detect food_type, taste, health, budget and category signals
    from a customer message (English, Tamil, Tanglish or a mix).
    """
    # Unicode-safe normalization, then Tanglish/Tamil food words are
    # canonicalized to their English catalogue words ("koli biriyani"
    # -> "chicken biryani") so the existing English patterns below work
    # unchanged for them.
    message = normalize_message(user_message)
    message = canonicalize_food_words(message)

    preferences = {
        "food_type": None,
        "taste": None,
        "health": None,
        "budget": None,
        "category": None
    }

    # 1. Food type
    non_veg_patterns = [
        "non-veg", "non veg", "nonvegetarian",
        "non-vegetarian", "chicken", "mutton",
        "fish", "seafood", "prawn"
    ]

    veg_patterns = ["vegetarian", "veg food", "veg meal"]

    if any(word in message for word in non_veg_patterns):
        preferences["food_type"] = "non-vegetarian"

    elif any(word in message for word in veg_patterns) or re.search(r"\bveg\b", message):
        preferences["food_type"] = "vegetarian"

    # Tamil/Tanglish phrases the canonicalization cannot express
    # (asaivam/saivam families and colloquial "kari").
    if preferences["food_type"] is None:
        if matches_any_entry(message, NONVEG_ENTRIES):
            preferences["food_type"] = "non-vegetarian"

        elif matches_any_entry(message, VEG_ENTRIES):
            preferences["food_type"] = "vegetarian"

    # 2. Taste preference
    mild_patterns = [
        r"\bdon't\s+want\s+spicy\b",
        r"\bdont\s+want\s+spicy\b",
        r"\bdo\s+not\s+want\s+spicy\b",
        r"\bdon't\s+like\s+spicy\b",
        r"\bnot\s+spicy\b",
        r"\bless\s+spicy\b",
        r"\bnot\s+too\s+spicy\b",
        r"\bno\s+spice\b",
        r"\bwithout\s+spice\b",
        r"\bavoid\s+spicy\b",
        r"\bmild\b"
    ]

    spicy_patterns = [
        r"\bspicy\b",
        r"\bhot\b",
        r"\bmasala\b"
    ]

    if any(re.search(pattern, message) for pattern in mild_patterns):
        preferences["taste"] = "mild"

    elif any(re.search(pattern, message) for pattern in spicy_patterns):
        preferences["taste"] = "spicy"

    # Tamil/Tanglish taste phrases (mild checked before spicy so
    # "karam illa" stays mild; the English negation patterns above
    # already work the same way).
    if preferences["taste"] is None:
        if any(contains_phrase(message, phrase) for phrase in MILD_PHRASES):
            preferences["taste"] = "mild"

        elif matches_any_entry(message, SPICY_ENTRIES):
            preferences["taste"] = "spicy"

    # Sweet (new Phase 5 taste value): English "sweet" plus the
    # Tamil/Tanglish inippu family.
    sweet_detected = False

    if preferences["taste"] is None:
        if re.search(r"\bsweet\b", message) or matches_any_entry(
            message, SWEET_ENTRIES
        ):
            preferences["taste"] = "sweet"
            sweet_detected = True

    # 3. Health preference
    health_patterns = [
        "healthy", "diet", "low calorie",
        "protein", "high protein", "nutritious"
    ]

    if any(word in message for word in health_patterns):
        preferences["health"] = "healthy"

    elif matches_any_entry(message, HEALTHY_ENTRIES):
        preferences["health"] = "healthy"

    # 4. Budget preference
    budget_patterns = [
        "cheap", "budget", "affordable",
        "low cost", "economical"
    ]

    if any(word in message for word in budget_patterns):
        preferences["budget"] = "budget-friendly"

    elif matches_any_entry(message, BUDGET_ENTRIES) or any(
        contains_phrase(message, phrase) for phrase in BUDGET_PHRASES
    ):
        preferences["budget"] = "budget-friendly"

    # 5. Food category
    categories = [
        "Biryani", "South Indian", "North Indian",
        "Chinese", "Burgers", "Pizza", "Seafood",
        "Desserts", "Breakfast", "Italian",
        "Mexican", "Japanese", "Korean"
    ]

    for category in categories:
        if category.lower() in message:
            preferences["category"] = category
            break

    # Negation-aware category: "biryani venam" (don't want biryani)
    # must NOT trigger the Biryani category fallback.
    if preferences["category"] is not None:
        tokens = message.split()
        negated_words = {
            tokens[index]
            for index in range(len(tokens) - 1)
            if tokens[index + 1] in NEGATION_WORDS
        }

        if preferences["category"].lower() in negated_words:
            preferences["category"] = None

    # Explicit sweet requests map to the Desserts category when no
    # other category was detected ("where appropriate" per Phase 5
    # decision 1).
    if preferences["category"] is None and sweet_detected:
        preferences["category"] = "Desserts"

    return preferences
