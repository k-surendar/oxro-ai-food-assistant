"""
Keyword-based food result ranking.

Migrated from the OXRO notebook (food_recommendation/ranking_engine.py).

Review fix (Phase 2 review, point 2):
    The notebook condition

        if category and food in results:
            score += 2

    was ineffective: `food in results` is True for every item (food is an
    element of the results list), so +2 was added uniformly and did not
    differentiate anything. It now compares each item's OWN category
    against the detected customer category:

        if category and item_category == category:
            score += 2

    The function accepts either:
        - a list of {"food": ..., "category": ...} dicts (preferred), or
        - a legacy list of plain food-name strings (categories are then
          resolved from the OXRO catalogue).

Note: the final food_search pipeline applies its own exact-match-first
ordering; this module is the shared ranking component available for the
Phase 4 chat integration and any future re-ranking needs.
"""

from food_recommendation.recommendation_engine import FOOD_CATEGORIES


def _resolve_category(food_name):
    """Legacy mode: look up a food name's category in the catalogue."""
    for category, items in FOOD_CATEGORIES.items():
        if food_name in items:
            return category
    return None


def rank_food_results(user_message, results, preferences):
    """
    Score food results against the customer's message and return them
    ordered by relevance (highest score first).

    Scoring:
        +10  exact food-name match in the query
        +3   per query word contained in the food name (words > 2 chars)
        +2   item's own category matches the detected customer category
    """
    query = user_message.lower()
    detected_category = preferences.get("category")

    ranked_results = []

    for entry in results:

        if isinstance(entry, dict):
            food = entry["food"]
            item_category = entry.get("category")
        else:
            food = entry
            item_category = _resolve_category(food)

        food_lower = food.lower()
        score = 0

        # Exact food-name match
        if food_lower in query:
            score += 10

        # Individual keyword matches
        for word in query.split():
            if len(word) > 2 and word in food_lower:
                score += 3

        # Category relevance: only when the ITEM's category matches the
        # detected customer category (previously ineffective condition).
        if detected_category and item_category == detected_category:
            score += 2

        ranked_results.append((food, score))

    ranked_results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return [food for food, score in ranked_results]
