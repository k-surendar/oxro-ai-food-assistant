"""
Core food catalogue access module.

Migrated from the OXRO notebook (food_recommendation/recommendation_engine.py).

Purpose:
    - Loads the OXRO food catalogue (food_categories.json) exactly once.
    - Provides category listing, category recommendations, keyword search
      and preference-based recommendations.

Migration fix:
    - recommend_by_category() previously used random.sample(), which made API
      responses non-deterministic (flaky for the mobile/web teams and tests).
      It now returns items in stable catalogue order.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOOD_FILE = os.path.join(BASE_DIR, "food_categories.json")

with open(FOOD_FILE, "r", encoding="utf-8") as file:
    FOOD_CATEGORIES = json.load(file)


def get_categories():
    """Return all main food category names."""
    return list(FOOD_CATEGORIES.keys())


def recommend_by_category(category, limit=5):
    """
    Return up to `limit` food items for a category in stable catalogue order.

    Changed from the notebook version: no random sampling, so results are
    deterministic and testable.
    """
    if category not in FOOD_CATEGORIES:
        return []

    items = FOOD_CATEGORIES[category]

    return items[:limit]


def search_food(food_name):
    """
    Substring search across the whole catalogue.

    Returns a list of {"category": ..., "food": ...} matches.
    """
    results = []

    for category, items in FOOD_CATEGORIES.items():
        for item in items:
            if food_name.lower() in item.lower():
                results.append({
                    "category": category,
                    "food": item
                })

    return results


def recommend_by_preference(preference, limit=5):
    """
    Return food items matching a coarse dietary preference
    ("vegetarian" or "non-vegetarian").
    """
    preference = preference.lower()

    if preference == "vegetarian":
        categories = [
            "Vegetarian",
            "Healthy Food",
            "South Indian"
        ]

    elif preference == "non-vegetarian":
        categories = [
            "Chicken",
            "Mutton",
            "Seafood",
            "Biryani"
        ]

    else:
        return []

    results = []

    for category in categories:
        for item in FOOD_CATEGORIES.get(category, []):
            results.append(item)

    return results[:limit]
