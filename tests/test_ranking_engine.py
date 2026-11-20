"""
Ranking engine tests (Phase 2 review point 2).

Verifies the corrected category-relevance scoring: the old condition
(`food in results`) added +2 uniformly to every item; the corrected one
adds +2 only when the item's own category matches the detected category.
"""

from food_recommendation.ranking_engine import rank_food_results


def test_category_relevance_uses_item_own_category():
    results = [
        {"food": "Chicken Fried Rice", "category": "Chinese"},
        {"food": "Hyderabadi Biryani", "category": "Biryani"},
    ]
    preferences = {"category": "Biryani"}

    ranked = rank_food_results("I want biryani", results, preferences)

    assert ranked == ["Hyderabadi Biryani", "Chicken Fried Rice"]


def test_category_relevance_legacy_string_results():
    # Legacy mode: plain names resolved from the catalogue.
    results = ["Chicken Biryani", "Fried Chicken"]
    preferences = {"category": "Biryani"}

    ranked = rank_food_results("I want biryani", results, preferences)

    assert ranked == ["Chicken Biryani", "Fried Chicken"]


def test_no_category_detected_no_bonus():
    results = [
        {"food": "Chicken Fried Rice", "category": "Chinese"},
        {"food": "Veg Pizza", "category": "Pizza"},
    ]
    preferences = {"category": None}

    ranked = rank_food_results("suggest rice", results, preferences)

    assert ranked == ["Chicken Fried Rice", "Veg Pizza"]


def test_wrong_category_no_bonus():
    results = [
        {"food": "Chicken Fried Rice", "category": "Chinese"},
        {"food": "Veg Pizza", "category": "Pizza"},
    ]
    preferences = {"category": "Biryani"}

    ranked = rank_food_results("suggest rice", results, preferences)

    # No item belongs to Biryani, so no +2 bonus; keyword match
    # ("rice" in "chicken fried rice") decides the order.
    assert ranked == ["Chicken Fried Rice", "Veg Pizza"]
