"""
Catalogue preservation tests (Phase 2 requirement: all 291 entries intact).
"""

from food_recommendation.recommendation_engine import (
    FOOD_CATEGORIES,
    get_categories,
)


def test_category_count():
    assert len(FOOD_CATEGORIES) == 30


def test_total_food_entries():
    total = sum(len(items) for items in FOOD_CATEGORIES.values())
    assert total == 291


def test_spot_check_entries():
    assert "Chicken Biryani" in FOOD_CATEGORIES["Biryani"]
    assert "Idli" in FOOD_CATEGORIES["South Indian"]
    assert "Falooda" in FOOD_CATEGORIES["Ice Cream"]
    assert "Chicken Shawarma" in FOOD_CATEGORIES["Arabian"]
    assert "Paneer Butter Masala" in FOOD_CATEGORIES["North Indian"]


def test_get_categories_matches_catalogue():
    assert get_categories() == list(FOOD_CATEGORIES.keys())
