"""
Food search / classifier / ranking behaviour tests.

Expected outputs are taken from the notebook's verified test runs so the
migrated code demonstrably preserves existing working behaviour.
"""

from food_recommendation.food_search import (
    NONVEG_KEYWORDS,
    smart_food_search,
)
from food_recommendation.preference_classifier import classify_preferences


def _foods(result):
    return [item["food"] for item in result["results"]]


def test_chicken_biryani_exact_match():
    result = smart_food_search("I want chicken biryani")
    assert _foods(result) == ["Chicken Biryani"]
    assert result["preferences"]["food_type"] == "non-vegetarian"


def test_chicken_fried_rice_exact_match():
    result = smart_food_search("Suggest chicken fried rice")
    assert _foods(result) == ["Chicken Fried Rice"]


def test_paneer_pizza_exact_match():
    result = smart_food_search("I want paneer pizza")
    assert _foods(result) == ["Paneer Pizza"]


def test_vegetarian_pizza_filters_nonveg():
    result = smart_food_search("I want vegetarian pizza")
    foods = _foods(result)

    assert foods == [
        "Cheese Pizza",
        "Farmhouse Pizza",
        "Margherita Pizza",
        "Mushroom Pizza",
        "Paneer Pizza",
        "Pizza Combo",
        "Veg Pizza",
    ]


def test_no_match_returns_empty():
    result = smart_food_search("zzzqqq")
    assert result["results"] == []


# ---------- broad request tests (review point 1) ----------

def test_broad_vegetarian_request_returns_results():
    result = smart_food_search("Suggest vegetarian food")
    foods = _foods(result)

    assert len(foods) > 0
    assert result["preferences"]["food_type"] == "vegetarian"

    # Name-based veg filter must hold for every returned item.
    for food in foods:
        assert not any(k in food.lower() for k in NONVEG_KEYWORDS)


def test_broad_nonvegetarian_request_returns_results():
    result = smart_food_search("Suggest non-vegetarian food")
    foods = _foods(result)

    assert len(foods) > 0
    assert result["preferences"]["food_type"] == "non-vegetarian"

    # Name-based non-veg filter: every item carries a non-veg keyword.
    for food in foods:
        assert any(k in food.lower() for k in NONVEG_KEYWORDS)


def test_broad_general_request_returns_popular_results():
    result = smart_food_search("Suggest some food")
    foods = _foods(result)

    assert len(foods) > 0


def test_veg_request_without_signal_word_returns_results():
    # No searchable keywords and no request-signal words - the dietary
    # fallback alone must still produce results.
    result = smart_food_search("I am veg")
    foods = _foods(result)

    assert len(foods) > 0
    assert result["preferences"]["food_type"] == "vegetarian"


def test_broad_vegetarian_request_is_deterministic():
    first = smart_food_search("Suggest vegetarian food")
    second = smart_food_search("Suggest vegetarian food")

    assert _foods(first) == _foods(second)


def test_broad_nonvegetarian_request_has_no_duplicates():
    result = smart_food_search("Suggest non-vegetarian food")
    foods = _foods(result)

    assert len(foods) == len(set(foods))


def test_classifier_spicy_biryani():
    prefs = classify_preferences("I want spicy chicken biryani")
    assert prefs["food_type"] == "non-vegetarian"
    assert prefs["taste"] == "spicy"
    assert prefs["category"] == "Biryani"


def test_classifier_not_spicy_is_mild():
    prefs = classify_preferences("I don't want spicy food")
    assert prefs["taste"] == "mild"


def test_classifier_vegetarian():
    prefs = classify_preferences("Suggest some vegetarian food")
    assert prefs["food_type"] == "vegetarian"
