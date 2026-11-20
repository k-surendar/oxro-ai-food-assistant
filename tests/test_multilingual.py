"""
Phase 5 Step 2 tests: Tamil / English / Tanglish / mixed-language support.

Covers:
    - Unicode normalization (NFC/NFD, zero-width, punctuation,
      apostrophes, underscores)
    - Tamil script and Tanglish preference detection
    - Sweet taste + Desserts category mapping
    - Tanglish food-word canonicalization in search (biriyani -> biryani)
    - Negation handling ("biryani venam" = don't want biryani)
    - Junk-input regression (Tamil and Tanglish junk -> no results)
    - Multilingual greetings (Tamil script + Tanglish)
    - Multilingual continuation requests
    - Multilingual conversation-context flow (refinement + continuation)
    - Same-language instructions in the grounded prompts and both
      Gemini system prompts

No Gemini, no network, no API keys.
"""

import time
import unicodedata

import ai.chat_service as chat_service_module
from ai.chat_service import (
    _is_continuation_request,
    handle_food_message,
)
from ai.conversation_manager import create_food_chat
from ai.greetings import detect_greeting, greeting_response
from ai.session_manager import SessionManager
from food_recommendation.food_search import (
    NONVEG_KEYWORDS,
    STOPWORDS,
    smart_food_search,
)
from food_recommendation.language_phrases import normalize_message
from food_recommendation.preference_classifier import classify_preferences


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
    from food_recommendation.recommendation_engine import FOOD_CATEGORIES

    return {
        food
        for items in FOOD_CATEGORIES.values()
        for food in items
    }


def _foods(result):
    return [item["food"] for item in result["results"]]


# ---------- Unicode normalization ----------

def test_normalize_basic_punctuation_and_case():
    assert normalize_message("Hello   World!") == "hello world"
    assert normalize_message("Show   MORE.") == "show more"
    assert normalize_message("") == ""
    assert normalize_message(None) == ""


def test_normalize_preserves_in_word_apostrophes():
    assert normalize_message("don't want spicy") == "don't want spicy"
    # Possessive/trailing and leading apostrophes become separators.
    assert normalize_message("dogs' bones") == "dogs bones"
    assert normalize_message("'tis") == "tis"


def test_normalize_curly_apostrophe_becomes_straight():
    assert normalize_message("don\u2019t want spicy") == "don't want spicy"


def test_normalize_removes_zero_width_characters():
    text = "enak\u200bku birya\u200cni"
    normalized = normalize_message(text)

    assert "\u200b" not in normalized
    assert "\u200c" not in normalized
    assert normalized == "enakku biryani"


def test_normalize_underscore_is_separator():
    assert normalize_message("sweet_corn soup") == "sweet corn soup"


def test_normalize_tamil_nfd_input_matches_nfc():
    nfc = "\u0b8e\u0ba9\u0b95\u0bcd\u0b95\u0bc1 \u0b9a\u0bc8\u0bb5 \u0b89\u0ba3\u0bb5\u0bc1 \u0bb5\u0bc7\u0ba3\u0bc1\u0bae\u0bcd"

    assert normalize_message(
        unicodedata.normalize("NFD", nfc)
    ) == normalize_message(nfc)


# ---------- preference classification ----------

def test_tamil_script_vegetarian():
    prefs = classify_preferences(
        "\u0b8e\u0ba9\u0b95\u0bcd\u0b95\u0bc1 \u0b9a\u0bc8\u0bb5 "
        "\u0b89\u0ba3\u0bb5\u0bc1 \u0bb5\u0bc7\u0ba3\u0bc1\u0bae\u0bcd"
    )

    assert prefs["food_type"] == "vegetarian"


def test_tanglish_nonvegetarian_asaivam():
    prefs = classify_preferences("enakku asaivam sapadu venum")

    assert prefs["food_type"] == "non-vegetarian"


def test_tanglish_kari_is_nonvegetarian():
    prefs = classify_preferences("kari sapadu venum")

    assert prefs["food_type"] == "non-vegetarian"


def test_tanglish_spicy_biriyani():
    prefs = classify_preferences("enakku karamaana biriyani venum")

    assert prefs["taste"] == "spicy"
    assert prefs["category"] == "Biryani"
    assert prefs["food_type"] is None  # biryani alone is not non-veg


def test_tanglish_mild_karam_illa():
    assert classify_preferences("karam illa")["taste"] == "mild"
    assert classify_preferences("konjam karam")["taste"] == "mild"


def test_tamil_script_mild():
    prefs = classify_preferences(
        "\u0b95\u0bbe\u0bb0\u0bae\u0bcd \u0b87\u0bb2\u0bcd\u0bb2\u0bc8"
    )

    assert prefs["taste"] == "mild"


def test_curly_apostrophe_negation_is_mild():
    prefs = classify_preferences("I don\u2019t want spicy food")

    assert prefs["taste"] == "mild"


def test_sweet_taste_and_desserts_category():
    prefs = classify_preferences("enakku inippu venum")

    assert prefs["taste"] == "sweet"
    assert prefs["category"] == "Desserts"


def test_english_sweet_maps_to_desserts():
    prefs = classify_preferences("I want something sweet")

    assert prefs["taste"] == "sweet"
    assert prefs["category"] == "Desserts"


def test_healthy_and_budget_tanglish():
    assert classify_preferences("aarogyamaana unavu")["health"] == "healthy"
    assert classify_preferences("kammi vilai sapadu")["budget"] == "budget-friendly"
    assert classify_preferences("malivaana unavu")["budget"] == "budget-friendly"


# ---------- food search ----------

def test_tanglish_biriyani_spelling_canonicalization():
    result = smart_food_search("enakku biriyani venum")

    foods = _foods(result)
    assert len(foods) > 0
    for food in foods:
        assert "biryani" in food.lower()


def test_tanglish_koli_biriyani_finds_chicken_biryani():
    result = smart_food_search("enakku koli biriyani venum")

    foods = _foods(result)
    assert "Chicken Biryani" in foods
    assert result["preferences"]["food_type"] == "non-vegetarian"


def test_tamil_script_food_words_canonicalize():
    result = smart_food_search(
        "\u0b8e\u0ba9\u0b95\u0bcd\u0b95\u0bc1 \u0b95\u0bcb\u0bb4\u0bbf "
        "\u0baa\u0bbf\u0bb0\u0bbf\u0baf\u0bbe\u0ba3\u0bbf \u0bb5\u0bc7\u0ba3\u0bc1\u0bae\u0bcd"
    )

    foods = _foods(result)
    assert "Chicken Biryani" in foods
    assert result["preferences"]["food_type"] == "non-vegetarian"


def test_punctuation_attached_tanglish_still_matches():
    result = smart_food_search(
        "\u0ba4\u0bcb\u0b9a\u0bc8!"  # "dosai!" in Tamil script
    )

    foods = _foods(result)
    assert len(foods) > 0
    for food in foods:
        assert "dosa" in food.lower()


def test_vegetarian_tamil_script_broad_request():
    result = smart_food_search(
        "\u0b8e\u0ba9\u0b95\u0bcd\u0b95\u0bc1 \u0b9a\u0bc8\u0bb5 "
        "\u0b89\u0ba3\u0bb5\u0bc1 \u0bb5\u0bc7\u0ba3\u0bc1\u0bae\u0bcd"
    )

    foods = _foods(result)
    assert len(foods) > 0
    assert result["preferences"]["food_type"] == "vegetarian"

    # Name-based veg filter (English catalogue) still holds.
    for food in foods:
        assert not any(k in food.lower() for k in NONVEG_KEYWORDS)


def test_negated_food_word_is_excluded():
    result = smart_food_search("enakku biryani venam, dosa venum")

    foods = _foods(result)
    assert len(foods) > 0
    for food in foods:
        assert "dosa" in food.lower()
        assert "biryani" not in food.lower()


def test_negation_only_request_returns_no_results():
    assert smart_food_search("biryani venam")["results"] == []


def test_negation_words_are_not_stopwords():
    # Negation words must stay meaningful, never silently swallowed.
    assert "venam" not in STOPWORDS
    assert "illa" not in STOPWORDS


def test_tamil_script_junk_returns_no_results():
    result = smart_food_search(
        "\u0b9c\u0bcd\u0b9c\u0b95\u0bcd\u0b9c\u0b9c\u0bcd qqq"
    )

    assert result["results"] == []


def test_tanglish_junk_returns_no_results():
    assert smart_food_search("asdasd zzzqqq")["results"] == []


def test_tanglish_broad_request_general_fallback():
    result = smart_food_search("enakku sapadu venum")

    assert len(_foods(result)) > 0


def test_mixed_language_message_finds_food():
    result = smart_food_search("Vanakkam, enakku biryani venum")

    foods = _foods(result)
    assert "Chicken Biryani" in foods


def test_sweet_search_returns_desserts():
    result = smart_food_search("I want something sweet")

    assert len(_foods(result)) > 0
    for item in result["results"]:
        assert item["category"] == "Desserts"


# ---------- greetings ----------

def test_detect_tamil_script_greeting():
    assert detect_greeting(
        "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd"
    )
    assert detect_greeting("Vanakam")
    assert detect_greeting("hi!")


def test_mixed_greeting_message_is_not_a_greeting():
    assert not detect_greeting("Vanakkam, enakku biryani venum")


def test_tamil_script_greeting_response():
    reply = greeting_response(
        "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd"
    )

    assert reply.startswith(
        "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd!"
    )


def test_tanglish_greeting_response():
    reply = greeting_response("vanakkam")

    assert reply.startswith("Vanakkam! OXRO-kku")


def test_english_greeting_response_unchanged():
    assert greeting_response("good morning") == (
        "Good morning! Welcome to OXRO. How can I help you today?"
    )
    assert greeting_response("hello") == (
        "Hello! Welcome to OXRO Food Delivery Support. "
        "How may I help you today?"
    )


# ---------- continuation requests ----------

def test_tanglish_continuation_requests():
    assert _is_continuation_request("Innum kaatunga")
    assert _is_continuation_request("Innum konjam options")
    assert _is_continuation_request("Vera edavadhu")
    assert _is_continuation_request("Innoru option")


def test_tanglish_food_requests_are_not_continuations():
    assert not _is_continuation_request("Innum biryani venum")
    assert not _is_continuation_request("enakku karam illa dosa venum")


# ---------- conversation context (fake sessions, no Gemini) ----------

def test_tanglish_refinement_uses_previous_context():
    sm = make_manager()
    handle_food_message("C1", "enakku koli biriyani venum", sm)

    result = handle_food_message("C1", "karama maathu", sm)

    assert result["status"] == "success"
    # Refinement: previously matched items, no new search.
    assert result["matched_foods"] == ["Chicken Biryani"]

    grounded = last_grounded_message(sm, "C1")
    assert "CUSTOMER FOLLOW-UP MESSAGE" in grounded
    assert "Chicken Biryani" in grounded
    assert '"taste": "spicy"' in grounded


def test_tanglish_continuation_returns_alternatives_without_repeats():
    sm = make_manager()
    handle_food_message("C1", "enakku koli biriyani venum", sm)

    result = handle_food_message("C1", "innum kaatunga", sm)

    assert result["status"] == "success"

    foods = result["matched_foods"]
    assert len(foods) > 0
    assert "Chicken Biryani" not in foods

    catalogue = all_catalogue_foods()
    for food in foods:
        assert food in catalogue


def test_mixed_language_turn_stores_context():
    sm = make_manager()

    result = handle_food_message(
        "C1", "Vanakkam, enakku biryani venum", sm
    )

    assert result["status"] == "success"
    assert "Chicken Biryani" in result["matched_foods"]

    context = sm.get_context("C1")
    assert context["preferences"]["category"] == "Biryani"
    assert "Chicken Biryani" in context["matched_foods"]


def test_session_isolation_with_multilingual_customers():
    sm = make_manager()

    handle_food_message("C_TA", "enakku koli biriyani venum", sm)
    handle_food_message("C_EN", "I want veg pizza", sm)

    context_ta = sm.get_context("C_TA")
    context_en = sm.get_context("C_EN")

    assert context_ta["preferences"]["food_type"] == "non-vegetarian"
    assert context_en["preferences"]["food_type"] == "vegetarian"
    assert "Chicken Biryani" in context_ta["matched_foods"]
    assert "Chicken Biryani" not in context_en["matched_foods"]


# ---------- same-language instructions ----------

def test_grounded_prompts_carry_language_instruction():
    sm = make_manager()
    handle_food_message("C1", "enakku koli biriyani venum", sm)
    grounded = last_grounded_message(sm, "C1")
    assert "Reply in the same language" in grounded

    handle_food_message("C1", "karama maathu", sm)
    followup = last_grounded_message(sm, "C1")
    assert "Reply in the same language" in followup


def test_conversation_system_prompt_carry_language_instruction():
    captured = {}

    class CapturingChats:
        def create(self, model=None, config=None):
            captured["config"] = config
            return FakeChat()

    class CapturingClient:
        chats = CapturingChats()

    create_food_chat(CapturingClient())

    assert "same language" in captured["config"]["system_instruction"]


def test_multilingual_context_survives_short_ttl_expiry():
    sm = make_manager(session_ttl_seconds=1)
    handle_food_message("C1", "enakku koli biriyani venum", sm)

    time.sleep(1.2)

    # Expired session: refinement behaves as a fresh unmatched request.
    result = handle_food_message("C1", "karama maathu", sm)

    assert result["matched_foods"] == []
    assert sm.get_context("C1") == {"preferences": {}, "matched_foods": []}
