"""
Offer discovery tests (Phase 7 Step 5).

All backend interaction is mocked (monkeypatched list_coupons):
- no real HTTP requests
- no Gemini calls
- no Supabase / Render connections
"""

from __future__ import annotations

import pytest

import backend_client
from ai.offer_discovery import (
    EMPTY_REPLY,
    UNAVAILABLE_REPLY,
    is_offer_request,
    offer_reply,
)


# Shape produced by backend GET /coupons (couponController.list).
SAMPLE_COUPONS = {
    "coupons": [
        {
            "code": "OXRO500",
            "title": "Flat Rs.500 OFF",
            "sub": "On orders above Rs.2,999",
            "off": 500,
            "percent": False,
            "cap": 0,
            "minOrder": 2999,
            "freeShip": False,
        },
        {
            "code": "WELCOME10",
            "title": "10% OFF up to Rs.300",
            "sub": "On orders above Rs.999",
            "off": 10,
            "percent": True,
            "cap": 300,
            "minOrder": 999,
            "freeShip": False,
        },
        {
            "code": "FREESHIP",
            "title": "Free Delivery",
            "sub": "No shipping fee above Rs.499",
            "off": 0,
            "percent": False,
            "cap": 0,
            "minOrder": 499,
            "freeShip": True,
        },
    ]
}


def _ok(payload):
    return {
        "ok": True,
        "data": payload,
        "error": None,
        "message": None,
        "status_code": 200,
    }


def _fail(error, message, status_code=None):
    return {
        "ok": False,
        "data": None,
        "error": error,
        "message": message,
        "status_code": status_code,
    }


def _install_backend(monkeypatch, result):
    """Replace backend_client.list_coupons with a fixed result."""

    def fake_list():
        return result

    monkeypatch.setattr(backend_client, "list_coupons", fake_list)


# ---------------------------------------------------------------------------
# Intent detection - positives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Any offers?",
        "Show me coupon codes.",
        "Are there any discounts?",
        "What promotions are available?",
        "Show available coupons.",
        "any offers",
        "What offers do you have?",
        "List the coupons",
        "Any promo codes?",
        "Show me current deals",
        "Are there any active offers today?",
        "Give me vouchers",
        "Check available discounts",
        "Which coupons are listed?",
        "I want offers",
    ],
)
def test_offer_intent_detected(message):
    assert is_offer_request(message) is True


# ---------------------------------------------------------------------------
# Intent detection - negatives (other pipeline concerns)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # Restaurant discovery
        "Show me restaurants.",
        "Which restaurants are available?",
        "List available restaurants.",
        "I want to order food.",
        "restaurants",
        # Food / dish search
        "I want chicken biryani",
        "Suggest vegetarian food",
        "Show me paneer pizza",
        "Make it spicy",
        "Show more options",
        # Dish-selection confirmations
        "yes",
        "confirm",
        "select it",
        "go ahead",
        "that's right",
        # Dish price questions
        "What's the price?",
        "How much is biryani?",
        "price of chicken biryani",
        "show me the price",
        # Cancellation / change selection
        "cancel",
        "no",
        "change my selection",
        "never mind",
        "reject",
        # Other
        "hello",
        "good morning",
        "How are you?",
        "",
        None,
        123,
        # Singular verb "offer" must not steal service questions
        "Do you offer free delivery?",
        "Does Paradise offer delivery?",
        "What do restaurants offer?",
    ],
)
def test_non_offer_messages_not_detected(message):
    assert is_offer_request(message) is False


def test_greeting_still_wins_over_offers():
    assert is_offer_request("hello") is False
    assert is_offer_request("hi") is False


# ---------------------------------------------------------------------------
# Formatting - only real backend fields
# ---------------------------------------------------------------------------


def test_successful_listing_multiple_coupons(monkeypatch):
    _install_backend(monkeypatch, _ok(SAMPLE_COUPONS))

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_listed"
    assert result["matched_foods"] == []
    assert result["detected_preferences"] is None

    reply = result["reply"]
    assert "Here are the current offers:" in reply
    assert "OXRO500" in reply
    assert "WELCOME10" in reply
    assert "FREESHIP" in reply
    assert "Flat Rs.500 OFF" in reply
    assert "10% OFF up to Rs.300" in reply
    assert "Free Delivery" in reply
    assert "On orders above Rs.2,999" in reply
    assert "Minimum order: Rs.2999" in reply
    assert "Minimum order: Rs.999" in reply
    assert "Free delivery" in reply
    assert reply.count("1.") == 1
    assert reply.count("2.") == 1
    assert reply.count("3.") == 1


def test_flat_coupon_shows_flat_off(monkeypatch):
    payload = {
        "coupons": [
            {
                "code": "OXRO500",
                "title": "Flat Rs.500 OFF",
                "sub": "On orders above Rs.2,999",
                "off": 500,
                "percent": False,
                "cap": 0,
                "minOrder": 2999,
                "freeShip": False,
            }
        ]
    }
    _install_backend(monkeypatch, _ok(payload))

    result = offer_reply("Show me coupon codes.")

    assert result["status"] == "offers_listed"
    reply = result["reply"]
    assert "OXRO500" in reply
    assert "Flat Rs.500 off" in reply or "Flat Rs.500 OFF" in reply
    assert "% off" not in reply


def test_percentage_coupon_shows_percent_and_cap(monkeypatch):
    payload = {
        "coupons": [
            {
                "code": "WELCOME10",
                "title": "10% OFF up to Rs.300",
                "sub": "On orders above Rs.999",
                "off": 10,
                "percent": True,
                "cap": 300,
                "minOrder": 999,
                "freeShip": False,
            }
        ]
    }
    _install_backend(monkeypatch, _ok(payload))

    result = offer_reply("Are there any discounts?")

    assert result["status"] == "offers_listed"
    reply = result["reply"]
    assert "WELCOME10" in reply
    assert "10% off" in reply or "10% OFF" in reply
    assert "up to Rs.300" in reply
    assert "Flat Rs." not in reply


def test_free_delivery_coupon(monkeypatch):
    payload = {
        "coupons": [
            {
                "code": "FREESHIP",
                "title": "Free Delivery",
                "sub": "No shipping fee above Rs.499",
                "off": 0,
                "percent": False,
                "cap": 0,
                "minOrder": 499,
                "freeShip": True,
            }
        ]
    }
    _install_backend(monkeypatch, _ok(payload))

    result = offer_reply("Show available coupons.")

    assert result["status"] == "offers_listed"
    reply = result["reply"]
    assert "FREESHIP" in reply
    assert "Free delivery" in reply
    assert "Free Delivery" in reply  # title as sent by backend
    assert "Minimum order: Rs.499" in reply


def test_missing_optional_fields_only_show_present_ones(monkeypatch):
    # Minimal object: only code + off/percent; title/sub/minOrder/freeShip
    # absent - lines for those fields must be omitted, not defaulted.
    payload = {
        "coupons": [
            {
                "code": "MINIMAL1",
                "off": 50,
                "percent": False,
            }
        ]
    }
    _install_backend(monkeypatch, _ok(payload))

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_listed"
    reply = result["reply"]
    assert "MINIMAL1" in reply
    assert "Flat Rs.50 off" in reply
    assert "Minimum order:" not in reply
    assert "Free delivery" not in reply
    # No title/sub invented
    assert "Rs.0" not in reply
    assert "On orders" not in reply


def test_non_dict_entries_are_skipped(monkeypatch):
    payload = {
        "coupons": [
            "garbage",
            None,
            {
                "code": "KEEPME",
                "title": "Keep Me",
                "off": 100,
                "percent": False,
                "cap": 0,
                "minOrder": 500,
                "freeShip": False,
            },
        ]
    }
    _install_backend(monkeypatch, _ok(payload))

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_listed"
    assert "KEEPME" in result["reply"]
    assert "garbage" not in result["reply"]


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


def test_empty_coupon_list(monkeypatch):
    _install_backend(monkeypatch, _ok({"coupons": []}))

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_empty"
    assert result["reply"] == EMPTY_REPLY
    assert "OXRO500" not in result["reply"]
    assert result["matched_foods"] == []


# ---------------------------------------------------------------------------
# Backend failures -> polite unavailable message, never fake data
# ---------------------------------------------------------------------------


def test_backend_failure_returns_unavailable(monkeypatch):
    _install_backend(
        monkeypatch,
        _fail("http_error", "OXRO backend returned HTTP 500", status_code=500),
    )

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY
    assert "OXRO500" not in result["reply"]
    assert result["matched_foods"] == []


def test_connection_error_returns_unavailable(monkeypatch):
    _install_backend(
        monkeypatch,
        _fail("connection_error", "Could not connect to OXRO backend"),
    )

    result = offer_reply("Show me coupons")

    assert result["status"] == "offers_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY


def test_missing_base_url_returns_unavailable(monkeypatch):
    _install_backend(
        monkeypatch,
        _fail(
            "base_url_not_configured",
            "OXRO backend base URL is not configured",
        ),
    )

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY


def test_malformed_data_returns_unavailable(monkeypatch):
    # ok=True but data missing coupons list entirely.
    _install_backend(monkeypatch, _ok({"offers": []}))

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY


def test_coupons_wrong_type_returns_unavailable(monkeypatch):
    _install_backend(monkeypatch, _ok({"coupons": {"code": "X"}}))

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY


def test_result_not_a_dict_returns_unavailable(monkeypatch):
    _install_backend(monkeypatch, None)

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY


def test_unexpected_exception_returns_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("secret detail should not leak")

    monkeypatch.setattr(backend_client, "list_coupons", boom)

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY
    assert "secret" not in result["reply"]


def test_only_non_dict_coupons_returns_unavailable(monkeypatch):
    _install_backend(monkeypatch, _ok({"coupons": ["a", "b", None]}))

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_unavailable"
    assert result["reply"] == UNAVAILABLE_REPLY


# ---------------------------------------------------------------------------
# Never invent expiry / eligibility / conditions
# ---------------------------------------------------------------------------


def test_never_invents_expiry_or_eligibility(monkeypatch):
    _install_backend(monkeypatch, _ok(SAMPLE_COUPONS))

    result = offer_reply("What promotions are available?")
    reply_lower = result["reply"].lower()

    assert result["status"] == "offers_listed"

    for banned in (
        "expires",
        "expiry",
        "valid until",
        "validity",
        "expire on",
        "eligibility",
        "eligible",
        "only for new",
        "only for first",
        "one time",
        "once per",
        "per customer",
        "stack",
        "cannot be combined",
        "terms and conditions",
        "t&c",
    ):
        assert banned not in reply_lower

    # Backend payload had no expiry/eligibility fields at all.
    assert "expiry" not in str(SAMPLE_COUPONS).lower()


def test_formatting_does_not_add_fields_absent_from_payload(monkeypatch):
    # Coupon with only code: reply must not invent title/sub/discount.
    _install_backend(
        monkeypatch,
        _ok({"coupons": [{"code": "CODEONLY"}]}),
    )

    result = offer_reply("Any offers?")

    assert result["status"] == "offers_listed"
    reply = result["reply"]
    assert "CODEONLY" in reply
    assert "Flat Rs." not in reply
    assert "% off" not in reply
    assert "Minimum order:" not in reply
    assert "Free delivery" not in reply
    assert "expires" not in reply.lower()
