"""
Unit tests for the read-only OXRO backend client (Phase 7 Step 1).

All tests are fully mocked via httpx.MockTransport:
- no real HTTP requests
- no Gemini calls
- no Supabase / Render connections
"""

from __future__ import annotations

import json

import httpx
import pytest

import backend_client

BASE_URL = "http://oxro-backend.test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_mock(monkeypatch, handler):
    """Point backend_client at BASE_URL with a MockTransport handler."""

    def _factory(base_url: str) -> httpx.Client:
        assert base_url == BASE_URL
        return httpx.Client(
            base_url=base_url,
            transport=httpx.MockTransport(handler),
            timeout=httpx.Timeout(backend_client.RESPONSE_TIMEOUT_SECONDS),
        )

    monkeypatch.setattr(backend_client, "_http_client", _factory)
    monkeypatch.setenv("OXRO_BACKEND_BASE_URL", BASE_URL)


def _json_response(payload, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )


def _assert_failure_shape(result, *, error: str, status_code=None):
    assert result["ok"] is False
    assert result["data"] is None
    assert result["error"] == error
    assert isinstance(result["message"], str) and result["message"]
    assert result["status_code"] == status_code
    assert set(result) == {"ok", "data", "error", "message", "status_code"}


def _assert_success_shape(result, *, status_code: int = 200):
    assert result["ok"] is True
    assert result["error"] is None
    assert result["message"] is None
    assert result["status_code"] == status_code
    assert result["data"] is not None
    assert set(result) == {"ok", "data", "error", "message", "status_code"}


# ---------------------------------------------------------------------------
# GET /restaurants
# ---------------------------------------------------------------------------

def test_list_restaurants_success(monkeypatch):
    payload = {"restaurants": [{"id": "r_paradise", "name": "Paradise"}]}
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["method"] = request.method
        return _json_response(payload)

    _install_mock(monkeypatch, handler)

    result = backend_client.list_restaurants()

    _assert_success_shape(result)
    assert result["data"] == payload
    assert seen["method"] == "GET"
    assert seen["path"] == "/restaurants"


def test_list_restaurants_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"error": "boom"}, status_code=500)

    _install_mock(monkeypatch, handler)

    result = backend_client.list_restaurants()

    _assert_failure_shape(result, error="http_error", status_code=500)


def test_list_restaurants_connection_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _install_mock(monkeypatch, handler)

    result = backend_client.list_restaurants()

    _assert_failure_shape(result, error="connection_error", status_code=None)


def test_list_restaurants_timeout(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    _install_mock(monkeypatch, handler)

    result = backend_client.list_restaurants()

    _assert_failure_shape(result, error="timeout", status_code=None)


def test_list_restaurants_invalid_json(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b"<html>not json</html>",
        )

    _install_mock(monkeypatch, handler)

    result = backend_client.list_restaurants()

    _assert_failure_shape(result, error="invalid_json", status_code=200)


def test_list_restaurants_empty_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, content=b"")

    _install_mock(monkeypatch, handler)

    result = backend_client.list_restaurants()

    _assert_failure_shape(result, error="empty_response", status_code=200)


def test_list_restaurants_missing_base_url(monkeypatch):
    monkeypatch.delenv("OXRO_BACKEND_BASE_URL", raising=False)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _json_response({})

    _install_mock(monkeypatch, handler)
    monkeypatch.delenv("OXRO_BACKEND_BASE_URL", raising=False)

    result = backend_client.list_restaurants()

    _assert_failure_shape(result, error="base_url_not_configured", status_code=None)
    assert calls == []


# ---------------------------------------------------------------------------
# GET /restaurants/{id}
# ---------------------------------------------------------------------------

def test_get_restaurant_success(monkeypatch):
    payload = {"restaurant": {"id": "r_paradise", "name": "Paradise"}}
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return _json_response(payload)

    _install_mock(monkeypatch, handler)

    result = backend_client.get_restaurant("r_paradise")

    _assert_success_shape(result)
    assert result["data"] == payload
    assert seen["path"] == "/restaurants/r_paradise"


def test_get_restaurant_not_found_is_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"error": "Restaurant not found"}, status_code=404)

    _install_mock(monkeypatch, handler)

    result = backend_client.get_restaurant("missing")

    _assert_failure_shape(result, error="http_error", status_code=404)


@pytest.mark.parametrize(
    "bad_id",
    ["", "   ", "r paradise", "a" * 129, "id/../etc", None, 123, "id\n"],
)
def test_get_restaurant_invalid_id(monkeypatch, bad_id):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _json_response({})

    _install_mock(monkeypatch, handler)

    result = backend_client.get_restaurant(bad_id)

    _assert_failure_shape(result, error="invalid_restaurant_id", status_code=None)
    assert calls == []


def test_get_restaurant_id_is_url_quoted(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return _json_response({"restaurant": {"id": "r:1"}})

    _install_mock(monkeypatch, handler)

    # Colon is allowed by validation; ensure it is passed through safely.
    result = backend_client.get_restaurant("r:1")

    _assert_success_shape(result)
    assert seen["path"] == "/restaurants/r:1"


# ---------------------------------------------------------------------------
# GET /restaurants/{id}/menu
# ---------------------------------------------------------------------------

def test_get_restaurant_menu_success(monkeypatch):
    payload = {"menu": [{"id": 1, "name": "Biryani", "price": 250}]}
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return _json_response(payload)

    _install_mock(monkeypatch, handler)

    result = backend_client.get_restaurant_menu("r_saravana")

    _assert_success_shape(result)
    assert result["data"] == payload
    assert seen["path"] == "/restaurants/r_saravana/menu"


def test_get_restaurant_menu_connection_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _install_mock(monkeypatch, handler)

    result = backend_client.get_restaurant_menu("r_saravana")

    _assert_failure_shape(result, error="connection_error", status_code=None)


@pytest.mark.parametrize("bad_id", ["", "bad id", None, 42])
def test_get_restaurant_menu_invalid_id(monkeypatch, bad_id):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _json_response({})

    _install_mock(monkeypatch, handler)

    result = backend_client.get_restaurant_menu(bad_id)

    _assert_failure_shape(result, error="invalid_restaurant_id", status_code=None)
    assert calls == []


# ---------------------------------------------------------------------------
# GET /search?q=
# ---------------------------------------------------------------------------

def test_search_food_success(monkeypatch):
    payload = {
        "query": "biryani",
        "results": [{"id": 1, "name": "Chicken Biryani"}],
        "restaurants": [{"id": "r_paradise"}],
    }
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["q"] = request.url.params.get("q")
        return _json_response(payload)

    _install_mock(monkeypatch, handler)

    result = backend_client.search_food("  biryani  ")

    _assert_success_shape(result)
    assert result["data"] == payload
    assert seen["path"] == "/search"
    assert seen["q"] == "biryani"


def test_search_food_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"error": "bad"}, status_code=400)

    _install_mock(monkeypatch, handler)

    result = backend_client.search_food("pizza")

    _assert_failure_shape(result, error="http_error", status_code=400)


def test_search_food_timeout(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    _install_mock(monkeypatch, handler)

    result = backend_client.search_food("pizza")

    _assert_failure_shape(result, error="timeout", status_code=None)


def test_search_food_invalid_json(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b"not-json{{",
        )

    _install_mock(monkeypatch, handler)

    result = backend_client.search_food("pizza")

    _assert_failure_shape(result, error="invalid_json", status_code=200)


@pytest.mark.parametrize(
    "bad_query",
    ["", "   ", "x" * 201, None, 99, "ok\n"],
)
def test_search_food_invalid_query(monkeypatch, bad_query):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _json_response({})

    _install_mock(monkeypatch, handler)

    result = backend_client.search_food(bad_query)

    _assert_failure_shape(result, error="invalid_query", status_code=None)
    assert calls == []


def test_search_food_max_length_query_accepted(monkeypatch):
    query = "a" * backend_client.MAX_QUERY_LENGTH

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"query": query, "results": [], "restaurants": []})

    _install_mock(monkeypatch, handler)

    result = backend_client.search_food(query)

    _assert_success_shape(result)


# ---------------------------------------------------------------------------
# GET /coupons
# ---------------------------------------------------------------------------

_COUPON_FIELDS = {
    "code",
    "title",
    "sub",
    "off",
    "percent",
    "cap",
    "minOrder",
    "freeShip",
}


def _sample_coupons_payload():
    return {
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
        ]
    }


def test_list_coupons_success(monkeypatch):
    payload = _sample_coupons_payload()
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["method"] = request.method
        seen["headers"] = dict(request.headers)
        return _json_response(payload)

    _install_mock(monkeypatch, handler)

    result = backend_client.list_coupons()

    _assert_success_shape(result)
    assert result["data"] == payload
    assert seen["method"] == "GET"
    assert seen["path"] == "/coupons"
    assert "authorization" not in seen["headers"]
    assert "x-api-key" not in seen["headers"]
    assert "cookie" not in seen["headers"]

    for coupon in result["data"]["coupons"]:
        assert _COUPON_FIELDS.issubset(coupon.keys())


def test_list_coupons_empty_list(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"coupons": []})

    _install_mock(monkeypatch, handler)

    result = backend_client.list_coupons()

    _assert_success_shape(result)
    assert result["data"] == {"coupons": []}


def test_list_coupons_missing_coupons_key(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"offers": []})

    _install_mock(monkeypatch, handler)

    result = backend_client.list_coupons()

    _assert_failure_shape(result, error="invalid_response", status_code=200)


def test_list_coupons_coupons_not_a_list(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"coupons": {"code": "OXRO500"}})

    _install_mock(monkeypatch, handler)

    result = backend_client.list_coupons()

    _assert_failure_shape(result, error="invalid_response", status_code=200)


def test_list_coupons_top_level_not_an_object(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(["OXRO500", "WELCOME10"])

    _install_mock(monkeypatch, handler)

    result = backend_client.list_coupons()

    _assert_failure_shape(result, error="invalid_response", status_code=200)


def test_list_coupons_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"error": "boom"}, status_code=500)

    _install_mock(monkeypatch, handler)

    result = backend_client.list_coupons()

    _assert_failure_shape(result, error="http_error", status_code=500)


def test_list_coupons_connection_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _install_mock(monkeypatch, handler)

    result = backend_client.list_coupons()

    _assert_failure_shape(result, error="connection_error", status_code=None)


def test_list_coupons_timeout(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    _install_mock(monkeypatch, handler)

    result = backend_client.list_coupons()

    _assert_failure_shape(result, error="timeout", status_code=None)


def test_list_coupons_invalid_json(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b"<html>not json</html>",
        )

    _install_mock(monkeypatch, handler)

    result = backend_client.list_coupons()

    _assert_failure_shape(result, error="invalid_json", status_code=200)


def test_list_coupons_empty_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, content=b"")

    _install_mock(monkeypatch, handler)

    result = backend_client.list_coupons()

    _assert_failure_shape(result, error="empty_response", status_code=200)


def test_list_coupons_missing_base_url(monkeypatch):
    monkeypatch.delenv("OXRO_BACKEND_BASE_URL", raising=False)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _json_response({"coupons": []})

    _install_mock(monkeypatch, handler)
    monkeypatch.delenv("OXRO_BACKEND_BASE_URL", raising=False)

    result = backend_client.list_coupons()

    _assert_failure_shape(
        result, error="base_url_not_configured", status_code=None
    )
    assert calls == []


# ---------------------------------------------------------------------------
# Contract / safety guarantees
# ---------------------------------------------------------------------------

def test_no_authorization_header_is_sent(monkeypatch):
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return _json_response({"restaurants": []})

    _install_mock(monkeypatch, handler)

    backend_client.list_restaurants()

    assert "authorization" not in seen_headers
    assert "x-api-key" not in seen_headers
    assert "cookie" not in seen_headers


def test_success_and_failure_share_the_same_keys(monkeypatch):
    def ok_handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"restaurants": []})

    _install_mock(monkeypatch, ok_handler)
    ok_result = backend_client.list_restaurants()

    def err_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=503, content=b"")

    _install_mock(monkeypatch, err_handler)
    err_result = backend_client.list_restaurants()

    assert set(ok_result) == set(err_result)
    assert set(ok_result) == {"ok", "data", "error", "message", "status_code"}


def test_connect_timeout_is_reported_as_timeout(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow connect")

    _install_mock(monkeypatch, handler)

    result = backend_client.list_restaurants()

    _assert_failure_shape(result, error="timeout", status_code=None)


def test_module_does_not_hardcode_production_url():
    source = open(backend_client.__file__, encoding="utf-8").read()
    assert "oxro-backend.onrender.com" not in source
    assert "render.com" not in source


def test_module_declares_no_credentials_usage():
    source = open(backend_client.__file__, encoding="utf-8").read()
    for forbidden in (
        "SUPABASE",
        "API_KEY",
        "GEMINI",
        "Authorization",
        "X-API-Key",
        "JWT",
    ):
        assert forbidden not in source
