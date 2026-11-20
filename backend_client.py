"""
Read-only OXRO backend HTTP client (Phase 7 Step 1).

Purpose:
    Reusable, standalone client for a small set of PUBLIC, read-only
    OXRO backend endpoints (restaurants, menu, search). Independent of
    Gemini, the RAG pipeline, and the chatbot conversation flow.

Security:
    - Base URL comes ONLY from the OXRO_BACKEND_BASE_URL environment
      variable (never hardcoded; fail-closed when unset).
    - No authentication credentials, API keys, or Supabase access.
    - Logging records method/path/status/error codes only - never
      secrets, credentials, or full response bodies.

Operations (read-only):
    GET /restaurants
    GET /restaurants/{restaurant_id}
    GET /restaurants/{restaurant_id}/menu
    GET /search?q={query}
    GET /coupons

Response contract (all operations):
    success: {"ok": True,  "data": <json>, "error": None,
              "message": None, "status_code": <int>}
    failure: {"ok": False, "data": None, "error": <code>,
              "message": <text>, "status_code": <int|None>}
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger("oxro.backend_client")

# Connection vs. overall response timeouts (seconds).
CONNECT_TIMEOUT_SECONDS = 3.0
RESPONSE_TIMEOUT_SECONDS = 5.0

# Search query bounds (characters, after strip).
MIN_QUERY_LENGTH = 1
MAX_QUERY_LENGTH = 200

# Restaurant IDs: same charset as OXRO identifiers (e.g. r_paradise).
_RESTAURANT_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")


# ---------------------------------------------------------------------------
# Response helpers (consistent dictionary shape)
# ---------------------------------------------------------------------------

def _success(data: Any, status_code: int) -> dict[str, Any]:
    return {
        "ok": True,
        "data": data,
        "error": None,
        "message": None,
        "status_code": status_code,
    }


def _failure(
    error: str,
    message: str,
    status_code: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": error,
        "message": message,
        "status_code": status_code,
    }


# ---------------------------------------------------------------------------
# Configuration / validation
# ---------------------------------------------------------------------------

def _base_url() -> Optional[str]:
    raw = os.getenv("OXRO_BACKEND_BASE_URL")
    if raw is None:
        return None
    url = raw.strip()
    return url or None


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _validate_restaurant_id(restaurant_id: Any) -> Optional[str]:
    if not isinstance(restaurant_id, str):
        return None
    if _has_control_chars(restaurant_id):
        return None
    rid = restaurant_id.strip()
    if not _RESTAURANT_ID_RE.match(rid):
        return None
    return rid


def _validate_query(query: Any) -> Optional[str]:
    if not isinstance(query, str):
        return None
    if _has_control_chars(query):
        return None
    q = query.strip()
    if len(q) < MIN_QUERY_LENGTH or len(q) > MAX_QUERY_LENGTH:
        return None
    return q


def _http_client(base_url: str) -> httpx.Client:
    timeout = httpx.Timeout(
        RESPONSE_TIMEOUT_SECONDS,
        connect=CONNECT_TIMEOUT_SECONDS,
    )
    return httpx.Client(base_url=base_url, timeout=timeout)


# ---------------------------------------------------------------------------
# Core request path
# ---------------------------------------------------------------------------

def _request(
    method: str,
    path: str,
    *,
    params: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    base_url = _base_url()
    if base_url is None:
        logger.error("OXRO_BACKEND_BASE_URL is not configured")
        return _failure(
            "base_url_not_configured",
            "OXRO backend base URL is not configured",
        )

    try:
        with _http_client(base_url) as client:
            response = client.request(method, path, params=params)
    except httpx.TimeoutException:
        logger.warning("Backend timeout: %s %s", method, path)
        return _failure("timeout", "Request to OXRO backend timed out")
    except httpx.TransportError:
        logger.warning("Backend connection error: %s %s", method, path)
        return _failure(
            "connection_error",
            "Could not connect to OXRO backend",
        )
    except httpx.HTTPError:
        logger.warning("Backend HTTP client error: %s %s", method, path)
        return _failure(
            "connection_error",
            "Could not connect to OXRO backend",
        )
    except Exception:
        logger.exception("Unexpected backend client error: %s %s", method, path)
        return _failure("unexpected_error", "Unexpected backend client error")

    status = response.status_code
    logger.debug("%s %s -> %s", method, path, status)

    if status < 200 or status >= 300:
        logger.warning("Backend HTTP error: %s %s status=%s", method, path, status)
        return _failure(
            "http_error",
            f"OXRO backend returned HTTP {status}",
            status_code=status,
        )

    body = response.text
    if body is None or not body.strip():
        logger.warning("Backend empty response: %s %s", method, path)
        return _failure(
            "empty_response",
            "OXRO backend returned an empty response",
            status_code=status,
        )

    try:
        data = response.json()
    except ValueError:
        logger.warning("Backend invalid JSON: %s %s", method, path)
        return _failure(
            "invalid_json",
            "OXRO backend returned invalid JSON",
            status_code=status,
        )

    return _success(data, status)


# ---------------------------------------------------------------------------
# Public read-only operations
# ---------------------------------------------------------------------------

def list_restaurants() -> dict[str, Any]:
    """GET /restaurants - list all restaurants."""
    return _request("GET", "/restaurants")


def get_restaurant(restaurant_id: str) -> dict[str, Any]:
    """GET /restaurants/{restaurant_id} - restaurant details."""
    rid = _validate_restaurant_id(restaurant_id)
    if rid is None:
        logger.warning("Rejected invalid restaurant id")
        return _failure(
            "invalid_restaurant_id",
            "restaurant_id must be 1-128 characters of letters, digits, "
            "dot, underscore, colon or hyphen",
        )
    return _request("GET", f"/restaurants/{quote(rid, safe='')}")


def get_restaurant_menu(restaurant_id: str) -> dict[str, Any]:
    """GET /restaurants/{restaurant_id}/menu - menu for one restaurant."""
    rid = _validate_restaurant_id(restaurant_id)
    if rid is None:
        logger.warning("Rejected invalid restaurant id")
        return _failure(
            "invalid_restaurant_id",
            "restaurant_id must be 1-128 characters of letters, digits, "
            "dot, underscore, colon or hyphen",
        )
    return _request(
        "GET",
        f"/restaurants/{quote(rid, safe='')}/menu",
    )


def search_food(query: str) -> dict[str, Any]:
    """GET /search?q={query} - search dishes and restaurants."""
    q = _validate_query(query)
    if q is None:
        logger.warning("Rejected invalid search query")
        return _failure(
            "invalid_query",
            f"query must be a non-empty string of at most "
            f"{MAX_QUERY_LENGTH} characters",
        )
    return _request("GET", "/search", params={"q": q})


def list_coupons() -> dict[str, Any]:
    """
    GET /coupons - public coupon catalog (read-only).

    Successful payload shape from the backend:
        {"coupons": [
            {"code", "title", "sub", "off", "percent", "cap",
             "minOrder", "freeShip"},
            ...
        ]}

    There is no auth header and no customer identity. The response is
    accepted only when "coupons" is present and is a list (empty list
    is valid). Missing keys or wrong types map to invalid_response.
    """
    result = _request("GET", "/coupons")
    if not result["ok"]:
        return result

    data = result["data"]
    if not isinstance(data, dict) or not isinstance(data.get("coupons"), list):
        logger.warning("Backend unexpected coupons payload structure")
        return _failure(
            "invalid_response",
            "OXRO backend returned an unexpected coupons response",
            status_code=result["status_code"],
        )
    return result
