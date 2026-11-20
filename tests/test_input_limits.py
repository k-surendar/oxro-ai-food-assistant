"""
Phase 6 Step 2 tests: customer_id validation and 32 KB request body
size limit.

Covers:
    - Valid customer IDs (charset, min length, max length 128).
    - Invalid customer IDs: whitespace, newlines, control characters,
      unsupported characters, empty and 129+ characters (422 format).
    - Path parameter /session/{customer_id}/end uses the same charset
      rules (422) while normal unknown IDs still return 404.
    - Body limit: just below, exactly at, and just above 32 KB; missing
      Content-Length (chunked stream); misleading Content-Length.
    - Normal JSON handling preserved for all in-limit requests.

All Gemini interaction is faked - no network, no API keys.
"""

import json

import pytest
from fastapi.testclient import TestClient

import main as main_module
from ai.session_manager import SessionManager


API_KEY = "test-sandbox-key"  # matches conftest default
H = {"X-API-Key": API_KEY}

MAX_BODY_BYTES = 32 * 1024


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeChat:
    def __init__(self, client=None):
        self.history = []

    def send_message(self, message):
        self.history.append(message)
        return FakeResponse(f"AI reply to: {message}")


@pytest.fixture
def client():
    """
    Runs the real lifespan (dummy keys from conftest), then swaps the
    SessionManager for an offline fake.
    """
    with TestClient(main_module.app) as test_client:
        main_module.app.state.session_manager = SessionManager(
            None,
            chat_factory=lambda cl: FakeChat()
        )
        yield test_client


def _customer_ids(field_loc, response):
    """Customer-id validation errors from a 422 response."""
    return [
        err for err in response.json()["detail"]
        if tuple(err["loc"]) == ("body", field_loc)
    ]


# ---------------------------------------------------------------------------
# Customer ID validation (body)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("customer_id", [
    "C1",
    "customer_42",
    "OXRO:app:user-7",
    "a.b_c:d-e",
    "A",                      # minimum length 1
    "A" * 128,                # maximum length 128
])
def test_valid_customer_ids_accepted(client, customer_id):
    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": customer_id, "message": "hello"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "greeting"


@pytest.mark.parametrize("customer_id", [
    "",          # empty
    " ",         # whitespace only
    "C 1",       # internal space
    " user",     # leading space
    "user\t1",   # tab
    "user\n1",   # newline
    "user\r1",   # carriage return
    "user\x001", # control character
    "customer!", # unsupported punctuation
    "user@example.com",  # @ is not allowed
    "C#1",
    "C/1",       # path-like injection
    "customer id üñî",   # non-ASCII letters
    "🙂",        # emoji
    "A" * 129,   # too long
])
def test_invalid_customer_ids_rejected_with_422(client, customer_id):
    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": customer_id, "message": "hello"}
    )

    assert response.status_code == 422

    # Existing FastAPI validation error format is preserved.
    errors = _customer_ids("customer_id", response)
    assert errors, f"customer_id not flagged: {response.json()}"
    assert all(err["type"].startswith(("value_error", "string_", "missing"))
               for err in errors)


def test_customer_id_min_length_boundary(client):
    response = client.post(
        "/chat", headers=H, json={"customer_id": "C", "message": "hello"}
    )
    assert response.status_code == 200


def test_customer_id_max_length_boundary(client):
    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C" * 128, "message": "hello"}
    )
    assert response.status_code == 200

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C" * 129, "message": "hello"}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Customer ID validation (path parameter)
# ---------------------------------------------------------------------------

def test_session_end_path_customer_id_charset_rejected(client):
    response = client.post("/session/bad%20id/end", headers=H)

    assert response.status_code == 422


def test_session_end_path_customer_id_charset_allows_valid_unknown(
    client
):
    # Valid charset but unknown customer: normal 404 contract holds.
    response = client.post("/session/OXRO:unknown-1/end", headers=H)

    assert response.status_code == 404
    assert response.json()["detail"]["status"] == "session_not_found"


# ---------------------------------------------------------------------------
# Request body size limit (32 KB)
# ---------------------------------------------------------------------------

def test_body_just_under_limit_normal_json_ok(client):
    # ~2 KB real JSON body: far below the limit, fully handled.
    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "x" * 1999}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_body_just_above_limit_rejected_413(client):
    response = client.post(
        "/chat",
        headers=H,
        content=b"x" * (MAX_BODY_BYTES + 1)
    )

    assert response.status_code == 413
    assert response.json()["detail"]["status"] == "payload_too_large"
    assert "32 KB" in response.json()["detail"]["reason"]


def test_body_exactly_at_limit_allowed(client):
    # 32768 bytes exactly: allowed through the size limit (it then
    # fails JSON validation instead - never a 413).
    response = client.post(
        "/chat", headers=H, content=b"x" * MAX_BODY_BYTES
    )

    assert response.status_code != 413


def test_recommend_food_oversized_body_rejected_413(client):
    response = client.post(
        "/recommend-food",
        headers=H,
        content=b"x" * (MAX_BODY_BYTES + 500)
    )

    assert response.status_code == 413
    assert response.json()["detail"]["status"] == "payload_too_large"


def test_missing_content_length_small_body_ok(client):
    """
    Chunked stream (no Content-Length): a small valid JSON body must
    flow through the normal JSON handling path.
    """
    payload = json.dumps(
        {"customer_id": "C1", "message": "I want chicken biryani"}
    ).encode("utf-8")

    def body_stream():
        yield payload

    response = client.post(
        "/chat",
        headers={**H, "Content-Type": "application/json"},
        content=body_stream()
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_missing_content_length_oversized_body_rejected_413(client):
    def big_stream():
        yield b"x" * (MAX_BODY_BYTES + 100)

    response = client.post("/chat", headers=H, content=big_stream())

    assert response.status_code == 413
    assert response.json()["detail"]["status"] == "payload_too_large"


def test_missing_content_length_empty_body_rejected_422(client):
    def empty_stream():
        yield b""

    response = client.post("/chat", headers=H, content=empty_stream())

    # Passed the size check with no body: normal validation applies.
    assert response.status_code == 422


def test_misleading_content_length_oversized_body_rejected_413(client):
    """
    Content-Length lies (claims 1 byte, body is 40+ KB): the streaming
    cap must still reject it.
    """
    response = client.post(
        "/chat",
        headers={**H, "Content-Length": "1"},
        content=b"x" * (MAX_BODY_BYTES + 500)
    )

    assert response.status_code == 413
    assert response.json()["detail"]["status"] == "payload_too_large"


def test_misleading_content_length_non_numeric_falls_through(client):
    """
    Non-numeric Content-Length must not crash the size check; the
    request falls through to normal handling (invalid JSON -> 422).
    """
    response = client.post(
        "/chat",
        headers={**H, "Content-Length": "abc"},
        content=b"not json"
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Existing endpoints unaffected
# ---------------------------------------------------------------------------

def test_health_and_ready_unaffected(client):
    assert client.get("/health").status_code == 200
    # conftest sets a dummy GEMINI_API_KEY, so the client initializes
    # offline and /ready is 200 (degraded mode would be 503).
    assert client.get("/ready").status_code == 200


def test_auth_still_applies_before_normal_body_processing(client):
    # In-limit body but missing API key: 401 contract unchanged.
    response = client.post(
        "/chat",
        json={"customer_id": "C1", "message": "hello"}
    )

    assert response.status_code == 401
    assert response.json()["detail"]["status"] == "unauthorized"
