"""
Phase 6 Step 4 tests: minimal in-process rate limiting.

Covers:
    - SlidingWindowRateLimiter unit behaviour (admission, rejection,
      sliding expiry via a fake clock, key independence, cleanup,
      Retry-After math, thread safety, env fallbacks).
    - AuthFailureThrottleMiddleware unit behaviour (401 counting,
      503 auth_not_configured NOT counted, short-circuit blocking,
      unrelated paths and probes untouched, expiry).
    - API level: per-customer limits on /chat, /recommend-food and
      /session/{customer_id}/end; greetings counted; other customers
      unaffected; valid-key requests blocked during an IP block;
      /health and /ready reachable during a block; existing 401/422/
      413/503/quota-exhaustion contracts unchanged; Retry-After on
      429s.

No sleeps: the limiter's injectable clock is advanced virtually.
All Gemini interaction is faked - no network, no API keys.
"""

import asyncio
import json
import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

import main as main_module
from ai.conversation_manager import GeminiQuotaError
from ai.session_manager import SessionManager
from main import AuthFailureThrottleMiddleware
from rate_limiter import (
    DEFAULT_RATE_LIMIT_AUTH_FAILURES,
    DEFAULT_RATE_LIMIT_CUSTOMER_REQUESTS,
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    SlidingWindowRateLimiter,
    load_auth_failures_from_env,
    load_customer_requests_from_env,
    load_window_seconds_from_env,
)


API_KEY = "test-sandbox-key"  # matches conftest default
H = {"X-API-Key": API_KEY}

RATE_LIMITED_BODY = {
    "detail": {
        "status": "rate_limited",
        "reason": "Too many requests. Please try again later."
    }
}


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeChat:
    def __init__(self, client=None):
        self.history = []

    def send_message(self, message):
        self.history.append(message)
        return FakeResponse(f"AI reply to: {message}")


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_limiter(limit=2, window=60, clock=None):
    return SlidingWindowRateLimiter(
        limit=limit,
        window_seconds=window,
        clock=clock or time.monotonic,
    )


# ---------------------------------------------------------------------------
# SlidingWindowRateLimiter unit tests
# ---------------------------------------------------------------------------

def test_requests_below_limit_are_allowed():
    limiter = make_limiter(limit=3)

    assert limiter.hit("C1") == (True, 0)
    assert limiter.hit("C1") == (True, 0)
    assert limiter.hit("C1") == (True, 0)


def test_requests_exceeding_limit_are_rejected():
    limiter = make_limiter(limit=2)

    limiter.hit("C1")
    limiter.hit("C1")

    allowed, retry_after = limiter.hit("C1")

    assert allowed is False
    assert retry_after >= 1


def test_sliding_window_expiry_with_fake_clock():
    clock = FakeClock()
    limiter = make_limiter(limit=2, window=60, clock=clock)

    limiter.hit("C1")  # t=1000
    limiter.hit("C1")  # t=1000

    clock.advance(59)
    assert limiter.hit("C1")[0] is False  # still inside the window

    clock.advance(1.5)  # t=1060.5: both timestamps have left the window
    allowed, _ = limiter.hit("C1")
    assert allowed is True


def test_window_slides_partially():
    clock = FakeClock()
    limiter = make_limiter(limit=2, window=60, clock=clock)

    limiter.hit("C1")       # t=1000
    clock.advance(30)
    limiter.hit("C1")       # t=1030

    clock.advance(30.5)     # t=1060.5: only the first left the window
    allowed, _ = limiter.hit("C1")
    assert allowed is True


def test_different_keys_have_independent_limits():
    limiter = make_limiter(limit=1)

    limiter.hit("C1")

    assert limiter.hit("C1")[0] is False
    assert limiter.hit("C2")[0] is True


def test_expired_entries_and_empty_keys_are_cleaned_up():
    clock = FakeClock()
    limiter = make_limiter(limit=1, window=60, clock=clock)

    limiter.hit("C1")

    assert "C1" in limiter._events

    clock.advance(61)
    assert limiter.is_blocked("C1") == (False, 0)

    # The prune removed the now-empty key entirely.
    assert "C1" not in limiter._events


def test_retry_after_calculation_is_correct():
    clock = FakeClock()
    limiter = make_limiter(limit=2, window=60, clock=clock)

    limiter.hit("C1")           # t=1000
    clock.advance(0.5)
    limiter.hit("C1")           # t=1000.5

    allowed, retry_after = limiter.hit("C1")  # oldest is 0.5 s old
    assert allowed is False
    assert retry_after == 60    # ceil(60 - 0.5)

    clock.advance(59)           # t=1059.5: oldest is 59.5 s old
    allowed, retry_after = limiter.hit("C1")
    assert allowed is False
    assert retry_after == 1     # ceil(0.5), minimum 1

    clock.advance(1)            # t=1060.5: window slid
    assert limiter.hit("C1")[0] is True


def test_rejected_events_are_not_recorded():
    clock = FakeClock()
    limiter = make_limiter(limit=1, window=60, clock=clock)

    limiter.hit("C1")

    for _ in range(5):
        limiter.hit("C1")       # all rejected, none recorded

    timestamps = limiter._events["C1"]
    assert len(timestamps) == 1


def test_concurrent_hits_admit_exactly_the_limit():
    limiter = make_limiter(limit=5, window=60)
    thread_count = 20
    barrier = threading.Barrier(thread_count)
    allowed_count = []
    allowed_lock = threading.Lock()

    def worker():
        barrier.wait()
        allowed, _ = limiter.hit("C_SHARED")
        with allowed_lock:
            allowed_count.append(allowed)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(allowed_count) == thread_count
    assert sum(1 for ok in allowed_count if ok) == 5


def test_env_defaults():
    assert DEFAULT_RATE_LIMIT_WINDOW_SECONDS == 60
    assert DEFAULT_RATE_LIMIT_CUSTOMER_REQUESTS == 20
    assert DEFAULT_RATE_LIMIT_AUTH_FAILURES == 10


@pytest.mark.parametrize("raw", ["abc", "0", "-5", " "])
def test_env_invalid_values_fall_back_to_defaults(monkeypatch, raw):
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", raw)
    monkeypatch.setenv("RATE_LIMIT_CUSTOMER_REQUESTS", raw)
    monkeypatch.setenv("RATE_LIMIT_AUTH_FAILURES", raw)

    assert load_window_seconds_from_env() == 60
    assert load_customer_requests_from_env() == 20
    assert load_auth_failures_from_env() == 10


def test_env_valid_overrides(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "90")
    monkeypatch.setenv("RATE_LIMIT_CUSTOMER_REQUESTS", "7")
    monkeypatch.setenv("RATE_LIMIT_AUTH_FAILURES", "3")

    assert load_window_seconds_from_env() == 90
    assert load_customer_requests_from_env() == 7
    assert load_auth_failures_from_env() == 3


def test_env_missing_values_fall_back_to_defaults(monkeypatch):
    for name in (
        "RATE_LIMIT_WINDOW_SECONDS",
        "RATE_LIMIT_CUSTOMER_REQUESTS",
        "RATE_LIMIT_AUTH_FAILURES",
    ):
        monkeypatch.delenv(name, raising=False)

    assert load_window_seconds_from_env() == 60
    assert load_customer_requests_from_env() == 20
    assert load_auth_failures_from_env() == 10


# ---------------------------------------------------------------------------
# AuthFailureThrottleMiddleware unit tests (middleware simulated)
# ---------------------------------------------------------------------------

def run_middleware(limiter, path, response_status, client_ip="203.0.113.9"):
    """
    Runs AuthFailureThrottleMiddleware against a stub inner app that
    replies with `response_status`. Returns (sent_messages,
    inner_was_called).
    """
    sent_messages = []
    calls = {"inner": 0}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent_messages.append(message)

    async def inner_app(scope, receive, send):
        calls["inner"] += 1
        await send({
            "type": "http.response.start",
            "status": response_status,
            "headers": [],
        })
        await send({
            "type": "http.response.body",
            "body": b"",
            "more_body": False,
        })

    app_stub = types.SimpleNamespace(
        state=types.SimpleNamespace(auth_failure_limiter=limiter)
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "client": (client_ip, 50000),
        "app": app_stub,
    }

    middleware = AuthFailureThrottleMiddleware(inner_app)
    asyncio.run(middleware(scope, receive, send))

    return sent_messages, calls["inner"]


def test_auth_failures_trigger_blocking():
    limiter = make_limiter(limit=2)

    run_middleware(limiter, "/chat", 401)
    assert limiter.is_blocked("203.0.113.9") == (False, 0)

    run_middleware(limiter, "/chat", 401)
    blocked, retry_after = limiter.is_blocked("203.0.113.9")
    assert blocked is True
    assert retry_after >= 1


def test_auth_failure_block_expires_after_window():
    clock = FakeClock()
    limiter = make_limiter(limit=1, window=60, clock=clock)

    limiter.register("1.2.3.4")
    assert limiter.is_blocked("1.2.3.4")[0] is True

    clock.advance(61)
    assert limiter.is_blocked("1.2.3.4") == (False, 0)


def test_middleware_does_not_count_503_auth_not_configured():
    limiter = make_limiter(limit=1)

    for _ in range(5):
        run_middleware(limiter, "/chat", 503)

    assert limiter.is_blocked("203.0.113.9") == (False, 0)


def test_middleware_blocked_ip_is_short_circuited():
    limiter = make_limiter(limit=1)
    limiter.register("203.0.113.9")

    sent_messages, inner_calls = run_middleware(limiter, "/chat", 200)

    # The inner app (and therefore auth and body handling) never ran.
    assert inner_calls == 0
    assert sent_messages[0]["status"] == 429

    retry_headers = [
        value.decode()
        for name, value in sent_messages[0]["headers"]
        if name == b"retry-after"
    ]
    assert retry_headers and int(retry_headers[0]) >= 1

    body = b"".join(
        m.get("body", b"") for m in sent_messages
        if m["type"] == "http.response.body"
    )
    assert json.loads(body)["detail"]["status"] == "rate_limited"


def test_middleware_does_not_count_or_block_unrelated_paths():
    limiter = make_limiter(limit=1)
    limiter.register("203.0.113.9")  # IP would be blocked on /chat

    sent_messages, inner_calls = run_middleware(limiter, "/other", 401)

    # Unrelated path: served normally, and the 401 is not counted.
    assert inner_calls == 1
    assert sent_messages[0]["status"] == 401
    assert limiter.is_blocked("203.0.113.9")[0] is True  # unchanged


def test_middleware_allows_health_and_ready_during_ip_block():
    limiter = make_limiter(limit=1)
    limiter.register("203.0.113.9")  # blocked

    for probe_path in ("/health", "/ready"):
        sent_messages, inner_calls = run_middleware(limiter, probe_path, 200)
        assert inner_calls == 1
        assert sent_messages[0]["status"] == 200


def test_middleware_counts_401_on_session_end_path():
    limiter = make_limiter(limit=2)

    run_middleware(limiter, "/session/C1/end", 401)
    run_middleware(limiter, "/session/C1/end", 401)

    assert limiter.is_blocked("203.0.113.9")[0] is True


# ---------------------------------------------------------------------------
# API level tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """
    Real lifespan with default limiters; SessionManager swapped for an
    offline fake.
    """
    with TestClient(main_module.app) as test_client:
        main_module.app.state.session_manager = SessionManager(
            None,
            chat_factory=lambda cl: FakeChat()
        )
        yield test_client


@pytest.fixture
def limited_client():
    """
    Real lifespan, then swap in a small per-customer limiter so the
    3rd request from one customer is rejected.
    """
    with TestClient(main_module.app) as test_client:
        main_module.app.state.session_manager = SessionManager(
            None,
            chat_factory=lambda cl: FakeChat()
        )
        main_module.app.state.rate_limiter = make_limiter(limit=2)
        yield test_client


@pytest.fixture
def auth_throttled_client():
    """
    Real lifespan, then swap in a small auth-failure limiter so the
    3rd failed authentication from one IP is blocked.
    """
    with TestClient(main_module.app) as test_client:
        main_module.app.state.session_manager = SessionManager(
            None,
            chat_factory=lambda cl: FakeChat()
        )
        main_module.app.state.auth_failure_limiter = make_limiter(limit=2)
        yield test_client


def _fill_customer_limit(client, customer_id, message):
    responses = [
        client.post(
            "/chat",
            headers=H,
            json={"customer_id": customer_id, "message": message}
        )
        for _ in range(2)
    ]
    for response in responses:
        assert response.status_code == 200
    return client.post(
        "/chat",
        headers=H,
        json={"customer_id": customer_id, "message": message}
    )


def test_customer_rate_limit_on_chat(limited_client):
    third = _fill_customer_limit(limited_client, "C1", "I want chicken biryani")

    assert third.status_code == 429
    assert third.json() == RATE_LIMITED_BODY
    assert int(third.headers["Retry-After"]) >= 1


def test_other_customers_remain_unaffected(limited_client):
    _fill_customer_limit(limited_client, "C1", "I want chicken biryani")

    response = limited_client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C2", "message": "I want biryani"}
    )

    assert response.status_code == 200


def test_greeting_requests_are_rate_limited(limited_client):
    first = limited_client.post(
        "/chat", headers=H, json={"customer_id": "C1", "message": "hello"}
    )
    assert first.status_code == 200
    assert first.json()["status"] == "greeting"

    limited_client.post(
        "/chat", headers=H, json={"customer_id": "C1", "message": "hello"}
    )

    third = limited_client.post(
        "/chat", headers=H, json={"customer_id": "C1", "message": "hello"}
    )

    assert third.status_code == 429
    assert third.json() == RATE_LIMITED_BODY


def test_recommend_food_is_rate_limited(limited_client, monkeypatch):
    monkeypatch.setattr(
        main_module,
        "recommend_food_with_ai",
        lambda message, client_arg, max_retries=3: {
            "status": "success",
            "preferences": {},
            "food_results": ["Chicken Biryani"],
            "ai_response": "Here are some options.",
        },
    )

    for _ in range(2):
        response = limited_client.post(
            "/recommend-food",
            headers=H,
            json={"customer_id": "C1", "message": "I want biryani"}
        )
        assert response.status_code == 200

    third = limited_client.post(
        "/recommend-food",
        headers=H,
        json={"customer_id": "C1", "message": "I want biryani"}
    )

    assert third.status_code == 429
    assert third.json() == RATE_LIMITED_BODY
    assert int(third.headers["Retry-After"]) >= 1


def test_session_end_is_rate_limited(limited_client):
    # The rate limit runs BEFORE session handling (404s are expected
    # for this unknown customer on the first two requests).
    first = limited_client.post("/session/C_UNKNOWN/end", headers=H)
    second = limited_client.post("/session/C_UNKNOWN/end", headers=H)
    third = limited_client.post("/session/C_UNKNOWN/end", headers=H)

    assert first.status_code == 404
    assert second.status_code == 404
    assert third.status_code == 429
    assert third.json() == RATE_LIMITED_BODY


def test_auth_failure_throttle_blocks_valid_key_requests(
    auth_throttled_client
):
    for _ in range(2):
        response = auth_throttled_client.post(
            "/chat",
            headers={"X-API-Key": "wrong-key"},
            json={"customer_id": "C1", "message": "hello"}
        )
        assert response.status_code == 401

    third_bad = auth_throttled_client.post(
        "/chat",
        headers={"X-API-Key": "wrong-key"},
        json={"customer_id": "C1", "message": "hello"}
    )
    assert third_bad.status_code == 429
    assert third_bad.json() == RATE_LIMITED_BODY

    # Even a VALID key cannot get through while the IP is blocked.
    valid = auth_throttled_client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "hello"}
    )
    assert valid.status_code == 429
    assert valid.json() == RATE_LIMITED_BODY
    assert int(valid.headers["Retry-After"]) >= 1


def test_health_and_ready_remain_accessible_during_ip_block(
    auth_throttled_client
):
    for _ in range(3):
        auth_throttled_client.post(
            "/chat",
            headers={"X-API-Key": "wrong-key"},
            json={"customer_id": "C1", "message": "hello"}
        )

    assert auth_throttled_client.get("/health").status_code == 200
    assert auth_throttled_client.get("/ready").status_code == 200


def test_existing_401_contract_unchanged(client):
    response = client.post(
        "/chat", json={"customer_id": "C1", "message": "hello"}
    )

    assert response.status_code == 401
    assert response.json()["detail"]["status"] == "unauthorized"


def test_existing_422_precedes_rate_limit(limited_client):
    # Invalid customer_id: validation rejects BEFORE the limiter is
    # consulted, so repeated invalid requests stay 422.
    for _ in range(4):
        response = limited_client.post(
            "/chat",
            headers=H,
            json={"customer_id": "bad id!", "message": "hello"}
        )
        assert response.status_code == 422


def test_existing_413_unaffected(client):
    response = client.post(
        "/chat",
        headers=H,
        content=b"x" * (32 * 1024 + 1)
    )

    assert response.status_code == 413
    assert response.json()["detail"]["status"] == "payload_too_large"


def test_existing_503_auth_not_configured_not_counted(
    client, monkeypatch
):
    with monkeypatch.context() as m:
        m.delenv("API_KEY", raising=False)

        for _ in range(3):
            response = client.post(
                "/chat",
                headers=H,
                json={"customer_id": "C1", "message": "hello"}
            )
            assert response.status_code == 503
            assert response.json()["detail"]["status"] == "auth_not_configured"

    # After the configuration is restored, the IP is NOT blocked.
    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "hello"}
    )
    assert response.status_code == 200


def test_existing_quota_exhaustion_response_unchanged(
    client, monkeypatch
):
    def raise_quota(customer_id, user_message, session_manager):
        raise GeminiQuotaError("quota")

    monkeypatch.setattr(main_module, "handle_food_message", raise_quota)

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "I want biryani"}
    )

    assert response.status_code == 429
    assert response.json() == {
        "detail": {
            "status": "quota_exhausted",
            "reason": "AI service quota exhausted. Please retry later."
        }
    }
