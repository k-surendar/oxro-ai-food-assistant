"""
CHATBOT_API_KEY authentication tests (Phase 7 backend-integration handover).

Covers:
    - 401 for missing / invalid X-API-Key on every protected endpoint
    - 200 for a valid CHATBOT_API_KEY
    - CHATBOT_API_KEY precedence over legacy API_KEY, and fallback to API_KEY
    - Fail-closed 503 auth_not_configured when neither variable is set
    - The secret value never appears in response bodies or logs
    - .env.example / local .env / .gitignore hygiene

All keys below are test-only sentinels (or the conftest sandbox value).
No real secrets, no network, no Gemini calls, no external services.
A fresh TestClient per test keeps the per-IP auth-failure throttle isolated.
"""

import logging
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main as main_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Test-only sentinel - never a real credential.
CHATBOT_KEY = "unit-test-chatbot-key-sentinel-DoNotLeak123"
# conftest.py default legacy value.
LEGACY_KEY = "test-sandbox-key"

GREETING_BODY = {"customer_id": "C1", "message": "Hi"}


@pytest.fixture
def client():
    """Fresh app + fresh rate limiters / throttle state per test."""
    with TestClient(main_module.app) as c:
        yield c


def _post_chat(client, key=None):
    headers = {"X-API-Key": key} if key is not None else {}
    return client.post("/chat", json=GREETING_BODY, headers=headers)


def test_missing_api_key_returns_401_on_all_protected_endpoints(client):
    endpoints = [
        ("/chat", GREETING_BODY),
        ("/recommend-food", GREETING_BODY),
        ("/session/C1/end", None),
    ]
    for path, body in endpoints:
        resp = client.post(path, json=body) if body is not None else client.post(path)
        assert resp.status_code == 401, path
        assert resp.json()["detail"]["status"] == "unauthorized"


def test_invalid_api_key_returns_401(client):
    resp = _post_chat(client, key="totally-wrong-key")
    assert resp.status_code == 401
    assert resp.json()["detail"]["status"] == "unauthorized"


def test_valid_chatbot_api_key_grants_access(client, monkeypatch):
    monkeypatch.setenv("CHATBOT_API_KEY", CHATBOT_KEY)
    resp = _post_chat(client, key=CHATBOT_KEY)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "greeting"
    assert payload["customer_id"] == "C1"


def test_chatbot_api_key_takes_precedence_over_legacy_api_key(
    client, monkeypatch
):
    monkeypatch.setenv("CHATBOT_API_KEY", CHATBOT_KEY)
    monkeypatch.setenv("API_KEY", LEGACY_KEY)

    ok = _post_chat(client, key=CHATBOT_KEY)
    assert ok.status_code == 200

    # Legacy value alone is no longer accepted while CHATBOT_API_KEY is set.
    stale = _post_chat(client, key=LEGACY_KEY)
    assert stale.status_code == 401


def test_falls_back_to_legacy_api_key(client, monkeypatch):
    monkeypatch.delenv("CHATBOT_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", LEGACY_KEY)

    ok = _post_chat(client, key=LEGACY_KEY)
    assert ok.status_code == 200

    wrong = _post_chat(client, key="wrong-key-value")
    assert wrong.status_code == 401


def test_fail_closed_when_neither_key_configured(client, monkeypatch):
    monkeypatch.delenv("CHATBOT_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    resp = _post_chat(client, key=CHATBOT_KEY)
    assert resp.status_code == 503
    assert resp.json()["detail"]["status"] == "auth_not_configured"


def test_valid_key_accepted_on_recommend_and_session_endpoints(
    client, monkeypatch
):
    monkeypatch.setenv("CHATBOT_API_KEY", CHATBOT_KEY)

    def fake_recommend(user_message, _client, max_retries=3):
        return {
            "status": "success",
            "preferences": {"taste": "spicy"},
            "food_results": ["Test Biryani"],
            "ai_response": "Try the test item.",
        }

    monkeypatch.setattr(main_module, "recommend_food_with_ai", fake_recommend)

    rec = client.post(
        "/recommend-food",
        json=GREETING_BODY,
        headers={"X-API-Key": CHATBOT_KEY},
    )
    assert rec.status_code == 200
    assert rec.json()["status"] == "success"

    # Unknown session is a 404 AFTER auth passed (never a 401).
    end = client.post(
        "/session/C1/end", headers={"X-API-Key": CHATBOT_KEY}
    )
    assert end.status_code == 404
    assert end.json()["detail"]["status"] == "session_not_found"


def test_invalid_key_still_rejected_on_recommend_and_session_end(
    client, monkeypatch
):
    monkeypatch.setenv("CHATBOT_API_KEY", CHATBOT_KEY)

    rec = client.post(
        "/recommend-food",
        json=GREETING_BODY,
        headers={"X-API-Key": "wrong-key-value"},
    )
    assert rec.status_code == 401

    end = client.post(
        "/session/C1/end", headers={"X-API-Key": "wrong-key-value"}
    )
    assert end.status_code == 401
    assert end.json()["detail"]["status"] == "unauthorized"


def test_secret_never_in_response_bodies(client, monkeypatch):
    monkeypatch.setenv("CHATBOT_API_KEY", CHATBOT_KEY)

    responses = [
        client.get("/health"),
        client.get("/ready"),
        _post_chat(client, key=CHATBOT_KEY),
        _post_chat(client, key="wrong-key-value"),
    ]
    for resp in responses:
        assert CHATBOT_KEY not in resp.text
        # No other secret-ish value either: legacy key must not leak.
        assert LEGACY_KEY not in resp.text


def test_secret_never_appears_in_logs(client, monkeypatch, caplog):
    monkeypatch.setenv("CHATBOT_API_KEY", CHATBOT_KEY)

    with caplog.at_level(logging.DEBUG):
        _post_chat(client, key=CHATBOT_KEY)  # 200
        _post_chat(client, key="wrong-key-value")  # 401

    assert CHATBOT_KEY not in caplog.text
    assert LEGACY_KEY not in caplog.text


def test_protected_endpoints_not_mounted_on_docs_when_disabled():
    assert os.getenv("DOCS_ENABLED", "false").lower() != "true"
    with TestClient(main_module.app) as client:
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/docs").status_code == 404


def test_env_example_contains_placeholders_only():
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for name in ("CHATBOT_API_KEY", "API_KEY"):
        line = next(
            l for l in example.splitlines() if l.startswith(f"{name}=")
        )
        value = line.split("=", 1)[1].strip()
        # Placeholder only: empty or an angle-bracket hint, never a value.
        assert value == "" or value.startswith("<"), f"{name} leaks a value"


def test_local_env_file_holds_key_and_gitignore_covers_secrets():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore.splitlines()
    assert "render_api_key.txt" in gitignore
    assert "apk for oxro app.txt" in gitignore

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        pytest.skip("no local .env (CI)")
    content = env_path.read_text(encoding="utf-8")
    key_lines = [
        l for l in content.splitlines() if l.startswith("CHATBOT_API_KEY=")
    ]
    assert key_lines, "local .env must define CHATBOT_API_KEY"
    value = key_lines[0].split("=", 1)[1].strip()
    assert len(value) >= 32, "key must be strong (>= 32 chars)"
    # Value must never be a placeholder.
    assert not value.startswith("<")
