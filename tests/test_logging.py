"""
Phase 6 Step 1 tests: server-side error logging and minimal request
logging middleware.

Verifies:
    - Unexpected /chat and /recommend-food errors are logged
      server-side (traceback visible in logs, never in responses).
    - HTTP method, path, status code and duration are logged.
    - API keys, customer messages, replies and personal customer
      information NEVER appear in logs.
    - Logging works for successful and failed requests.
    - LOG_LEVEL environment handling (default INFO, valid levels,
      invalid fallback).

All Gemini interaction is faked - no network, no API keys.
"""

import logging
import re

import pytest
from fastapi.testclient import TestClient

import main as main_module
from ai.session_manager import SessionManager


API_KEY = "test-sandbox-key"  # matches conftest default
H = {"X-API-Key": API_KEY}


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


def _oxro_log_text(caplog):
    """
    Joined text of all oxro.api log records, including exception
    tracebacks attached via logger.exception (exc_info).
    """
    parts = []

    for record in caplog.records:
        if record.name != "oxro.api":
            continue

        parts.append(record.getMessage())

        if record.exc_text:
            parts.append(record.exc_text)
        elif record.exc_info:
            parts.append(
                logging.Formatter().formatException(record.exc_info)
            )

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

def test_request_method_path_status_and_duration_logged(caplog, client):
    caplog.set_level(logging.INFO, logger="oxro.api")

    response = client.get("/health")

    assert response.status_code == 200

    text = _oxro_log_text(caplog)
    assert "GET /health" in text
    assert "-> 200" in text
    assert re.search(r"\(\d+\.\d+ ms\)", text)


def test_request_logged_for_failed_request(caplog, client):
    caplog.set_level(logging.INFO, logger="oxro.api")

    response = client.post(
        "/chat",
        json={"customer_id": "C1", "message": "Hi"}
    )

    assert response.status_code == 401

    text = _oxro_log_text(caplog)
    assert "POST /chat -> 401" in text
    assert re.search(r"\(\d+\.\d+ ms\)", text)


def test_session_end_path_logged_without_customer_id(caplog, client):
    caplog.set_level(logging.INFO, logger="oxro.api")

    marker = "C_SECRETMARKER"
    response = client.post(f"/session/{marker}/end", headers=H)

    assert response.status_code == 404

    text = _oxro_log_text(caplog)
    # Route path TEMPLATE is logged; the actual customer id is not.
    assert "/session/{customer_id}/end" in text
    assert marker not in text


# ---------------------------------------------------------------------------
# Privacy: nothing sensitive in the logs
# ---------------------------------------------------------------------------

def test_customer_message_and_api_key_never_logged(caplog, client):
    caplog.set_level(logging.INFO, logger="oxro.api")

    secret_message = "SUPER-SECRET-MESSAGE-XYZ"
    wrong_key = "WRONG-API-KEY-XYZ"

    response = client.post(
        "/chat",
        headers={"X-API-Key": wrong_key},
        json={"customer_id": "C1", "message": secret_message}
    )

    assert response.status_code == 401

    text = _oxro_log_text(caplog)
    assert secret_message not in text
    assert wrong_key not in text


def test_successful_chat_logs_no_message_or_reply(caplog, client):
    caplog.set_level(logging.INFO, logger="oxro.api")

    marker_message = "SECRET-CUSTOMER-MESSAGE-QQQ"

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": marker_message}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"

    text = _oxro_log_text(caplog)
    assert marker_message not in text
    assert "AI reply to" not in text


def test_greeting_success_still_logged_without_message(caplog, client):
    caplog.set_level(logging.INFO, logger="oxro.api")

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C_GREET", "message": "hello"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "greeting"

    text = _oxro_log_text(caplog)
    assert "POST /chat -> 200" in text


# ---------------------------------------------------------------------------
# Server-side error logging
# ---------------------------------------------------------------------------

def test_unexpected_chat_error_logged_server_side(
    caplog, client, monkeypatch
):
    caplog.set_level(logging.ERROR, logger="oxro.api")

    def boom(customer_id, user_message, session_manager):
        raise RuntimeError("boom-unexpected-chat-test")

    monkeypatch.setattr(main_module, "handle_food_message", boom)

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "I want biryani"}
    )

    assert response.status_code == 502
    assert response.json()["detail"]["status"] == "service_error"
    # The client-facing reason stays generic.
    assert "boom" not in response.text

    error_records = [
        record for record in caplog.records
        if record.name == "oxro.api"
        and record.levelno == logging.ERROR
    ]
    assert error_records
    assert "boom-unexpected-chat-test" in _oxro_log_text(caplog)


def test_unexpected_recommend_food_error_logged_server_side(
    caplog, client, monkeypatch
):
    caplog.set_level(logging.ERROR, logger="oxro.api")

    def boom(user_message, client_arg, max_retries=3):
        raise RuntimeError("boom-unexpected-recommend-test")

    monkeypatch.setattr(main_module, "recommend_food_with_ai", boom)

    response = client.post(
        "/recommend-food",
        headers=H,
        json={"customer_id": "C1", "message": "I want biryani"}
    )

    assert response.status_code == 502
    assert response.json()["detail"]["status"] == "service_error"
    assert "boom" not in response.text

    error_records = [
        record for record in caplog.records
        if record.name == "oxro.api"
        and record.levelno == logging.ERROR
    ]
    assert error_records
    assert "boom-unexpected-recommend-test" in _oxro_log_text(caplog)


def test_unexpected_error_request_still_logged(caplog, client, monkeypatch):
    """
    The middleware must log the request even when the endpoint fails
    with an unexpected 502.
    """
    caplog.set_level(logging.INFO, logger="oxro.api")

    def boom(customer_id, user_message, session_manager):
        raise RuntimeError("boom-logging-flow-test")

    monkeypatch.setattr(main_module, "handle_food_message", boom)

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "I want biryani"}
    )

    assert response.status_code == 502

    text = _oxro_log_text(caplog)
    assert "POST /chat -> 502" in text


# ---------------------------------------------------------------------------
# LOG_LEVEL environment configuration
# ---------------------------------------------------------------------------

def test_log_level_defaults_to_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    assert main_module._load_log_level_from_env() == logging.INFO


def test_log_level_accepts_valid_levels(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert main_module._load_log_level_from_env() == logging.DEBUG

    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert main_module._load_log_level_from_env() == logging.WARNING

    monkeypatch.setenv("LOG_LEVEL", "  error  ")
    assert main_module._load_log_level_from_env() == logging.ERROR


def test_log_level_invalid_falls_back_to_info(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_LEVEL")

    assert main_module._load_log_level_from_env() == logging.INFO
