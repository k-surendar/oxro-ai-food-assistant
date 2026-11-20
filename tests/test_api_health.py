"""
API liveness / readiness tests (Phase 2 review point 4 + review-pass point 2).

- /health must be 200 whenever the API process is up, independent of
  Gemini state (liveness).
- /ready must be 503 until Gemini is initialized, 200 afterwards.
- The ACTUAL FastAPI lifespan is executed in these tests (TestClient
  context manager), including the missing-key case: the process must
  start degraded, not crash.
"""

from fastapi.testclient import TestClient

import main as main_module


def test_health_is_liveness_only_without_gemini():
    # No lifespan run (plain TestClient) -> app.state has no gemini_client,
    # yet /health must still answer 200 so platforms can detect liveness.
    client = TestClient(main_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_returns_503_before_startup():
    client = TestClient(main_module.app)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "not_ready"


def test_ready_returns_200_when_gemini_initialized(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-lifespan-test")

    with TestClient(main_module.app) as client:
        response = client.get("/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

        # Liveness also holds once started.
        assert client.get("/health").status_code == 200


# ---------- actual lifespan execution with a missing key ----------

def test_lifespan_with_missing_key_starts_degraded(monkeypatch):
    """
    The API must START even without GEMINI_API_KEY (degraded mode):
    /health stays 200 for platform liveness checks, /ready reports 503
    with the reason. This test executes the real lifespan.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with TestClient(main_module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        ready = client.get("/ready")
        assert ready.status_code == 503

        reason = ready.json()["detail"]["reason"]
        assert "GEMINI_API_KEY" in reason


def test_lifespan_with_sdk_failure_starts_degraded(monkeypatch):
    """
    Even if the Gemini SDK itself fails during startup, the process must
    start degraded (/health 200, /ready 503), never crash.
    Security: the raw SDK exception must NOT be exposed to clients -
    the client-facing reason is a static safe message.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")

    def broken_client(api_key):
        raise RuntimeError("simulated SDK failure LEAK_MARKER_41287")

    monkeypatch.setattr(main_module.genai, "Client", broken_client)

    with TestClient(main_module.app) as client:
        assert client.get("/health").status_code == 200

        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["detail"]["status"] == "not_ready"

        # Internal exception details are not leaked to clients.
        assert "LEAK_MARKER_41287" not in ready.text
        assert "Check the server logs" in ready.json()["detail"]["reason"]
