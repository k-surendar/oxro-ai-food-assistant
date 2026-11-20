"""
Documentation endpoint control tests (Phase 6 Step 6).

DOCS_ENABLED is read ONCE at import time and applied during FastAPI
construction, so it cannot be toggled on an already-imported module.
To keep global application state untouched, every case runs the probe
in an ISOLATED SUBPROCESS:

    - the subprocess sets DOCS_ENABLED (or removes it) BEFORE
      importing main,
    - drives the real app through FastAPI's TestClient (offline: the
      dummy keys make the lifespan degrade gracefully without any
      Gemini call),
    - prints the observed status codes as JSON on one marker line,
      which the parent parses.

The parent pytest process never re-imports or reloads main, so other
test modules keep their single shared import.

Verified matrix (fail-closed):

    value                       /docs  /redoc  /openapi.json
    unset                       404    404     404
    "" (empty)                  404    404     404
    "false" / "1" / "yes" /     404    404     404
    any other string
    "true" / "TRUE" / "True" /  200    200     200
    surrounding whitespace

/health must stay available in every case.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PROBE_SOURCE = """
import json
import os
import sys

sys.path.insert(0, os.environ["DOCS_PROBE_PROJECT_ROOT"])

docs_value = os.environ["DOCS_PROBE_VALUE"]
if docs_value == "__unset__":
    os.environ.pop("DOCS_ENABLED", None)
else:
    os.environ["DOCS_ENABLED"] = docs_value

os.environ.setdefault("API_KEY", "test-sandbox-key")
os.environ.setdefault("GEMINI_API_KEY", "dummy-test-key")

from fastapi.testclient import TestClient
import main

with TestClient(main.app) as client:
    results = {}
    for path in ("/docs", "/redoc", "/openapi.json", "/health"):
        response = client.get(path)
        entry = {"status": response.status_code}
        if path == "/openapi.json" and response.status_code == 200:
            body = response.json()
            entry["is_openapi_schema"] = (
                "openapi" in body and "/health" in body.get("paths", {})
            )
        results[path] = entry

print("DOCS_PROBE_RESULT=" + json.dumps(results))
"""

_DISABLED_CASES = [
    pytest.param("__unset__", id="unset"),
    pytest.param("", id="empty"),
    pytest.param("false", id="false"),
    pytest.param("1", id="one"),
    pytest.param("yes", id="yes"),
    pytest.param("maybe", id="invalid_string"),
]

_ENABLED_CASES = [
    pytest.param("true", id="true"),
    pytest.param("TRUE", id="TRUE"),
    pytest.param("True", id="True"),
    pytest.param("  true  ", id="surrounding_whitespace"),
]


def _probe(docs_value):
    """Run one isolated subprocess probe and return the parsed results."""
    env = os.environ.copy()
    env["DOCS_PROBE_PROJECT_ROOT"] = str(_PROJECT_ROOT)
    env["DOCS_PROBE_VALUE"] = docs_value

    process = subprocess.run(
        [sys.executable, "-c", _PROBE_SOURCE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert process.returncode == 0, (
        "probe subprocess failed for DOCS_ENABLED="
        f"{docs_value!r}\\nstdout:\\n{process.stdout}\\nstderr:\\n{process.stderr}"
    )

    for line in process.stdout.splitlines():
        if line.startswith("DOCS_PROBE_RESULT="):
            return json.loads(line[len("DOCS_PROBE_RESULT="):])

    pytest.fail(
        "probe subprocess printed no result marker for DOCS_ENABLED="
        f"{docs_value!r}\\nstdout:\\n{process.stdout}\\nstderr:\\n{process.stderr}"
    )


# ---------------------------------------------------------------------------
# Default-disabled behaviour
# ---------------------------------------------------------------------------

def test_docs_disabled_when_unset():
    results = _probe("__unset__")

    assert results["/docs"]["status"] == 404
    assert results["/redoc"]["status"] == 404
    assert results["/openapi.json"]["status"] == 404


def test_health_remains_available_when_docs_disabled():
    results = _probe("__unset__")

    assert results["/health"]["status"] == 200


@pytest.mark.parametrize("docs_value", _DISABLED_CASES)
def test_all_disabling_values_keep_docs_disabled(docs_value):
    results = _probe(docs_value)

    assert results["/docs"]["status"] == 404, f"DOCS_ENABLED={docs_value!r}"
    assert results["/redoc"]["status"] == 404, f"DOCS_ENABLED={docs_value!r}"
    assert results["/openapi.json"]["status"] == 404, (
        f"DOCS_ENABLED={docs_value!r}"
    )


# ---------------------------------------------------------------------------
# Enabled behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("docs_value", _ENABLED_CASES)
def test_true_variants_enable_all_documentation_endpoints(docs_value):
    results = _probe(docs_value)

    assert results["/docs"]["status"] == 200, f"DOCS_ENABLED={docs_value!r}"
    assert results["/redoc"]["status"] == 200, f"DOCS_ENABLED={docs_value!r}"
    assert results["/openapi.json"]["status"] == 200, (
        f"DOCS_ENABLED={docs_value!r}"
    )


def test_openapi_schema_is_valid_when_enabled():
    results = _probe("true")

    assert results["/openapi.json"]["status"] == 200
    assert results["/openapi.json"]["is_openapi_schema"] is True


def test_health_remains_available_when_docs_enabled():
    results = _probe("true")

    assert results["/health"]["status"] == 200
