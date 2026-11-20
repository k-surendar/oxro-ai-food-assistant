import os
import sys

# Ensure the project root is importable when pytest is invoked from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Test-environment defaults (set BEFORE main.py is imported).
# Only dummies - never real API keys. Real services are never contacted:
# tests swap in fake Gemini clients / fake chat objects.
# ---------------------------------------------------------------------------
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")
os.environ.setdefault("API_KEY", "test-sandbox-key")
os.environ.setdefault("GEMINI_API_KEY", "dummy-test-key")
