FROM python:3.11-slim

WORKDIR /app

# Production dependencies only (dev/test deps are NOT installed in the image).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Dedicated non-root runtime user (UID 10001).
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Stdlib-only health check: GET /health must return HTTP 200.
# urllib.request.urlopen raises on connection errors, timeouts and
# non-2xx responses, which marks the container unhealthy. The port
# respects the container's PORT environment override.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["sh", "-c", "python -c 'import os, urllib.request; r = urllib.request.urlopen(\"http://127.0.0.1:%s/health\" % os.environ.get(\"PORT\", \"8000\"), timeout=4); exit(0 if r.status == 200 else 1)'"]

# Listen on 0.0.0.0; PORT environment override preserved (default 8000).
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
