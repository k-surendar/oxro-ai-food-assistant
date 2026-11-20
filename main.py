"""
OXRO AI Food Assistant API - unified FastAPI backend.

Purpose:
    - One backend serving both the OXRO mobile application and website.
    - Gemini client is created exactly once at startup (FastAPI lifespan)
      using the GEMINI_API_KEY environment variable. No keys are hardcoded
      and the client is passed to modules explicitly (no monkey-patching).

Security (Phase 3):
    - X-API-Key authentication on all customer endpoints, fail-closed:
      if neither CHATBOT_API_KEY (preferred) nor legacy API_KEY is
      configured on the server, every protected request is rejected
      with 503 auth_not_configured (never silently allowed).
    - /health and /ready stay unauthenticated for hosting-platform probes.
    - CORS allow-list from the CORS_ORIGINS environment variable
      (comma-separated). If unset/empty, no browser origin is allowed;
      mobile apps are not affected by CORS.
    - API keys are never exposed in responses and internal exception
      details are never returned to clients.

Security (Phase 6):
    - customer_id values are restricted to OXRO identifier characters
      (letters, digits, dot, underscore, colon, hyphen), 1-128 chars.
    - Request bodies above 32 KB are rejected with 413
      (payload_too_large) before reaching the endpoint logic.
    - NEW customers are rejected with a static 503
      session_capacity_reached when MAX_SESSIONS (default 500) is
      full; expired sessions are swept first and existing customers
      are never affected.
    - Minimal in-process rate limiting: per-customer request limits
      (429 rate_limited with a Retry-After header) on all customer
      endpoints and per-IP authentication-failure throttling (socket
      peer IP only; X-Forwarded-For is never trusted).
    - Documentation endpoints (/docs, /redoc, /openapi.json) are
      disabled by default and only enabled when DOCS_ENABLED is
      exactly "true" (case-insensitive); the value is read once at
      import, so changing it requires an application restart.

Endpoints:
    GET  /health                     liveness (always 200 while process runs)
    GET  /ready                      readiness (503 until Gemini initialized)
    POST /chat                       conversational, catalogue-grounded assistant
    POST /recommend-food             single-shot catalogue-grounded recommendation
    POST /session/{customer_id}/end  clears a customer's conversation session
"""

import logging
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Path, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from starlette.responses import JSONResponse

from google import genai

from ai.chat_service import greeting_reply, handle_food_message, is_greeting
from ai.conversation_manager import (
    GeminiQuotaError,
    GeminiServiceError,
    GeminiTimeoutError,
    GeminiTransientError,
)
from ai.dish_selection import handle_dish_selection
from ai.offer_discovery import is_offer_request, offer_reply
from ai.recommendation_assistant import recommend_food_with_ai
from ai.restaurant_discovery import (
    is_restaurant_discovery,
    restaurant_discovery_reply,
)
from ai.restaurant_menu import is_menu_request, restaurant_menu_reply
from ai.session_manager import (
    SessionCapacityError,
    SessionManager,
    SessionNotFoundError,
)
from rate_limiter import (
    SlidingWindowRateLimiter,
    load_auth_failures_from_env,
    load_customer_requests_from_env,
    load_window_seconds_from_env,
)


logger = logging.getLogger("oxro.api")


def _load_log_level_from_env():
    """
    Optional LOG_LEVEL environment configuration (default INFO).
    Invalid values fall back to INFO, matching the SESSION_TTL_SECONDS
    error-handling pattern.
    """
    raw = os.getenv("LOG_LEVEL")

    if not raw:
        return logging.INFO

    level = logging.getLevelName(raw.strip().upper())

    if isinstance(level, int):
        return level

    return logging.INFO


def _load_docs_enabled_from_env():
    """
    Optional DOCS_ENABLED environment configuration (default disabled).

    Documentation endpoints (/docs, /redoc, /openapi.json) are
    fail-closed: they are enabled ONLY when DOCS_ENABLED is exactly
    the string "true" (case-insensitive, surrounding whitespace
    ignored). Any other value - unset, empty, "false", "1", "yes" or
    anything else - keeps them disabled. The value is read once at
    import and applied during application construction, so changing
    it requires an application restart.
    """
    raw = os.getenv("DOCS_ENABLED")

    if not raw:
        return False

    return raw.strip().lower() == "true"


logger.setLevel(_load_log_level_from_env())


# ---------------------------------------------------------------------------
# Configuration (environment-based)
# ---------------------------------------------------------------------------

# Read once at import: CORS middleware must be registered before startup.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

# Read once at import: applied during FastAPI construction below
# (restart required after changing DOCS_ENABLED).
DOCS_ENABLED = _load_docs_enabled_from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.

    The API process ALWAYS starts. A missing GEMINI_API_KEY or a Gemini
    SDK failure puts the process in a degraded state
    (gemini_client=None, gemini_init_error set) so /health stays
    available and /ready reports 503. Restart is required after adding
    or changing the key (no hot reload).
    """
    app.state.gemini_client = None
    app.state.session_manager = None
    app.state.gemini_init_error = None

    # Rate limiting (Phase 6 Step 4): fresh limiter state on every
    # startup, so every TestClient lifespan in tests starts clean.
    app.state.rate_limiter = SlidingWindowRateLimiter(
        limit=load_customer_requests_from_env(),
        window_seconds=load_window_seconds_from_env(),
    )
    app.state.auth_failure_limiter = SlidingWindowRateLimiter(
        limit=load_auth_failures_from_env(),
        window_seconds=load_window_seconds_from_env(),
    )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        app.state.gemini_init_error = (
            "GEMINI_API_KEY environment variable is not set. "
            "Set it (environment or .env, never in code) and restart the "
            "API to enable AI responses."
        )
        logger.warning(
            "Starting in DEGRADED mode: %s", app.state.gemini_init_error
        )
    else:
        try:
            client = genai.Client(api_key=api_key)

            app.state.gemini_client = client
            app.state.session_manager = SessionManager(client)
            logger.info("Gemini client initialized successfully.")

        except Exception as exc:
            # The raw exception is logged server-side only; the
            # client-facing reason is a static, safe message so no
            # internal details (or key material) can leak via /ready.
            app.state.gemini_init_error = (
                "Gemini client initialization failed. "
                "Check the server logs and restart the API."
            )
            logger.error("Gemini initialization failed: %s", exc)

    yield

    app.state.gemini_client = None
    app.state.session_manager = None
    app.state.rate_limiter = None
    app.state.auth_failure_limiter = None


app = FastAPI(
    title="OXRO AI Food Assistant API",
    description=(
        "AI-powered food discovery and recommendation service "
        "for the OXRO mobile app and website."
    ),
    version="1.2.0",
    # Fail-closed documentation endpoints (Phase 6 Step 6): only
    # DOCS_ENABLED=true (case-insensitive) exposes /docs, /redoc and
    # /openapi.json. Unset/empty/any other value keeps them disabled.
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
    lifespan=lifespan
)


# CORS allow-list (no wildcard; empty list = no browser origin allowed).
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


# ---------------------------------------------------------------------------
# Request body size limit (Phase 6 Step 2)
# ---------------------------------------------------------------------------

# /chat and /recommend-food messages are capped at 2000 characters by
# Pydantic, so 32 KB is a generous ceiling. Anything larger is rejected
# with 413 before it can consume memory or bandwidth.
MAX_BODY_BYTES = 32 * 1024


class BodySizeLimitMiddleware:
    """
    Pure-ASGI middleware that rejects request bodies larger than
    MAX_BODY_BYTES with HTTP 413 (payload_too_large).

    Safety rules:
        - A valid Content-Length above the limit returns 413
          immediately, WITHOUT buffering or reading the body.
        - A missing or misleading Content-Length falls through to
          streaming enforcement: body chunks are consumed (never
          buffered beyond the limit) and rejected as soon as the
          accumulated size exceeds the cap.
        - Accepted bodies are replayed to the inner app unchanged, so
          normal JSON request handling is preserved.
        - GET/OPTIONS requests (no body) pass through untouched.
    """

    def __init__(self, app, max_body_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") not in ("POST", "PUT", "PATCH")
        ):
            await self.app(scope, receive, send)
            return

        content_length = None
        for name, value in scope.get("headers") or []:
            if name == b"content-length":
                content_length = value.decode("latin-1")
                break

        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._send_payload_too_large(
                        scope, receive, send
                    )
                    return
            except ValueError:
                # Misleading/non-numeric Content-Length: fall through to
                # streaming enforcement, which cannot be lied to.
                pass

        body = bytearray()
        exceeded = False

        while True:
            message = await receive()

            if message["type"] == "http.disconnect":
                return

            body.extend(message.get("body", b""))

            if len(body) > self.max_body_bytes:
                exceeded = True
                body.clear()  # never buffer beyond the limit
                # Drain the remaining chunks without storing them so
                # the transport stays consistent, then reject.
                while message.get("more_body", False):
                    message = await receive()
                break

            if not message.get("more_body", False):
                break

        if exceeded:
            await self._send_payload_too_large(scope, receive, send)
            return

        replayed_body = bytes(body)

        async def replay_receive():
            return {
                "type": "http.request",
                "body": replayed_body,
                "more_body": False,
            }

        await self.app(scope, replay_receive, send)

    async def _send_payload_too_large(self, scope, receive, send):
        response = JSONResponse(
            status_code=413,
            content={
                "detail": {
                    "status": "payload_too_large",
                    "reason": "Request body exceeds the 32 KB limit.",
                }
            },
        )
        await response(scope, receive, send)


app.add_middleware(BodySizeLimitMiddleware)


# ---------------------------------------------------------------------------
# Authentication-failure throttling (Phase 6 Step 4)
# ---------------------------------------------------------------------------

class AuthFailureThrottleMiddleware:
    """
    Pure-ASGI middleware that temporarily blocks client IPs with too
    many HTTP 401 authentication failures on the protected customer
    endpoints (/chat, /recommend-food and /session/{customer_id}/end).

    Rules:
        - Counts ONLY 401 responses on the protected paths. 503
          auth_not_configured (server misconfiguration) is never
          counted; 404/422/405 responses are never counted either.
        - Keys are the SOCKET PEER IP (scope["client"]) only.
          X-Forwarded-For and X-Real-IP are never trusted or parsed
          (they are client-spoofable when the server is directly
          exposed; trusted-proxy support is out of scope).
        - While an IP is blocked, protected requests get a static 429
          rate_limited response immediately, BEFORE any body reading.
          Unrelated public endpoints are never blocked, and the
          unauthenticated probes /health and /ready stay reachable.
        - The limiter instance is read from app.state at request time
          (created fresh in the lifespan), so tests get clean state
          per TestClient context.

    Registered OUTSIDE the body-size middleware (so blocked IPs are
    rejected before any body reading) and INSIDE the request-logging
    middleware (so 429 responses are logged like everything else).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Only the protected customer endpoints are counted/blocked;
        # every other path (including /health and /ready) passes
        # through untouched.
        is_protected = (
            path == "/chat"
            or path == "/recommend-food"
            or path.startswith("/session/")
        )

        if not is_protected:
            await self.app(scope, receive, send)
            return

        limiter = getattr(
            getattr(scope.get("app"), "state", None),
            "auth_failure_limiter",
            None,
        )

        if limiter is None:
            await self.app(scope, receive, send)
            return

        client_ip = self._client_ip(scope)

        blocked, retry_after = limiter.is_blocked(client_ip)

        if blocked:
            await self._send_rate_limited(scope, receive, send, retry_after)
            return

        observed_status = {}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                observed_status["status"] = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        if observed_status.get("status") == 401:
            limiter.register(client_ip)

    @staticmethod
    def _client_ip(scope):
        client = scope.get("client")

        if not client or not client[0]:
            return "unknown"

        return client[0]

    async def _send_rate_limited(self, scope, receive, send, retry_after):
        response = JSONResponse(
            status_code=429,
            content={
                "detail": {
                    "status": "rate_limited",
                    "reason": "Too many requests. Please try again later.",
                }
            },
            headers={"Retry-After": str(retry_after)},
        )
        await response(scope, receive, send)


app.add_middleware(AuthFailureThrottleMiddleware)


# ---------------------------------------------------------------------------
# Request logging (Phase 6 Step 1)
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Minimal request logging: HTTP method, request path, response status
    and duration.

    Privacy rules (never logged):
        - API keys / authorization headers
        - request or response bodies
        - customer messages or Gemini responses
        - personal customer information

    The route's path TEMPLATE is logged when available, so dynamic
    segments (e.g. the customer_id in /session/{customer_id}/end) are
    never recorded. Unmatched paths have no template and fall back to
    the raw path (which by definition carries no valid customer data).
    """
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0

    route = request.scope.get("route")
    logged_path = getattr(route, "path", request.url.path)

    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        logged_path,
        response.status_code,
        duration_ms,
    )

    return response


# ---------------------------------------------------------------------------
# Request / response models (Pydantic validation)
# ---------------------------------------------------------------------------

# Customer IDs are OXRO-internal identifiers: letters, digits, dot,
# underscore, colon and hyphen only. Whitespace, newlines, control
# characters and everything else are rejected (422).
_CUSTOMER_ID_PATTERN = re.compile(r"[A-Za-z0-9._:\-]+")


class ChatRequest(BaseModel):
    customer_id: str = Field(
        min_length=1,
        max_length=128,
        description="Unique OXRO customer identifier for conversation memory."
    )
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="The customer's chat message."
    )

    @field_validator("message")
    @classmethod
    def value_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace")
        return stripped

    @field_validator("customer_id")
    @classmethod
    def customer_id_charset(cls, value: str) -> str:
        # Identifiers are never silently transformed: surrounding
        # whitespace is rejected, not stripped away.
        if not value or not value.strip():
            raise ValueError("must not be blank or whitespace")

        if value != value.strip():
            raise ValueError(
                "customer_id must not contain leading or trailing "
                "whitespace"
            )

        if not _CUSTOMER_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "customer_id may only contain letters, digits, dot, "
                "underscore, colon and hyphen (1-128 characters)"
            )

        return value


class ChatResponse(BaseModel):
    customer_id: str
    reply: str
    detected_preferences: Optional[dict]
    matched_foods: List[str]
    status: str


class RecommendResponse(BaseModel):
    customer_id: str
    status: str
    preferences: Optional[dict] = None
    food_results: List[str] = []
    ai_response: str


class SessionEndResponse(BaseModel):
    customer_id: str
    status: str


# ---------------------------------------------------------------------------
# Authentication (fail-closed X-API-Key)
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(x_api_key: Optional[str] = Security(api_key_header)):
    """
    Validates the X-API-Key header against the CHATBOT_API_KEY
    environment variable (legacy fallback: API_KEY, used when
    CHATBOT_API_KEY is unset or empty). Server-to-server callers (the
    OXRO Node.js backend) send the shared key on every protected
    request. Fail-closed: when NEITHER variable is configured on the
    server, ALL protected requests are rejected (503), never allowed.

    Security rules:
        - The expected key is only ever read from the environment;
          it is never hardcoded, logged, echoed or returned in any
          response, error message or documentation.
        - Comparison is constant-time (secrets.compare_digest).
    """
    expected = os.getenv("CHATBOT_API_KEY") or os.getenv("API_KEY")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "auth_not_configured",
                "reason": (
                    "CHATBOT_API_KEY (or legacy API_KEY) environment "
                    "variable is not configured on the server. Contact "
                    "the OXRO backend team."
                )
            }
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail={
                "status": "unauthorized",
                "reason": "Missing or invalid X-API-Key header."
            }
        )


# ---------------------------------------------------------------------------
# State helpers (degraded-mode awareness)
# ---------------------------------------------------------------------------

def _get_session_manager():
    session_manager = getattr(app.state, "session_manager", None)

    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": (
                    getattr(app.state, "gemini_init_error", None)
                    or "Gemini client is not initialized."
                )
            }
        )

    return session_manager


def _get_gemini_client():
    client = getattr(app.state, "gemini_client", None)

    if client is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": (
                    getattr(app.state, "gemini_init_error", None)
                    or "Gemini client is not initialized."
                )
            }
        )

    return client


def _get_rate_limiter():
    """
    Per-customer rate limiter (created fresh at startup).

    Returns None only when the lifespan never ran - a state real
    traffic cannot reach (the server accepts requests only after
    startup completes). In that case the customer rate limit fails
    open, preserving the approved degraded-mode contract that
    greetings must work without any lifespan-initialized
    infrastructure.
    """
    return getattr(app.state, "rate_limiter", None)


def _enforce_customer_rate_limit(customer_id):
    """
    Per-customer rate limit for /chat, /recommend-food and
    /session/{customer_id}/end. Runs in the endpoint body, i.e. AFTER
    authentication and request validation (422/401 win), and counts
    greeting requests too. Exceeding the limit raises a static 429
    with a Retry-After header.

    If the limiter is missing (lifespan never ran - unreachable for
    real traffic), the limit fails open so the greeting path keeps
    its guaranteed no-infrastructure behaviour.
    """
    rate_limiter = _get_rate_limiter()

    if rate_limiter is None:
        return

    allowed, retry_after = rate_limiter.hit(customer_id)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "status": "rate_limited",
                "reason": "Too many requests. Please try again later."
            },
            headers={"Retry-After": str(retry_after)},
        )


# ---------------------------------------------------------------------------
# Health / readiness (unauthenticated, for hosting platforms)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Liveness probe: always 200 while the API process is alive.
    Never touches Gemini.
    """
    return {
        "status": "ok",
        "service": "OXRO AI Food Assistant API",
        "version": app.version
    }


@app.get("/ready")
def ready():
    """
    Readiness probe: 200 when the Gemini client is initialized, 503
    otherwise (with reason). Does not ping Gemini (quota protection).
    """
    if getattr(app.state, "gemini_client", None) is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": (
                    getattr(app.state, "gemini_init_error", None)
                    or "Gemini client is not initialized"
                )
            }
        )

    return {
        "status": "ready",
        "service": "OXRO AI Food Assistant API",
        "version": app.version
    }


# ---------------------------------------------------------------------------
# Customer endpoints (X-API-Key protected)
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _: None = Security(require_api_key)):
    """
    Conversational food assistant.

    - Greetings are answered LOCALLY first: they work even when Gemini
      is unavailable (degraded mode), use no Gemini quota and create no
      session.
    - Food requests require the Gemini client: they run through the
      OXRO catalogue search first; matched items are passed to Gemini,
      which must recommend ONLY those items.
    - Conversation history is preserved per customer_id.
    """
    # Rate limit first: greetings count too, and this runs after auth
    # and request validation (422/401 take precedence).
    _enforce_customer_rate_limit(request.customer_id)

    # Greeting short-circuit BEFORE any Gemini/SessionManager access.
    if is_greeting(request.message):
        result = greeting_reply(request.message)
        return ChatResponse(
            customer_id=request.customer_id,
            reply=result["reply"],
            detected_preferences=result["detected_preferences"],
            matched_foods=result["matched_foods"],
            status=result["status"]
        )

    # Restaurant discovery short-circuit (Phase 7 Step 2): answered
    # locally from the OXRO backend - no Gemini, no session, works in
    # degraded mode. Dish-level food requests still take the path below.
    if is_restaurant_discovery(request.message):
        result = restaurant_discovery_reply(request.message)
        return ChatResponse(
            customer_id=request.customer_id,
            reply=result["reply"],
            detected_preferences=result["detected_preferences"],
            matched_foods=result["matched_foods"],
            status=result["status"]
        )

    # Restaurant menu short-circuit (Phase 7 Step 3): answered locally
    # from the OXRO backend - no Gemini. On success the structured menu
    # is stored on the customer's session when a SessionManager is
    # available (Phase 7 Step 4 selection); capacity errors degrade to
    # "menu shown, selection unavailable" and never fail the reply.
    if is_menu_request(request.message):
        result = restaurant_menu_reply(request.message)

        if result.get("menu_context") is not None:
            session_manager = getattr(app.state, "session_manager", None)
            if session_manager is not None:
                try:
                    session_manager.set_menu_state(
                        request.customer_id,
                        result["menu_context"],
                    )
                except SessionCapacityError:
                    logger.warning(
                        "Menu state not stored: session capacity full"
                    )
                except Exception as exc:
                    logger.error(
                        "Menu state store failure (%s)",
                        type(exc).__name__,
                    )

        return ChatResponse(
            customer_id=request.customer_id,
            reply=result["reply"],
            detected_preferences=result["detected_preferences"],
            matched_foods=result["matched_foods"],
            status=result["status"]
        )

    session_manager = _get_session_manager()

    # Dish selection short-circuit (Phase 7 Step 4): confirmation-only
    # picks from a previously viewed menu - no Gemini, no cart/orders.
    # Returns None when the message is not a selection concern so the
    # food path below stays unchanged.
    selection_result = handle_dish_selection(
        request.customer_id,
        request.message,
        session_manager,
    )
    if selection_result is not None:
        return ChatResponse(
            customer_id=request.customer_id,
            reply=selection_result["reply"],
            detected_preferences=selection_result["detected_preferences"],
            matched_foods=selection_result["matched_foods"],
            status=selection_result["status"]
        )

    # Offer discovery short-circuit (Phase 7 Step 5): answered locally
    # from the OXRO backend - no Gemini, no session creation, works in
    # degraded mode when a SessionManager is present but unused. Runs
    # AFTER dish selection so confirmations/price/cancel keep priority.
    # Coupon apply/validate and cart/checkout stay out of scope.
    if is_offer_request(request.message):
        result = offer_reply(request.message)
        return ChatResponse(
            customer_id=request.customer_id,
            reply=result["reply"],
            detected_preferences=result["detected_preferences"],
            matched_foods=result["matched_foods"],
            status=result["status"]
        )

    try:
        result = handle_food_message(
            request.customer_id,
            request.message,
            session_manager
        )

    except GeminiQuotaError:
        raise HTTPException(
            status_code=429,
            detail={
                "status": "quota_exhausted",
                "reason": "AI service quota exhausted. Please retry later."
            }
        )

    except GeminiTransientError:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "temporary_error",
                "reason": "AI service temporarily unavailable. Please retry shortly."
            }
        )

    except SessionCapacityError:
        # NEW customer, but MAX_SESSIONS is full (the session manager
        # sweeps expired sessions first; existing customers are never
        # affected). Retries are client-managed for now.
        raise HTTPException(
            status_code=503,
            detail={
                "status": "session_capacity_reached",
                "reason": "The service is temporarily at session capacity. Please try again later."
            }
        )

    except GeminiTimeoutError:
        raise HTTPException(
            status_code=504,
            detail={
                "status": "timeout",
                "reason": "AI service timed out. Please retry shortly."
            }
        )

    except GeminiServiceError:
        raise HTTPException(
            status_code=502,
            detail={
                "status": "service_error",
                "reason": "AI service failure."
            }
        )

    except Exception:
        # Any unexpected bug maps to a safe 502; internal details are
        # never exposed to clients (FastAPI's default 500 would also be
        # safe, but 502 keeps the client contract consistent).
        # Server-side only: the full traceback goes to the log so real
        # bugs are debuggable; nothing is returned to the client.
        logger.exception("Unhandled error in /chat")

        raise HTTPException(
            status_code=502,
            detail={
                "status": "service_error",
                "reason": "AI service failure."
            }
        )

    return ChatResponse(
        customer_id=request.customer_id,
        reply=result["reply"],
        detected_preferences=result["detected_preferences"],
        matched_foods=result["matched_foods"],
        status=result["status"]
    )


@app.post("/recommend-food", response_model=RecommendResponse)
def recommend_food(request: ChatRequest, _: None = Security(require_api_key)):
    """
    Single-shot catalogue-grounded recommendation (no conversation
    memory). Gemini receives only catalogue-matched food items and is
    instructed never to invent items, prices or availability.

    Error handling is identical to /chat: the recommendation module
    raises the same typed Gemini errors, mapped here to the same HTTP
    statuses (429/503/504/502). Internal details are never exposed.
    """
    _enforce_customer_rate_limit(request.customer_id)

    client = _get_gemini_client()

    try:
        result = recommend_food_with_ai(request.message, client)

    except GeminiQuotaError:
        raise HTTPException(
            status_code=429,
            detail={
                "status": "quota_exhausted",
                "reason": "AI service quota exhausted. Please retry later."
            }
        )

    except GeminiTransientError:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "temporary_error",
                "reason": "AI service temporarily unavailable. Please retry shortly."
            }
        )

    except GeminiTimeoutError:
        raise HTTPException(
            status_code=504,
            detail={
                "status": "timeout",
                "reason": "AI service timed out. Please retry shortly."
            }
        )

    except GeminiServiceError:
        raise HTTPException(
            status_code=502,
            detail={
                "status": "service_error",
                "reason": "AI service failure."
            }
        )

    except Exception:
        # Any unexpected bug maps to a safe 502; internal details are
        # never exposed to clients.
        # Server-side only: the full traceback goes to the log so real
        # bugs are debuggable; nothing is returned to the client.
        logger.exception("Unhandled error in /recommend-food")

        raise HTTPException(
            status_code=502,
            detail={
                "status": "service_error",
                "reason": "AI service failure."
            }
        )

    # Only "success" and "no_results" remain as dict statuses.
    return RecommendResponse(
        customer_id=request.customer_id,
        status=result["status"],
        preferences=result.get("preferences"),
        food_results=result.get("food_results", []),
        ai_response=result["ai_response"]
    )


@app.post("/session/{customer_id}/end", response_model=SessionEndResponse)
def end_customer_session(
    customer_id: str = Path(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:\-]+$"
    ),
    _: None = Security(require_api_key)
):
    """
    Clears a customer's conversation session. Subsequent /chat messages
    from this customer start a fresh conversation.
    """
    _enforce_customer_rate_limit(customer_id)

    session_manager = _get_session_manager()

    try:
        session_manager.end_session(customer_id)

    except SessionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "session_not_found",
                "reason": "No active session for this customer."
            }
        )

    return SessionEndResponse(
        customer_id=customer_id,
        status="ended"
    )
