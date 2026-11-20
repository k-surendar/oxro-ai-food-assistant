# OXRO AI Food Assistant API

Unified FastAPI backend for AI food discovery, serving both the OXRO
mobile application and the OXRO website. Migrated from the original
Google Colab notebook (which remains untouched as a backup).

All recommendations are grounded in the local OXRO catalogue
(`food_recommendation/food_categories.json`, 30 categories / 291
items). Nothing is invented: Gemini only ever receives real catalogue
items that a rule-based search matched.

## Project structure and file purposes

```
oxro-chatbot/
├── main.py                          FastAPI app. Gemini client + SessionManager
│                                    are created once at startup (lifespan) from
│                                    the GEMINI_API_KEY environment variable.
│                                    Endpoints: GET /health, GET /ready,
│                                    POST /chat, POST /recommend-food and
│                                    POST /session/{customer_id}/end.
│                                    X-API-Key auth (fail-closed) and a CORS
│                                    allow-list from CORS_ORIGINS protect the
│                                    customer endpoints.
├── ai/
│   ├── chat_service.py              /chat pipeline: local greeting path first
│   │                                (no Gemini quota, no session, works even
│   │                                when Gemini is unavailable / degraded),
│   │                                then catalogue search, preference merging
│   │                                and a grounded message (matched items +
│   │                                guardrails) sent through the customer's
│   │                                chat session. Phase 4: contextual
│   │                                follow-ups ("Make it spicy") reuse the
│   │                                previously matched items; continuation
│   │                                requests ("Show more", "Suggest
│   │                                something else") surface genuine
│   │                                catalogue alternatives without repeats.
│   ├── conversation_manager.py      Creates a Gemini chat per customer with the
│   │                                OXRO system instruction; send_message raises
│   │                                typed errors (quota / transient / timeout /
│   │                                service) and passes them through untouched.
│   ├── session_manager.py           One isolated conversation per customer_id.
│   │                                Thread-safe with a persistent per-customer
│   │                                lock: same-customer send/end/create
│   │                                operations serialize in order; different
│   │                                customers run in parallel. TTL-based
│   │                                expiry controlled by SESSION_TTL_SECONDS
│   │                                (default 3600). Stores per-customer
│   │                                conversation context (merged preferences +
│   │                                previously matched items, most recent 30
│   │                                kept).
│   ├── recommendation_assistant.py  Merged catalogue-grounded recommendation
│   │                                prompt (single-shot, no conversation
│   │                                memory). Only catalogue results are passed
│   │                                to Gemini; raises the same typed errors
│   │                                as /chat (5xx retried with backoff).
│   └── greetings.py                 Local greeting handling so greetings do not
│                                    consume Gemini quota.
├── food_recommendation/
│   ├── food_categories.json         OXRO catalogue: 30 categories, 291 items.
│   ├── language_phrases.py          Shared multilingual utility: Unicode
│   │                                normalization (NFC, zero-width chars,
│   │                                punctuation), Tamil/Tanglish phrase
│   │                                dictionaries, food-word
│   │                                canonicalization (biriyani -> biryani),
│   │                                fillers, negation words and Tanglish
│   │                                continuation patterns. No network, no
│   │                                translation APIs.
│   ├── recommendation_engine.py     Category listing / recommendation / search.
│   │                                (Deterministic now - no random sampling.)
│   ├── preference_classifier.py     Rule-based veg/non-veg, taste, health,
│   │                                budget and category detection (English).
│   ├── food_search.py               Smart search pipeline: keyword match ->
│   │                                category fallback -> dietary fallback ->
│   │                                general popular fallback (broad requests
│   │                                like "Suggest vegetarian food") ->
│   │                                dietary filter -> exact-match-first
│   │                                ordering. Junk input ("zzzqqq") still
│   │                                returns no results.
│   └── ranking_engine.py            Keyword scoring/ranking applied to /chat
│                                    results before the grounded prompt is
│                                    built. Category-relevance bonus (+2)
│                                    applies only when the ITEM's own
│                                    category matches the detected customer
│                                    category.
├── tests/
│   ├── test_catalogue.py            Asserts 30 categories / 291 entries intact.
│   ├── test_food_logic.py           Search, filter, ranking and classifier
│   │                                outputs vs notebook-verified results.
│   ├── test_session_manager.py      Session isolation, TTL expiration,
│   │                                deletion, concurrency, context storage.
│   ├── test_ranking_engine.py       Category-relevance bonus rules.
│   ├── test_api_health.py           /health liveness, /ready readiness,
│   │                                degraded startup (real lifespan runs).
│   ├── test_api_endpoints.py        Endpoint contract: X-API-Key auth
│   │                                (401 / fail-closed 503), validation
│   │                                (422), /chat greeting and grounded
│   │                                flow, Gemini error mapping
│   │                                (429/503/504/502), /recommend-food
│   │                                mapping, /session end (200/404),
│   │                                degraded mode 503, CORS allow-list,
│   │                                context-aware follow-ups.
│   ├── test_chat_context.py         Phase 4: preference merging and
│   │                                overrides, contextual follow-ups,
│   │                                continuation alternatives with no
│   │                                repeats, dietary respect, price
│   │                                honesty, context isolation/expiry,
│   │                                ranking integration.
│   ├── test_multilingual.py         Phase 5: Unicode normalization,
│                                    Tamil script / Tanglish / mixed
│                                    language detection, food-word
│                                    canonicalization, negation and
│                                    junk-input regression, multilingual
│                                    greetings, continuations and
│                                    conversation-context flows.
│   ├── test_logging.py              Phase 6: request logging (route
│   │                                templates, never keys/bodies/
│   │                                messages/customer data) and
│   │                                LOG_LEVEL configuration.
│   ├── test_input_limits.py         Phase 6: customer_id charset
│   │                                validation and the 32 KB body
│   │                                limit (413 payload_too_large).
│   ├── test_session_capacity.py     Phase 6: MAX_SESSIONS gate,
│   │                                expired-session sweep and
│   │                                503 session_capacity_reached.
│   ├── test_rate_limiting.py        Phase 6: sliding-window limits
│   │                                (429 + Retry-After), per-IP
│   │                                auth-failure throttle, spoofed
│   │                                forwarded-header immunity.
│   └── test_docs_endpoints.py       Phase 6: DOCS_ENABLED control of
│                                      /docs, /redoc, /openapi.json
│                                      (isolated subprocess probes).
├── rate_limiter.py                   Thread-safe in-process
│                                     sliding-window rate limiter
│                                     (Phase 6): per-customer request
│                                     limits and per-IP auth-failure
│                                     throttling; limits come from the
│                                     RATE_LIMIT_* environment
│                                     variables.
├── conftest.py                      Makes the project root importable for pytest;
│                                    sets dummy test environment defaults.
├── requirements.txt                 Production dependencies (exact pins).
├── requirements-dev.txt             Development/test dependencies
│                                    (includes requirements.txt).
├── Dockerfile                       Docker image for the OXRO sandbox server.
├── .env.example                     Template for environment configuration.
├── .gitignore                       Keeps .env and caches out of git.
└── README.md                        This file.
```

## Environment variables

| Variable | Purpose | Default behaviour if unset |
|----------|---------|---------------------------|
| `GEMINI_API_KEY` | Gemini API key for the AI responses. Configure via environment or `.env`, never in code. | Process still starts in **degraded mode** (`/ready` = 503); restart required after setting it |
| `CHATBOT_API_KEY` | Shared **server-to-server** key the OXRO backend must send as the `X-API-Key` header. Store only in the local git-ignored `.env`; never in code, logs, responses or git. | Fail-closed unless legacy `API_KEY` is set |
| `API_KEY` | Legacy fallback, used only when `CHATBOT_API_KEY` is unset or empty. | Fail-closed: every protected request returns `503 auth_not_configured` |
| `CORS_ORIGINS` | Comma-separated browser (website) origin allow-list, e.g. `http://localhost:3000`. | No browser origin is allowed (fail-closed); mobile apps are unaffected by CORS |
| `LOG_LEVEL` | Application log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`); invalid values fall back to the default. | `INFO` |
| `SESSION_TTL_SECONDS` | Customer session lifetime in seconds; invalid or non-positive values fall back to the default. | 3600 |
| `MAX_SESSIONS` | Maximum simultaneous sessions; new customers get `503 session_capacity_reached` while full (expired sessions are swept first); invalid or non-positive values fall back to the default. | 500 |
| `RATE_LIMIT_WINDOW_SECONDS` | Sliding-window size for both rate limiters; invalid, zero or negative values fall back to the default. | 60 |
| `RATE_LIMIT_CUSTOMER_REQUESTS` | Per-customer request cap inside the window on the protected endpoints; invalid, zero or negative values fall back to the default. | 20 |
| `RATE_LIMIT_AUTH_FAILURES` | Per-IP cap on HTTP 401 authentication failures inside the window; invalid, zero or negative values fall back to the default. | 10 |
| `PORT` | Listen port inside the Docker container (used by the container CMD and its health check, not read by the Python code). | 8000 |
| `DOCS_ENABLED` | Enables `/docs`, `/redoc` and `/openapi.json` **only** when set to exactly `true` (case-insensitive); any other value keeps them disabled. Restart required after changing it. | Documentation endpoints disabled |

Never commit `.env` to git. Any key previously exposed in plaintext
files must be rotated. This repository contains no real keys - only
`.env.example` placeholders and test dummies.

## Local Windows setup

```powershell
cd C:\Users\SURENDAR\oxro-chatbot
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env    # then fill in GEMINI_API_KEY and CHATBOT_API_KEY
```

To run the test suite, install the development dependencies instead
(they include everything in `requirements.txt`):

```powershell
pip install -r requirements-dev.txt
```

## Running tests

```powershell
cd C:\Users\SURENDAR\oxro-chatbot
.venv\Scripts\activate
.venv\Scripts\python.exe -m pytest tests\ -v
```

Current verified result: **274 tests passed, 0 failed.**

Tests do not call Gemini - they use fake chat objects and fake
clients, so no API key or network access is required for the full
suite.

## Running the API locally (smoke)

```powershell
.venv\Scripts\activate
uvicorn main:app --port 8000
# then open http://127.0.0.1:8000/health
```

The documentation endpoints (`/docs`, `/redoc`, `/openapi.json`) are
**disabled by default** (fail-closed). To browse the Swagger UI,
start the process with `DOCS_ENABLED=true` in the environment -
the value is read once at import, so **restart the application**
after changing it:

```powershell
$env:DOCS_ENABLED = "true"
uvicorn main:app --port 8000
# then open http://127.0.0.1:8000/docs
```

Without `GEMINI_API_KEY` the process still starts (degraded mode):
`/health` returns 200 and `/ready` reports 503. With the key set,
`/ready` returns 200.

## Docker (prepared for OXRO sandbox server)

```bash
docker build -t oxro-chatbot .
docker run -p 8000:8000 --env-file .env oxro-chatbot
```

Hardening already implemented in the Dockerfile:

- Runs as the dedicated non-root user `appuser` (UID 10001).
- A stdlib-only `HEALTHCHECK` requests `/health` every 30 s
  (5 s timeout, 10 s start period, 3 retries) and respects `PORT`.
- The image installs production dependencies only
  (`requirements.txt`); the tests and dev/test dependencies never
  enter the image (`.dockerignore`).

`PORT` can override the listen port inside the container.

The image has **not** been build/runtime validated yet - Docker is
not available in the development environment. Validation is planned
for the OXRO sandbox (Phase 7).

## API endpoints

### GET /health
Liveness probe. Always 200 while the process is alive; never touches
Gemini.

```json
{ "status": "ok", "service": "OXRO AI Food Assistant API", "version": "1.2.0" }
```

### GET /ready
Readiness probe. 200 once the Gemini client is initialized, 503
otherwise (with a safe reason). Deliberately does not ping Gemini so
probes cannot burn quota.

```json
{ "status": "ready", "service": "OXRO AI Food Assistant API", "version": "1.2.0" }
```

### POST /chat
Conversational, catalogue-grounded assistant with per-customer memory.
Protected by `X-API-Key`.

Request:

```json
{ "customer_id": "C1", "message": "I want chicken biryani" }
```

(customer_id 1-128 chars, message 1-2000 chars; blank/whitespace
rejected with 422.)

Response 200 (food request):

```json
{
  "customer_id": "C1",
  "reply": "...",
  "detected_preferences": { "food_type": "non-vegetarian", "taste": null, "health": null, "budget": null, "category": "Biryani" },
  "matched_foods": ["Chicken Biryani"],
  "status": "success"
}
```

Response 200 (greeting - answered locally, no Gemini quota, no session
created, works even when Gemini is unavailable):

```json
{
  "customer_id": "C1",
  "reply": "Hello! Welcome to OXRO Food Delivery Support. How may I help you today?",
  "detected_preferences": null,
  "matched_foods": [],
  "status": "greeting"
}
```

Follow-up example: after the food request above, a message like
"Make it spicy." keeps the previously matched catalogue item
(`matched_foods` stays `["Chicken Biryani"]`) and merges the new
preference (`detected_preferences.taste` becomes `"spicy"`).
"Show more." returns genuine catalogue alternatives that were not
previously recommended.

### POST /recommend-food
Single-shot catalogue-grounded recommendation (no conversation
memory). Protected by `X-API-Key`. `no_results` returns 200 with an
empty `food_results` list.

Request: same shape as `/chat`.

Response 200:

```json
{
  "customer_id": "C1",
  "status": "success",
  "preferences": { "...": "..." },
  "food_results": ["Chicken Biryani"],
  "ai_response": "..."
}
```

### POST /session/{customer_id}/end
Clears the customer's conversation session (including its stored
context). Protected by `X-API-Key`.

Response 200:

```json
{ "customer_id": "C1", "status": "ended" }
```

Response 404 (no active session):

```json
{ "detail": { "status": "session_not_found", "reason": "No active session for this customer." } }
```

### Error mapping (all protected endpoints; shape: `{"detail": {"status", "reason"}}`)

| HTTP | status | Trigger |
|------|--------|---------|
| 401 | `unauthorized` | Missing/invalid X-API-Key |
| 404 | `session_not_found` | Session end requested for a customer without an active session |
| 413 | `payload_too_large` | Request body above 32 KB |
| 422 | *(FastAPI validation format)* | Body fails Pydantic validation (blank/whitespace message, invalid customer_id characters, over-long fields) |
| 429 | `rate_limited` | Local rate limit exceeded (per-customer request cap or per-IP authentication-failure throttle); includes a `Retry-After` header - **not** related to Gemini quota |
| 429 | `quota_exhausted` | Gemini quota exhausted (upstream 429) - distinct from the local `rate_limited` 429 above |
| 502 | `service_error` | Any other Gemini failure or unexpected bug |
| 503 | `auth_not_configured` | Server has neither `CHATBOT_API_KEY` nor legacy `API_KEY` configured |
| 503 | `not_ready` | Process started degraded (no Gemini key) |
| 503 | `temporary_error` | Gemini temporarily unavailable (5xx) |
| 503 | `session_capacity_reached` | MAX_SESSIONS full (new customers only) |
| 504 | `timeout` | Gemini call timed out (408 / client timeout) |

Internal exception details are never returned to clients; they are
logged server-side only. One nuance: `422` validation responses
follow the FastAPI default format and can contain field-level
validation details for the submitted body.

## Authentication and CORS

- Every customer endpoint requires an `X-API-Key` header. The expected
  value is the server's **`CHATBOT_API_KEY`** environment variable
  (preferred, server-to-server key); the legacy **`API_KEY`** variable
  is used only when `CHATBOT_API_KEY` is unset or empty. The comparison
  uses `secrets.compare_digest` (constant-time).
- **Fail-closed:** if neither variable is configured on the server,
  every protected request is rejected with `503 auth_not_configured` -
  never silently allowed. `/health` and `/ready` stay unauthenticated
  for hosting-platform probes.
- CORS: browser (website) origins come from `CORS_ORIGINS`
  (comma-separated). Unset/empty means no browser origin is allowed.
  There is no wildcard support by design. Mobile apps are not
  affected by CORS.

## Security and configuration summary

- **API key authentication:** every customer endpoint requires the
  `X-API-Key` header, compared in constant time against the server's
  `CHATBOT_API_KEY` (preferred) or legacy `API_KEY` environment
  variable. If neither is configured, every protected request is
  rejected with `503 auth_not_configured`
  (fail-closed); nothing is ever silently allowed. `/health` and
  `/ready` stay unauthenticated for hosting-platform probes.
- **CORS allow-list:** browser origins come from `CORS_ORIGINS`
  (comma-separated). Unset/empty means no browser origin is allowed;
  there is no wildcard support by design.
- **Request body limit:** bodies above 32 KB are rejected with
  `413 payload_too_large` before any endpoint logic runs.
- **Customer ID validation:** 1-128 characters from letters, digits,
  dot, underscore, colon and hyphen only; surrounding whitespace is
  rejected, never stripped.
- **Session capacity and expiration:** at most `MAX_SESSIONS`
  (default 500) simultaneous sessions; sessions expire after
  `SESSION_TTL_SECONDS` (default 3600). When capacity is full, NEW
  customers receive `503 session_capacity_reached` (expired sessions
  are swept first); existing customers are never affected.
- **Rate limiting:** a per-customer request cap
  (`RATE_LIMIT_CUSTOMER_REQUESTS`, default 20 per
  `RATE_LIMIT_WINDOW_SECONDS`, default 60 s) on the protected
  endpoints, plus a per-IP authentication-failure throttle
  (`RATE_LIMIT_AUTH_FAILURES`, default 10 per window; only HTTP 401
  responses are counted). Excess requests receive
  `429 rate_limited` with a `Retry-After` header.
  - **Per-process limitation:** the limiters live in process memory.
    Limits reset on restart and are NOT shared between multiple
    worker processes.
- **Client IP handling:** the authentication-failure throttle keys
  on the socket peer IP only. `X-Forwarded-For` and `X-Real-IP` are
  never trusted or parsed (they are client-spoofable when the server
  is directly exposed).
- **Error detail policy:** internal exception details are logged
  server-side and never returned to clients; generic error bodies
  carry only a fixed `{"status", "reason"}` pair. One nuance: `422`
  validation responses follow the FastAPI default format and can
  contain field-level validation details for the submitted body.
- **Documentation endpoints:** `/docs`, `/redoc` and
  `/openapi.json` are disabled by default (fail-closed) and enabled
  only when `DOCS_ENABLED=true` (case-insensitive). Changing the
  value requires an application restart.
- **Secrets hygiene:** never commit `.env` to git (it is
  git-ignored). Store API keys in environment variables or an
  appropriate secret manager, and rotate any key that has ever been
  exposed in a plaintext file.
- **Client integration rule:** the shared chatbot `X-API-Key` must
  NEVER be embedded in a public mobile application or website
  JavaScript - anyone could extract it there. The trusted OXRO
  backend should authenticate end users itself and call this API
  server-to-server, attaching the `X-API-Key` header on its own
  server side.

## Session context and conversational memory

- Each customer's session stores isolated context:
  `{"preferences": {...}, "matched_foods": [...]}` (merged preferences
  and the catalogue items already recommended to them, most recent 30
  kept, deduplicated).
- **Preference retention with overrides:** a newly detected explicit
  preference overrides the stored one; unset values inherit.
  Example: "I want chicken biryani" (non-veg) followed by
  "What about vegetarian?" switches the stored preference to
  vegetarian in both filtering and recommendations, while
  "Make it spicy." keeps non-vegetarian and adds taste.
- **Refinement follow-ups** ("Make it spicy.", "Show me cheaper
  options."): when the message produces no new direct search matches,
  the previously matched catalogue items are embedded in the grounded
  follow-up message so Gemini can respond conversationally using ONLY
  those real items. The shown list is preserved; merged preferences
  are refreshed.
- **Continuation requests** ("Show more.", "Suggest something
  else."): the pipeline searches the real catalogue for genuine
  alternatives that were NOT previously recommended (stored category
  first, then dietary/general fallback categories), ranked with
  `rank_food_results`. If no unshown alternatives remain, the response
  honestly falls back to the earlier items instead of inventing new
  ones.
- Context is isolated per customer, reset when an expired session is
  recreated, and removed with the session on
  `POST /session/{customer_id}/end`.
- Concurrency: same-customer operations (send / context read / context
  update / end) are serialized through a persistent per-customer lock;
  different customers never block each other.
  - **Ordering caveat:** `threading.Lock` guarantees mutual exclusion
    but NOT strict FIFO arrival order - under concurrent same-customer
    requests a later-arriving request could be processed first. Clients
    should send the next message only after receiving the previous
    response (normal chat UX).
  - **Lock accumulation limitation:** one small lock object per unique
    customer_id is kept for the process lifetime (deliberate fix for a
    deletion race; negligible memory, not yet cleaned up).

## Dietary filtering

The vegetarian/non-vegetarian filter is **name-based, not
ingredient-level**: it reads food names only (e.g. "Grilled Chicken
Salad" is excluded from vegetarian results because the name contains
"chicken"). A hypothetically non-vegetarian dish whose name contains
no meat word would be treated as vegetarian. The merged dietary
preference is applied to direct matches, continuation alternatives and
follow-up context alike. Ingredient-level verification requires
structured menu data from the OXRO backend (future integration).

Dietary preference *detection* understands English, Tamil script and
Tanglish (saivam/சைவம், asaivam/அசைவம், koli, meen, kari, ...), but the
filtering itself always runs on the English catalogue item names.

## What is implemented vs. what is not

**Implemented (catalogue-based, verified by tests):**
- Catalogue keyword/category/dietary search over the 30-category,
  291-item local OXRO catalogue
- Preference detection and conversation-aware preference merging
- Multilingual understanding (English, Tamil script, Tanglish and
  mixed messages): greetings, preferences (veg/non-veg, spicy/mild/
  sweet, healthy, budget), food-word canonicalization (biriyani ->
  biryani, koli -> chicken), filler words and negation handling
  ("biryani venam" = don't want biryani); same-language reply
  instructions in the Gemini prompts
- Catalogue-grounded Gemini recommendations: Gemini only ever sees
  real matched catalogue items and is instructed never to invent food,
  restaurants, prices, ratings, availability or order status
- Contextual follow-ups and continuation requests (no repeats when
  alternatives exist)
- Honest price limitation: the catalogue contains no price
  information, so the assistant never claims an item is cheaper or
  more expensive and explains this if asked
- Session memory, per-customer isolation, TTL expiry, session end

**NOT implemented (future live OXRO backend integrations):**
- Live restaurant menus or real-time catalogue synchronization
- Prices, discounts or offers
- Live availability, stock or orderable status
- Cart, checkout or ordering capabilities
- Payments
- Order tracking, status updates or refunds

The assistant never claims an item is currently available, in stock or
orderable - live availability must be confirmed by the backend. These
integrations await the OXRO team's sandbox server and backend API
documentation.

## Known limitations

1. **Multilingual support is dictionary-based, not fuzzy.** English,
   Tamil script and Tanglish are understood for greetings, preference
   classification, food searches and continuation requests, but only
   through the curated phrase dictionaries and spelling variants in
   `language_phrases.py` (with conservative Tamil stem matching).
   Unlisted spellings, dialects or sentence structures will fall
   through to the existing behaviour (honest no-match or English
   assumptions). Negation is handled at keyword/category level
   ("biryani venam" excludes biryani), but negating a dietary
   preference ("chicken venam" while requesting other non-veg food)
   is not yet understood. The sweet taste maps to the Desserts
   category when no other category is detected (an edge case such as
   "sweet corn soup" is labelled sweet without changing search
   results).
2. **Dietary filtering is name-based, not ingredient-level** (see the
   section above).
3. **Conversation memory is in-process.** Sessions and their context
   are isolated per customer and serialized safely, but they do not
   survive a process restart. Persistent storage (Redis/DB) is a
   future step.
4. **Free-tier Gemini quota (~20 requests/day)** applies until a
   higher quota is configured.
5. **Readiness does not ping Gemini** by design (probes would burn
   quota); transient Gemini unavailability surfaces per request.
6. **Degraded mode is static.** If the API starts without
   `GEMINI_API_KEY`, it stays degraded until restarted with the key
   configured; there is no runtime re-initialization.
7. **Per-customer locks accumulate** for the process lifetime (deliberate
   fix for the deletion race; negligible memory, but not yet cleaned up).
8. **No live OXRO backend integration yet** - see the section above;
   menus, prices, availability, cart, checkout, payments, order
   tracking and refunds are not connected.

## Deployment status

- No deployment has been made. Waiting on OXRO team for sandbox server
  details and backend API documentation.
- No external hosting provider (Render/Railway/etc.) is configured.

## Roadmap

- Phase 3 (done): /chat, /recommend-food, /session/{customer_id}/end
  endpoints, X-API-Key authentication (fail-closed), CORS allow-list,
  structured error responses, greeting availability without Gemini.
- Phase 4 (done): catalogue + conversational memory integration -
  session context, preference merging with overrides, contextual
  follow-ups, continuation alternatives without repeats, ranking in
  the chat pipeline.
- Phase 5 (done): Tamil / Tanglish / mixed-language support -
  shared language phrase module, Unicode normalization, multilingual
  greetings and preference detection, food-word canonicalization,
  negation handling, multilingual continuations, same-language reply
  instructions.
- Phase 6 (implemented): production security and error-handling
  hardening - request logging, customer_id validation and the 32 KB
  body limit, the MAX_SESSIONS capacity gate, rate limiting, exact
  dependency pinning, and documentation endpoints disabled by
  default behind `DOCS_ENABLED`.
- Phase 7: final Docker hardening for the OXRO sandbox.
- Phase 8: API documentation and handover package.
