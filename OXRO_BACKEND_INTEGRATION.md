# OXRO Backend ↔ AI Chatbot Integration Guide

Audience: the OXRO web/backend team.
Owner: AI Chatbot team (FastAPI service).
Scope: how your Node.js/Express backend calls the AI Food Assistant API
**server-to-server**. The mobile app must never talk to the chatbot
directly and must never hold the chatbot API key.

```
OXRO Mobile App ──(your normal API)──► OXRO Node.js/Express Backend
                                              │
                                              │  X-API-Key (server-to-server)
                                              ▼
                                    AI Food Assistant (FastAPI)
```

---

## 1. Base URL (placeholder)

```
https://YOUR_CHATBOT_HOST
```

Replace `YOUR_CHATBOT_HOST` with the HTTPS hostname your team provisions
for this service (hosting provider, internal domain, etc.). No production
URL is published in this document. All examples below use this placeholder.

Local development:

```
http://127.0.0.1:8000
```

---

## 2. Authentication method

- Shared-secret authentication using the HTTP header **`X-API-Key`**.
- The expected value lives only in the chatbot server environment variable
  **`CHATBOT_API_KEY`** (legacy fallback: `API_KEY`). Your backend stores
  the same value in **its** environment (e.g. `process.env.CHATBOT_API_KEY`).
- The chatbot validates the header with a **constant-time comparison**.
- **Fail-closed**: if the chatbot server has neither variable configured,
  every protected request returns `503 auth_not_configured` (requests are
  never silently allowed).
- Send the key over **HTTPS only**. Never place it in URLs, query strings,
  request bodies, mobile apps, frontend JavaScript, logs or git.

---

## 3. Required header

```http
X-API-Key: YOUR_CHATBOT_API_KEY
Content-Type: application/json
```

`YOUR_CHATBOT_API_KEY` is the secret value provisioned to you by the AI
chatbot team (stored server-side in both environments). It is intentionally
not reproduced in this document.

---

## 4 & 5. Endpoints and HTTP methods

| # | Method | Path | Auth | Purpose |
|---|--------|------|------|---------|
| 1 | `GET` | `/health` | none | Liveness probe (always 200 while process runs) |
| 2 | `GET` | `/ready` | none | Readiness probe (200 when AI client initialized, else 503) |
| 3 | `POST` | `/chat` | `X-API-Key` | Conversational assistant (memory per `customer_id`) |
| 4 | `POST` | `/recommend-food` | `X-API-Key` | Single-shot catalogue-grounded recommendation (no memory) |
| 5 | `POST` | `/session/{customer_id}/end` | `X-API-Key` | Clear a customer's conversation session |

No other endpoints exist. Documentation endpoints (`/docs`, `/redoc`,
`/openapi.json`) are disabled by default.

---

## 6. Required request body fields

### `POST /chat` and `POST /recommend-food`

| Field | Type | Rules |
|-------|------|-------|
| `customer_id` | string | Required. 1–128 chars, charset `[A-Za-z0-9._:-]` only (letters, digits, dot, underscore, colon, hyphen). No surrounding whitespace. **Scope key, not an authenticated identity** — see §11. |
| `message` | string | Required. 1–2000 chars, must not be blank/whitespace-only (422 otherwise). |

### `POST /session/{customer_id}/end`

- `customer_id` is a **path parameter** with the same charset/length rules.
- No body.

---

## 7. Sample JSON requests

### Chat

```http
POST /chat HTTP/1.1
Host: YOUR_CHATBOT_HOST
X-API-Key: YOUR_CHATBOT_API_KEY
Content-Type: application/json
```

```json
{
  "customer_id": "cust-1001",
  "message": "Show me restaurants near me"
}
```

### Recommendation

```http
POST /recommend-food HTTP/1.1
Host: YOUR_CHATBOT_HOST
X-API-Key: YOUR_CHATBOT_API_KEY
Content-Type: application/json
```

```json
{
  "customer_id": "cust-1001",
  "message": "I want something spicy and under 300"
}
```

### End session

```http
POST /session/cust-1001/end HTTP/1.1
Host: YOUR_CHATBOT_HOST
X-API-Key: YOUR_CHATBOT_API_KEY
```

---

## 8. Sample JSON responses

### `POST /chat` — success (HTTP 200)

```json
{
  "customer_id": "cust-1001",
  "reply": "Hi! Here are the restaurants I found for you: ...",
  "detected_preferences": null,
  "matched_foods": [],
  "status": "restaurant_discovery"
}
```

`status` is a machine-readable string. Known values include:
`greeting`, `success` (AI food path), `restaurant_discovery`,
`restaurant_menu`, `restaurant_unavailable`, `restaurant_empty`,
`offers_listed`, `offers_empty`, `offers_unavailable`,
`dish_selection_proposed`, `dish_selected`, `dish_price`,
`selection_cancelled`, `dish_not_on_menu`, and other dish-selection
variants. Treat unknown statuses as generic success when `reply` is present.

### `POST /recommend-food` — success (HTTP 200)

```json
{
  "customer_id": "cust-1001",
  "status": "success",
  "preferences": {"taste": "spicy", "budget": "300"},
  "food_results": ["Hyderabadi Chicken Biryani", "Paneer Butter Masala"],
  "ai_response": "Based on your preferences, try these: ..."
}
```

`status` is `success` or `no_results`.

### `POST /session/{id}/end` — success (HTTP 200)

```json
{
  "customer_id": "cust-1001",
  "status": "ended"
}
```

### `GET /health` — HTTP 200

```json
{
  "status": "ok",
  "service": "OXRO AI Food Assistant API",
  "version": "1.2.0"
}
```

---

## 9. Error response formats

All chatbot errors use FastAPI's `detail` envelope. Two shapes exist:

**Object shape** (most errors):

```json
{
  "detail": {
    "status": "machine_readable_code",
    "reason": "Human-readable explanation."
  }
}
```

**Array shape** (HTTP 422 request-validation errors only):

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "message"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

### Status code catalogue

| HTTP | `detail.status` | When | Client action |
|------|-----------------|------|---------------|
| 401 | `unauthorized` | Missing/invalid `X-API-Key` | Fix key config; do not retry blindly |
| 404 | `session_not_found` | Ending a session that does not exist | Treat as already-ended |
| 413 | `payload_too_large` | Body over 32 KB | Shrink request |
| 422 | *(array)* | Validation failure (bad charset, blank message, etc.) | Fix payload |
| 429 | `rate_limited` | Per-customer rate limit (default 20 req/60 s) or per-IP auth-failure throttle | Wait, honour `Retry-After` header |
| 429 | `quota_exhausted` | AI provider quota exhausted | Back off, retry later |
| 502 | `service_error` | Unexpected AI/backend failure | Retry with backoff (limited) |
| 503 | `auth_not_configured` | Chatbot missing its key env vars | Ops fix on chatbot side |
| 503 | `not_ready` | Gemini client not initialized (`/ready` also) | Wait/retry; check `/ready` |
| 503 | `session_capacity_reached` | Session cap (default 500) reached for a NEW customer | Retry later |
| 503 | `temporary_error` | Transient AI provider failure | Retry shortly |
| 504 | `timeout` | AI provider timed out | Retry with backoff |

Only `429` responses carry a `Retry-After` header (seconds).

---

## 10. Session ID handling

- The chatbot has **no session cookies and no session tokens**.
  Conversation memory is scoped by `customer_id`, which you send on every
  `/chat` call.
- Sessions expire after `SESSION_TTL_SECONDS` (default **3600 s**), are
  recreated automatically on the next message (fresh memory), and are
  capped at `MAX_SESSIONS` (default **500** concurrent).
- `POST /session/{customer_id}/end` deletes the conversation immediately
  (404 `session_not_found` if none exists — safe to treat as idempotent).
- End the session when the user logs out or you want to guarantee a fresh
  conversation; otherwise TTL handles cleanup.

---

## 11. Customer context handling

- `customer_id` is an **opaque isolation scope key** for the chatbot — it
  is **not** an authenticated identity. The chatbot performs no login and
  verifies no JWT.
- **Your backend must derive `customer_id` server-side from your verified
  user session** (e.g. `cust-<your-user-id>`). Never pass a raw
  client-supplied customer id through to the chatbot without verifying it
  belongs to the authenticated user — otherwise one user could read another
  user's conversation memory (the chatbot will happily scope to any valid
  id).
- Use a **stable, validated internal customer identifier** that already
  meets the chatbot's format (1–128 chars from `[A-Za-z0-9._:-]`, no
  surrounding whitespace — see §6). Never sanitize or transform it into
  shape: stripping/replacing characters can collide two different users
  onto one conversation scope. If the identifier does not qualify, fix it
  upstream instead (see the `customerIdFor()` example in §14).
- Do not send personal data beyond the id. The chatbot logs never record
  message bodies or keys.
- Guest users: use a stable per-device/per-session guest id from your
  backend (same charset rules).

---

## 12. Timeout and retry recommendations

Recommended settings for your backend → chatbot calls:

| Setting | Value |
|---------|-------|
| Connect timeout | 3–5 s |
| Total request timeout | **10 s** (AI turns can take a few seconds) |
| Retries on `429` | Honour `Retry-After`; max 1–2 attempts |
| Retries on `502/503/504` | Max 2 attempts, exponential backoff (e.g. 500 ms, 2 s) |
| Retries on `401/404/413/422` | **Never** (determinent failures) |

Idempotency caution: `/chat` and `/recommend-food` consume AI quota; only
retry when the connection failed before a response, or on the 5xx/429
codes above. `/session/{id}/end` may be retried freely.

Circuit-breaker suggestion: if `/ready` returns 503 for an extended
period, fail fast to your client with a friendly "assistant unavailable"
message instead of queueing requests.

---

## 13. Deployment and environment-variable requirements

Chatbot service (your counterpart team operates it; for reference):

| Variable | Purpose |
|----------|---------|
| `CHATBOT_API_KEY` | Shared server-to-server key (required; `API_KEY` is legacy fallback) |
| `GEMINI_API_KEY` | AI provider key; without it the service runs degraded (greetings still work, `/ready`=503) |
| `OXRO_BACKEND_BASE_URL` | Base URL **of your Node backend** that the chatbot uses for its read-only public catalogue calls (`/restaurants`, `/search`, `/coupons`). Must be set or restaurant/menu/offers features fail closed with a polite "unavailable" reply |
| `CORS_ORIGINS` | Browser allow-list — **not needed for server-to-server calls** |
| `SESSION_TTL_SECONDS` / `MAX_SESSIONS` | Conversation memory lifetime / capacity |
| `RATE_LIMIT_CUSTOMER_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | Per-customer chat rate limit (default 20/60 s) |
| `RATE_LIMIT_AUTH_FAILURES` | Per-IP 401 throttle (protects against key guessing) |
| `DOCS_ENABLED` | Keep `false` in production |
| `PORT` | Listen port (default 8000) |

Your backend requirements:

1. Store `CHATBOT_API_KEY` in your server environment/secret manager only.
2. All chatbot traffic stays server-side over HTTPS.
3. Never log the key value (redact `X-API-Key` in access logs).
4. Rotate the key immediately if it is ever exposed; coordinate with the
   chatbot team to update both sides.
5. Keep your **public read-only** `GET /restaurants`,
   `GET /restaurants/{id}`, `GET /restaurants/{id}/menu`, `GET /search`,
   and `GET /coupons` reachable from the chatbot's server (for
   `OXRO_BACKEND_BASE_URL`). Do **not** expose any customer-scoped
   endpoints without auth (see chatbot team's backend security audit).
6. Probe `GET /health` (liveness) and `GET /ready` (readiness) from your
   deployment/monitoring.

---

## 14. Node.js / Express example

```javascript
// chatbotClient.js - server-side only. NEVER import this from frontend code.
const axios = require('axios');

const CHATBOT_BASE_URL = process.env.CHATBOT_BASE_URL; // https://YOUR_CHATBOT_HOST
const CHATBOT_API_KEY = process.env.CHATBOT_API_KEY;   // shared secret

if (!CHATBOT_BASE_URL || !CHATBOT_API_KEY) {
  throw new Error('CHATBOT_BASE_URL and CHATBOT_API_KEY must be set in the server environment');
}

const chatbot = axios.create({
  baseURL: CHATBOT_BASE_URL,
  timeout: 10_000, // total request timeout (ms)
  headers: {
    'X-API-Key': CHATBOT_API_KEY,
    'Content-Type': 'application/json',
  },
  validateStatus: () => true, // classify statuses ourselves
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function callChatbot(path, body, { retries = 2 } = {}) {
  let attempt = 0;
  for (;;) {
    const response = await chatbot.post(path, body).catch((err) => {
      // Network/timeout error before a response: retryable.
      if (attempt < retries) {
        attempt += 1;
        return null;
      }
      throw err;
    });

    if (response === null) {
      await sleep(500 * 2 ** (attempt - 1));
      continue;
    }

    const { status, data, headers } = response;

    if (status >= 200 && status < 300) {
      return data;
    }

    const retriable = status === 429 || status === 502 || status === 503 || status === 504;
    if (retriable && attempt < retries) {
      attempt += 1;
      const retryAfter = Number(headers['retry-after']);
      const delay = Number.isFinite(retryAfter) && retryAfter > 0
        ? retryAfter * 1000
        : 500 * 2 ** (attempt - 1);
      await sleep(delay);
      continue;
    }

    const error = new Error(`Chatbot request failed: HTTP ${status}`);
    error.status = status;
    error.detail = data && data.detail;
    throw error;
  }
}

// ---- Express routes (mounted behind YOUR user authentication) ----------

function requireUser(req, res, next) {
  // Your existing JWT/session verification middleware goes here.
  if (!req.user || !req.user.id) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  next();
}

// customer_id rules accepted by the chatbot (see section 6): 1-128
// characters from [A-Za-z0-9._:-] (letters, digits, dot, underscore,
// colon, hyphen), no surrounding whitespace.
const CUSTOMER_ID_RE = /^[A-Za-z0-9._:-]{1,128}$/;

// IMPORTANT: customer_id is derived from the VERIFIED user session,
// never taken from the mobile client's request body, and must be your
// STABLE internal customer/user identifier used as-is.
//
// Do NOT sanitize or transform the id (stripping or replacing invalid
// characters, truncating, etc.): distinct users can then map onto the
// same conversation scope - a collision and a cross-user data leak. A
// fixed constant prefix (e.g. "cust-") is safe because it is
// one-to-one; character removal is not. Your internal identifier must
// already satisfy the format above; if it does not, fix the
// identifier upstream - never mangle it here.
function customerIdFor(user) {
  const id = String(user.id);
  if (!CUSTOMER_ID_RE.test(id)) {
    throw new Error('internal user id does not satisfy chatbot customer_id format');
  }
  return id;
}

// Resolve the customer id once per request; on a misconfigured
// identity mapping, fail closed instead of "repairing" the value.
function resolveCustomerId(req, res) {
  try {
    return customerIdFor(req.user);
  } catch {
    res.status(500).json({ error: 'identity_mapping_invalid' });
    return null;
  }
}

app.post('/api/assistant/chat', requireUser, async (req, res) => {
  const customerId = resolveCustomerId(req, res);
  if (!customerId) return;
  try {
    const data = await callChatbot('/chat', {
      customer_id: customerId,
      message: String(req.body.message || '').slice(0, 2000),
    });
    res.json(data);
  } catch (err) {
    if (err.status === 429) {
      res.setHeader('Retry-After', (err.detail && err.detail.retryAfter) || '60');
      return res.status(429).json({ error: 'assistant_busy' });
    }
    // Never leak chatbot internals or the API key to clients.
    res.status(502).json({ error: 'assistant_unavailable' });
  }
});

app.post('/api/assistant/recommend', requireUser, async (req, res) => {
  const customerId = resolveCustomerId(req, res);
  if (!customerId) return;
  try {
    const data = await callChatbot('/recommend-food', {
      customer_id: customerId,
      message: String(req.body.message || '').slice(0, 2000),
    });
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: 'assistant_unavailable' });
  }
});

app.post('/api/assistant/logout', requireUser, async (req, res) => {
  const customerId = resolveCustomerId(req, res);
  if (!customerId) return;
  try {
    await callChatbot(`/session/${encodeURIComponent(customerId)}/end`, {}, { retries: 1 });
  } catch (err) {
    // 404 session_not_found is fine - already ended.
  }
  res.json({ ok: true });
});
```

---

## Appendix A — Features currently implemented (chatbot side)

| Feature | Status |
|---------|--------|
| `X-API-Key` server-to-server auth (constant-time, fail-closed) | ✅ Ready |
| `POST /chat` conversational pipeline: greeting → restaurant discovery → menu → dish selection → offers → grounded AI food path | ✅ Ready |
| `POST /recommend-food` single-shot grounded recommendations | ✅ Ready |
| `POST /session/{id}/end` conversation reset | ✅ Ready |
| Per-customer conversation memory with TTL + capacity controls | ✅ Ready |
| Per-customer rate limiting + per-IP auth-failure throttling | ✅ Ready |
| 32 KB body cap, docs disabled by default, hardened logging (no bodies/keys) | ✅ Ready |
| `GET /health`, `GET /ready` probes | ✅ Ready |
| Restaurant/menu/offers features (require `OXRO_BACKEND_BASE_URL` → your public read-only endpoints) | ✅ Ready when URL configured |
| Internal local cart store + session-bound cart state (in development) | ⚠️ Internal only — **not exposed via any API** |

## Appendix B — Not implemented / not production-ready

| Feature | Status |
|---------|--------|
| Cart API endpoints / cart sync with your backend | ❌ Not implemented (internal dev store only) |
| Customer profile endpoints | ❌ Not implemented |
| Coupon apply / validate | ❌ List/display only |
| Checkout, payments, order placement | ❌ Out of scope by design |
| User login / JWT verification inside the chatbot | ❌ Out of scope — your backend authenticates users (see §11) |
| Real-time push / websockets | ❌ Not implemented |
| Multi-worker shared rate-limit/session state | ⚠️ Per-process only (single-worker deployment recommended) |

## Appendix C — Integration requirements checklist (web team)

1. Provision `CHATBOT_API_KEY` in **both** environments from a secret
   manager; never commit it, never send it to clients.
2. Proxy every chatbot call through your backend (example in §14).
3. Derive `customer_id` from your **verified** user session only (§11).
4. Apply the timeout/retry policy in §12; surface `assistant_unavailable`
   gracefully in the app.
5. Set `CHATBOT_BASE_URL` (placeholder → real host at deployment time)
   in your backend environment.
6. Keep your public read-only catalogue endpoints reachable by the
   chatbot server and set `OXRO_BACKEND_BASE_URL` on the chatbot side.
7. Do not log `X-API-Key` headers or chatbot request bodies with user
   content in your access logs.
8. Rotate the shared key on any exposure; coordinate both sides.
9. Monitor `/health` and `/ready`.
10. Do not begin live integration until both teams sign off on the
    security audit items (backend JWT verification / customer IDOR).

---

*Document version: Phase 7 handover. All URLs above are placeholders;
no production deployment URL or secret value is contained in this file.*
