"""
Minimal in-process rate limiting (Phase 6 Step 4).

Purpose:
    - One thread-safe sliding-window limiter class used for two
      independent protections:
          * per-customer request limits (keyed by the VALIDATED
            customer_id, applied in the endpoint layer)
          * per-IP authentication-failure throttling (keyed by the
            socket peer IP only; X-Forwarded-For / X-Real-IP are never
            trusted or parsed - they are client-spoofable when the
            server is directly exposed)
    - Thread-safe: sync endpoints run in a threadpool, so all shared
      state is guarded by one threading.Lock with tiny critical
      sections and nothing awaited inside.
    - No background threads: expired timestamps are pruned lazily on
      every access and keys whose window becomes empty are removed
      entirely, so memory is bounded by keys active within the last
      window.
    - Injectable clock (time.monotonic by default) so tests can slide
      the window virtually without sleeping.

Limits come from environment variables. Missing, invalid, zero or
negative values fall back to the defaults, and rate limiting is
ALWAYS enabled - there is no unlimited or disabled mode:

    RATE_LIMIT_WINDOW_SECONDS      default 60
    RATE_LIMIT_CUSTOMER_REQUESTS   default 20
    RATE_LIMIT_AUTH_FAILURES       default 10
"""

import math
import os
import threading
import time
from collections import deque


DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_RATE_LIMIT_CUSTOMER_REQUESTS = 20
DEFAULT_RATE_LIMIT_AUTH_FAILURES = 10


def load_window_seconds_from_env():
    """RATE_LIMIT_WINDOW_SECONDS; invalid/zero/negative -> default 60."""
    return _load_positive_int_env(
        "RATE_LIMIT_WINDOW_SECONDS",
        DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    )


def load_customer_requests_from_env():
    """RATE_LIMIT_CUSTOMER_REQUESTS; invalid/zero/negative -> default 20."""
    return _load_positive_int_env(
        "RATE_LIMIT_CUSTOMER_REQUESTS",
        DEFAULT_RATE_LIMIT_CUSTOMER_REQUESTS,
    )


def load_auth_failures_from_env():
    """RATE_LIMIT_AUTH_FAILURES; invalid/zero/negative -> default 10."""
    return _load_positive_int_env(
        "RATE_LIMIT_AUTH_FAILURES",
        DEFAULT_RATE_LIMIT_AUTH_FAILURES,
    )


def _load_positive_int_env(name, default):
    raw = os.getenv(name)

    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    if value <= 0:
        return default

    return value


class SlidingWindowRateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    A key is admitted while it has fewer than `limit` recorded events
    inside the last `window_seconds`. Events are admitted (hit) or
    recorded unconditionally (register); rejected events are never
    recorded, so the Retry-After estimate stays stable.

    Modes of use:
        - hit(key)       record one request -> (allowed, retry_after)
        - register(key)  record an event unconditionally (auth failures)
        - is_blocked(key) -> (blocked, retry_after) while the number of
          recorded events inside the window is >= limit

    Retry-After is the number of whole seconds (minimum 1) until the
    OLDEST event inside the window leaves it - admitting again later
    frees a slot (hit) or drops the count below the block threshold
    (is_blocked).
    """

    def __init__(self, limit, window_seconds, clock=time.monotonic):
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, float(window_seconds))
        self._clock = clock
        # key -> deque of monotonic timestamps (oldest first)
        self._events = {}
        self._lock = threading.Lock()

    def _prune_locked(self, key, now):
        """
        Drop timestamps that have left the sliding window. Caller must
        hold self._lock. Removes the key entirely once its deque is
        empty (automatic cleanup). Returns the live deque or None.
        """
        timestamps = self._events.get(key)

        if timestamps is None:
            return None

        while timestamps and (now - timestamps[0]) >= self.window_seconds:
            timestamps.popleft()

        if not timestamps:
            del self._events[key]
            return None

        return timestamps

    def hit(self, key):
        """
        Record one event if capacity allows.

        Returns:
            (True, 0)        - event admitted.
            (False, seconds) - limit reached; retry after `seconds`.
        """
        with self._lock:
            now = self._clock()
            timestamps = self._prune_locked(key, now)

            if timestamps is None or len(timestamps) < self.limit:
                if timestamps is None:
                    timestamps = self._events.setdefault(key, deque())
                timestamps.append(now)
                return True, 0

            retry_after = self.window_seconds - (now - timestamps[0])
            return False, max(1, math.ceil(retry_after))

    def register(self, key):
        """
        Record one event unconditionally (used for authentication
        failures, which are counted but never "admitted").
        """
        with self._lock:
            now = self._clock()
            timestamps = self._prune_locked(key, now)

            if timestamps is None:
                timestamps = self._events.setdefault(key, deque())

            timestamps.append(now)

    def is_blocked(self, key):
        """
        Returns:
            (False, 0)       - not blocked.
            (True, seconds)  - blocked; retry after `seconds`.
        """
        with self._lock:
            now = self._clock()
            timestamps = self._prune_locked(key, now)

            if timestamps is None or len(timestamps) < self.limit:
                return False, 0

            retry_after = self.window_seconds - (now - timestamps[0])
            return True, max(1, math.ceil(retry_after))
