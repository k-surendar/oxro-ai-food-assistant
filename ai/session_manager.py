"""
Customer session manager.

Migrated from the OXRO notebook (ai/session_manager.py).

Purpose:
    - Keeps one isolated Gemini chat per customer_id (conversation memory).
    - Provides create / send / end session operations.

Concurrency strategy (review points 3 + review-pass point 1):
    - Each customer_id has ONE persistent threading.Lock, created on first
      use and deliberately NEVER removed for the process lifetime.
    - create_session(), send_customer_message() and end_session() all
      acquire the customer's lock, so:
        * messages from the SAME customer are serialized in order and
          never interleave (Gemini chat objects are not thread-safe);
        * an in-flight message completes before a concurrent
          end_session()/create_session() takes effect;
        * a new lock can never be created while another operation for
          that customer is running (locks are not popped on deletion -
          this removes the deletion race where a new lock could be
          created alongside an old, still-held one).
    - Different customers are NOT blocked by each other.
    - Lock ordering is always customer-lock -> global-lock, so no deadlock.
    - Trade-off: customer locks accumulate for the process lifetime (one
      small Lock object per unique customer_id). Session data itself is
      deleted by end_session(); the lock is just an empty sync primitive.

Session expiry:
    - Each session carries a created_at timestamp and is lazily evicted
      when older than the configured TTL. TTL comes from the
      SESSION_TTL_SECONDS environment variable (default 3600 seconds)
      and can also be set per instance.

Session capacity (Phase 6 Step 3):
    - MAX_SESSIONS (environment variable, default 500; unset, invalid
      or non-positive values fall back to the default - there is no
      unlimited mode) caps the number of tracked session entries.
    - NEW customers are capacity-gated BEFORE their persistent lock is
      created, so an ordinary capacity rejection creates neither a
      lock nor a session. Existing customers are always allowed to use
      or recreate their own session, even when capacity is full
      (replacing their own slot never changes the total count).
    - When capacity is reached, expired sessions are swept first
      (sweeping happens ONLY at capacity). A session is swept only if
      it is expired AND its customer lock is not currently held (a
      non-blocking Lock.locked() peek - customer locks are never
      acquired during the sweep). An in-flight operation therefore can
      never lose its session, and only expired sessions are removed,
      so active conversation context is never discarded. Any future
      acquirer of a swept customer's lock must first pass the global
      lock or simply recreates the expired session on entry (the
      existing lazy-expiry behaviour).
    - Race note: in the rare window between the cheap pre-check and
      the authoritative gate inside the creation critical section, a
      race loser may leave ONE empty Lock object behind (the same
      documented trade-off as lifetime lock retention). The session
      count itself can never exceed the cap: the capacity check and
      the session insert are atomic under the global lock.

Conversation context (Phase 4):
    - Each session additionally stores a small "context" dict:
          {"preferences": {...}, "matched_foods": [...]}
      holding the customer's merged preferences and the catalogue items
      matched in their most recent successful food turn. The chat
      pipeline uses this to understand follow-up messages such as
      "make it spicy" or "show more" without re-searching from scratch,
      while keeping recommendations grounded in real catalogue items.
    - Context is isolated per customer_id (it lives inside that
      customer's session), is reset when an expired session is
      recreated, and disappears together with the session on
      end_session(). Stored matched-food lists are capped
      (MAX_CONTEXT_MATCHED_FOODS, most recent entries kept) so prompts
      stay small while the newest recommendations are retained.

Cart state (Phase 7 Step 6.2):
    - Each session MAY hold a "cart" entry: one instance of
      ai.cart_store.CartStore for that customer, created lazily by
      ensure_cart() via the injectable cart_store_factory (defaults
      to CartStore). The Step 6.1 store is used UNCHANGED - its
      validation, quantity-update, subtotal and envelope logic are
      reused as-is.
    - get_cart() is read-only: it never creates a session, a customer
      lock or a cart, and returns None when the session is missing or
      expired (same contract as get_menu_state()).
    - Because the cart object lives INSIDE the session entry, every
      existing deletion path - end_session(), the capacity sweep, lazy
      expiry recreation and create_session() reset - discards it
      automatically. No cleanup hooks are added to those paths.
    - ensure_cart() follows the set_menu_state() lifecycle exactly:
      capacity-gated for NEW customers (SessionCapacityError at full
      MAX_SESSIONS), coordinated on the customer lock then the global
      lock, and an expired session is recreated fresh with an EMPTY
      cart.

Other behaviour:
    - end_session() raises SessionNotFoundError so the API layer can
      return a proper 404.
    - chat_factory is injectable so tests can run without a real Gemini
      connection.
"""

import copy
import os
import threading
import time

from ai.cart_store import CartStore
from ai.conversation_manager import create_food_chat, send_message


class SessionNotFoundError(Exception):
    """Raised when ending or using a session that does not exist."""


class SessionCapacityError(Exception):
    """
    Raised when a NEW customer cannot be admitted because the session
    capacity (MAX_SESSIONS) is full. The API layer maps this to a
    static 503 session_capacity_reached response. Existing customers
    are never affected by this error.
    """


DEFAULT_SESSION_TTL_SECONDS = 3600
DEFAULT_MAX_SESSIONS = 500

# Upper bound for the matched-foods list kept in session context so the
# grounded prompt sent to Gemini stays small even after broad fallback
# searches (up to ~20 items today).
MAX_CONTEXT_MATCHED_FOODS = 30


def _default_chat_factory(client):
    return create_food_chat(client)


def _load_ttl_from_env():
    raw = os.getenv("SESSION_TTL_SECONDS")

    if not raw:
        return DEFAULT_SESSION_TTL_SECONDS

    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_SESSION_TTL_SECONDS


def _load_max_sessions_from_env():
    """
    MAX_SESSIONS from the environment. Unset, invalid or non-positive
    values fall back to DEFAULT_MAX_SESSIONS - there is no unlimited
    mode (same error-handling pattern as SESSION_TTL_SECONDS).
    """
    raw = os.getenv("MAX_SESSIONS")

    if not raw:
        return DEFAULT_MAX_SESSIONS

    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_SESSIONS

    if value <= 0:
        return DEFAULT_MAX_SESSIONS

    return value


class SessionManager:

    def __init__(self, client, session_ttl_seconds=None, chat_factory=None,
                 max_sessions=None, cart_store_factory=None):
        self.client = client
        self.chat_factory = chat_factory or _default_chat_factory
        # Phase 7 Step 6.2: one CartStore per session, created lazily
        # by ensure_cart(). Injectable so tests can substitute a fake
        # without touching the real Step 6.1 store.
        self.cart_store_factory = cart_store_factory or CartStore

        if session_ttl_seconds is None:
            self.session_ttl_seconds = _load_ttl_from_env()
        else:
            self.session_ttl_seconds = max(1, int(session_ttl_seconds))

        if max_sessions is None:
            self.max_sessions = _load_max_sessions_from_env()
        else:
            self.max_sessions = max(1, int(max_sessions))

        # customer_id -> {"chat": ..., "created_at": monotonic timestamp,
        #                 "context": {"preferences": ..., "matched_foods": ...}}
        self.sessions = {}
        # customer_id -> threading.Lock (persistent, never removed; see
        # module docstring for the deletion-race rationale).
        self._customer_locks = {}
        # Protects the two dicts above.
        self._lock = threading.Lock()

    def _is_expired(self, session):
        age = time.monotonic() - session["created_at"]
        return age > self.session_ttl_seconds

    def _get_customer_lock(self, customer_id):
        """
        Get the customer's persistent lock, creating it only on first use.
        Because locks are never removed, every operation for a customer
        always coordinates on the same lock object.
        """
        with self._lock:
            customer_lock = self._customer_locks.get(customer_id)

            if customer_lock is None:
                customer_lock = threading.Lock()
                self._customer_locks[customer_id] = customer_lock

        return customer_lock

    def _sweep_expired_safe_locked(self):
        """
        Remove expired sessions that are provably safe to delete.
        Caller must hold self._lock.

        Safe-to-remove predicate (BOTH must hold):
            1. The session is expired (age > TTL). By the TTL contract
               its context is already defined as "reset on recreate",
               so removing it discards nothing that is active.
            2. The customer's persistent lock is NOT currently held.
               An actively running operation (send / create / end /
               update) always holds the customer lock, so a held lock
               means the session may be in use and is skipped - an
               in-flight operation can never lose its session.
               A missing lock entry means no operation for that
               customer can be running (every operation creates the
               lock first), so it is safe to remove.

        The check is a non-blocking PEEK (Lock.locked()); customer
        locks are never ACQUIRED here, so the existing customer-lock
        -> global-lock ordering is preserved and no deadlock or lock
        inversion can occur. A future acquirer of a swept customer's
        lock either must first pass the global lock (held here) or,
        if already past it, simply recreates the expired session on
        entry - the existing lazy-expiry behaviour.
        """
        for customer_id in list(self.sessions.keys()):
            session = self.sessions.get(customer_id)

            if session is None or not self._is_expired(session):
                continue

            customer_lock = self._customer_locks.get(customer_id)

            if customer_lock is not None and customer_lock.locked():
                continue  # operation possibly in flight - skip

            del self.sessions[customer_id]

    def _gate_new_customer_locked(self, customer_id):
        """
        Capacity gate for NEW customers. Caller must hold self._lock.

        Existing customers (session present, live or expired) always
        pass: recreating or replacing their own slot never changes the
        total count, so they keep working at full capacity.

        When the session dict is at capacity, expired sessions are
        swept first (sweep only happens at capacity). If the dict is
        still full afterwards, the new customer is rejected with
        SessionCapacityError - atomically with any insert, because the
        gate and the insert both run under self._lock.
        """
        if customer_id in self.sessions:
            return

        if len(self.sessions) < self.max_sessions:
            return

        self._sweep_expired_safe_locked()

        if len(self.sessions) >= self.max_sessions:
            raise SessionCapacityError(
                f"Session capacity ({self.max_sessions}) is full."
            )

    def _reject_new_customer_if_capacity_reached(self, customer_id):
        """
        Cheap pre-check run BEFORE the customer's persistent lock is
        created, so an ORDINARY capacity rejection creates neither a
        lock nor a session for the rejected customer.

        Race note: two new customers may pass this pre-check
        simultaneously. The authoritative gate inside the creation
        critical section (customer lock -> global lock) then rejects
        the loser, so the cap is never exceeded; in that rare race the
        rejected customer may leave one empty Lock object behind
        (documented, accepted trade-off).
        """
        with self._lock:
            self._gate_new_customer_locked(customer_id)

    def create_session(self, customer_id):
        """
        Create a fresh conversation session for the customer.
        Re-creates (resets) the session if it already exists or has expired.
        Waits for any in-flight operation for this customer to finish.

        NEW customers are capacity-gated: at full capacity they are
        rejected with SessionCapacityError (before any lock creation
        via the pre-check). Existing customers may always reset their
        own session, even when capacity is full.
        """
        self._reject_new_customer_if_capacity_reached(customer_id)

        customer_lock = self._get_customer_lock(customer_id)

        with customer_lock:
            with self._lock:
                self._gate_new_customer_locked(customer_id)

                self.sessions[customer_id] = {
                    "chat": self.chat_factory(self.client),
                    "created_at": time.monotonic(),
                    "context": {
                        "preferences": {},
                        "matched_foods": []
                    }
                }

        return customer_id

    def _get_live_chat(self, customer_id):
        """
        Return the customer's chat object, evicting and recreating the
        session if it has expired. Caller must hold self._lock.
        Recreated sessions start with empty context (fresh conversation).

        NEW customers are capacity-gated here (this is the lazy session
        creation point of the production send path). Existing customers
        - including those whose session has expired - always pass and
        recreate their own session, even when capacity is full.
        """
        self._gate_new_customer_locked(customer_id)

        session = self.sessions.get(customer_id)

        if session is None or self._is_expired(session):
            self.sessions[customer_id] = {
                "chat": self.chat_factory(self.client),
                "created_at": time.monotonic(),
                "context": {
                    "preferences": {},
                    "matched_foods": []
                }
            }
            session = self.sessions[customer_id]

        return session["chat"]

    def send_customer_message(self, customer_id, message):
        """
        Send a message within the customer's own conversation context.

        Same-customer calls (send / end / create) are serialized through
        the customer's persistent lock; different customers run in
        parallel. Expired sessions are silently recreated (fresh
        conversation). NEW customers are capacity-gated BEFORE the
        customer lock is created, so an ordinary capacity rejection
        leaves no lock and no session.
        """
        self._reject_new_customer_if_capacity_reached(customer_id)

        customer_lock = self._get_customer_lock(customer_id)

        with customer_lock:
            with self._lock:
                chat = self._get_live_chat(customer_id)

            return send_message(chat, message)

    def get_context(self, customer_id):
        """
        Return the customer's stored conversation context:
            {"preferences": {...}, "matched_foods": [...]}
        Always this exact shape; empty values when the customer has no
        live session (missing or expired).

        Read-only: never creates a session, never resets expiry, and
        never creates a customer lock.

        Thread safety (Phase 6 Step 3): every context mutation happens
        under the global lock (create_session, _get_live_chat
        recreate, update_context, end_session and the expired-session
        sweep all replace or delete whole entries while holding it),
        so reading under the global lock alone yields an atomic,
        torn-read-free snapshot. The customer lock is not needed for a
        read, and omitting it keeps the read path free of lock
        creation for any customer.
        """
        empty_context = {"preferences": {}, "matched_foods": []}

        with self._lock:
            session = self.sessions.get(customer_id)

            if session is None or self._is_expired(session):
                return dict(empty_context)

            context = session.get("context")

            if context is None:
                return dict(empty_context)

            return {
                "preferences": dict(context.get("preferences") or {}),
                "matched_foods": list(
                    context.get("matched_foods") or []
                ),
            }

    def update_context(self, customer_id, context):
        """
        Store conversation context on the customer's live session.

        The matched_foods list is capped at MAX_CONTEXT_MATCHED_FOODS
        entries, keeping the MOST RECENT ones (older overflow is
        dropped, newest recommendations are always retained). Expired
        sessions are NOT resurrected: if the session is missing or
        expired the update is dropped.

        Returns:
            True  - context stored.
            False - no live session (dropped; nothing to store on).
        """
        preferences = dict(context.get("preferences") or {})
        matched_foods = list(context.get("matched_foods") or [])
        # Keep the MOST RECENT entries: the chat pipeline appends newly
        # recommended items to the end, so slicing from the end retains
        # the latest recommendations and drops the oldest overflow.
        matched_foods = matched_foods[-MAX_CONTEXT_MATCHED_FOODS:]

        stored = {
            "preferences": preferences,
            "matched_foods": matched_foods
        }

        customer_lock = self._get_customer_lock(customer_id)

        with customer_lock:
            with self._lock:
                session = self.sessions.get(customer_id)

                if session is None or self._is_expired(session):
                    return False

                session["context"] = stored

        return True

    def get_menu_state(self, customer_id):
        """
        Return the customer's stored restaurant-menu selection state
        (Phase 7 Step 4), or None when there is no live session / no
        stored menu.

        Shape (when set by set_menu_state):
            {"restaurant_id", "restaurant_name", "items": [...],
             "pending_dish": dict|None, "confirmed_dish": dict|None}

        Read-only: never creates a session, never resets expiry, never
        creates a customer lock. Returns a deep copy so callers cannot
        mutate stored state.
        """
        with self._lock:
            session = self.sessions.get(customer_id)

            if session is None or self._is_expired(session):
                return None

            state = session.get("menu")

            if not isinstance(state, dict):
                return None

            return copy.deepcopy(state)

    def set_menu_state(self, customer_id, menu_state):
        """
        Store restaurant-menu selection state on the customer's session
        (Phase 7 Step 4).

        - Creates a live session when the customer does not have one
          (same capacity gate as create_session). An EXISTING live
          session keeps its food context and chat history: only the
          session["menu"] entry is written.
        - Expired sessions are recreated fresh (empty food context and
          empty menu) then the new state is stored.
        - The stored state is deep-copied so later caller mutations
          cannot affect the session.

        Returns:
            True  - state stored.
            False - never; raises SessionCapacityError when a NEW
                    customer cannot be admitted at full capacity.
        """
        if not isinstance(menu_state, dict):
            raise ValueError("menu_state must be a dict")

        self._reject_new_customer_if_capacity_reached(customer_id)

        customer_lock = self._get_customer_lock(customer_id)

        with customer_lock:
            with self._lock:
                self._gate_new_customer_locked(customer_id)

                session = self.sessions.get(customer_id)

                if session is None or self._is_expired(session):
                    session = {
                        "chat": self.chat_factory(self.client),
                        "created_at": time.monotonic(),
                        "context": {
                            "preferences": {},
                            "matched_foods": []
                        },
                        "menu": None,
                    }
                    self.sessions[customer_id] = session

                session["menu"] = copy.deepcopy(menu_state)

        return True

    def get_cart(self, customer_id):
        """
        Return the customer's CartStore instance (Phase 7 Step 6.2),
        or None when the session is missing or expired, or when no
        cart has been created yet for a live session.

        Read-only: never creates a session, never creates a customer
        lock, never resets expiry and never creates a cart - the same
        contract as get_menu_state(). The store object itself is
        returned (not a copy): callers mutate it only through its own
        validated, thread-safe add_item/get_cart/remove_item/
        clear_cart methods.
        """
        with self._lock:
            session = self.sessions.get(customer_id)

            if session is None or self._is_expired(session):
                return None

            cart = session.get("cart")

            if cart is None:
                return None

            return cart

    def ensure_cart(self, customer_id):
        """
        Return the customer's CartStore, creating the session and/or
        the cart when needed (Phase 7 Step 6.2).

        Follows the set_menu_state() lifecycle exactly:
            - NEW customers are capacity-gated BEFORE any lock is
              created (SessionCapacityError at full MAX_SESSIONS).
            - Coordinates on the customer's persistent lock, then the
              global lock (same ordering as every other writer).
            - A missing or expired session is recreated fresh (empty
              food context, no menu): the expired session's cart is
              discarded with it, so an expired cart can never be
              resurrected.
            - At most ONE CartStore exists per session (lazily created
              through cart_store_factory; repeated calls return the
              same instance).

        Raises SessionCapacityError when a NEW customer cannot be
        admitted at full capacity.
        """
        self._reject_new_customer_if_capacity_reached(customer_id)

        customer_lock = self._get_customer_lock(customer_id)

        with customer_lock:
            with self._lock:
                self._gate_new_customer_locked(customer_id)

                session = self.sessions.get(customer_id)

                if session is None or self._is_expired(session):
                    session = {
                        "chat": self.chat_factory(self.client),
                        "created_at": time.monotonic(),
                        "context": {
                            "preferences": {},
                            "matched_foods": []
                        },
                        "menu": None,
                    }
                    self.sessions[customer_id] = session

                cart = session.get("cart")

                if cart is None:
                    cart = self.cart_store_factory()
                    session["cart"] = cart

                return cart

    def end_session(self, customer_id):
        """
        Delete the customer's conversation session.

        Coordinates on the customer's persistent lock, so an in-flight
        message completes first. The lock itself is kept (never popped)
        to prevent the deletion race.

        Raises:
            SessionNotFoundError: if the customer has no active session.
        """
        customer_lock = self._get_customer_lock(customer_id)

        with customer_lock:
            with self._lock:
                if customer_id not in self.sessions:
                    raise SessionNotFoundError(
                        f"No active session for customer '{customer_id}'."
                    )

                del self.sessions[customer_id]

    def active_session_count(self):
        """
        Number of sessions currently tracked (including expired ones
        that have not been touched yet - they are evicted lazily).
        """
        with self._lock:
            return len(self.sessions)
