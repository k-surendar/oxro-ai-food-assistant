"""
Session manager tests (Phase 2 requirements):
    - per-customer conversation isolation
    - session expiration (TTL)
    - session deletion
    - TTL from the SESSION_TTL_SECONDS environment variable

All tests use a fake chat factory, so no real Gemini connection or API key
is needed.
"""

import threading
import time

import pytest

from ai.cart_store import CartStore
from ai.session_manager import (
    SessionCapacityError,
    SessionManager,
    SessionNotFoundError,
)


class FakeResponse:
    """Mimics the Gemini SDK response object (has .text)."""

    def __init__(self, text):
        self.text = text


class FakeChat:
    """Stands in for a Gemini chat object; records its own history."""

    def __init__(self, client=None):
        self.history = []

    def send_message(self, message):
        self.history.append(message)
        return FakeResponse(f"echo:{message}")


# ---------- FakeChat contract (review point 7) ----------

def test_fake_chat_response_mimics_gemini_text_attribute():
    chat = FakeChat()

    response = chat.send_message("hello")

    assert hasattr(response, "text")
    assert response.text == "echo:hello"


def make_manager(**kwargs):
    return SessionManager(
        client=None,
        chat_factory=lambda client: FakeChat(),
        **kwargs
    )


# ---------------- isolation ----------------

def test_sessions_are_isolated_between_customers():
    sm = make_manager()

    sm.create_session("CUST_A")
    sm.create_session("CUST_B")

    sm.send_customer_message("CUST_A", "I want chicken biryani")
    sm.send_customer_message("CUST_B", "I want veg pizza")
    sm.send_customer_message("CUST_A", "make it spicy")

    chat_a = sm.sessions["CUST_A"]["chat"]
    chat_b = sm.sessions["CUST_B"]["chat"]

    assert chat_a is not chat_b
    assert chat_a.history == ["I want chicken biryani", "make it spicy"]
    assert chat_b.history == ["I want veg pizza"]


def test_same_customer_reuses_same_chat():
    sm = make_manager()

    sm.create_session("CUST_A")
    first_chat = sm.sessions["CUST_A"]["chat"]

    sm.send_customer_message("CUST_A", "hello")
    second_chat = sm.sessions["CUST_A"]["chat"]

    assert first_chat is second_chat
    assert first_chat.history == ["hello"]


# ---------------- expiration (TTL) ----------------

def test_expired_session_is_recreated():
    sm = make_manager(session_ttl_seconds=60)

    sm.create_session("CUST_A")
    old_chat = sm.sessions["CUST_A"]["chat"]
    old_chat.history.append("old message")

    # Force the session to look expired without waiting.
    sm.sessions["CUST_A"]["created_at"] -= 61

    sm.send_customer_message("CUST_A", "new message")

    new_chat = sm.sessions["CUST_A"]["chat"]

    assert new_chat is not old_chat
    assert new_chat.history == ["new message"]


def test_ttl_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("SESSION_TTL_SECONDS", "120")

    sm = SessionManager(client=None, chat_factory=lambda c: FakeChat())

    assert sm.session_ttl_seconds == 120


def test_ttl_defaults_to_3600(monkeypatch):
    monkeypatch.delenv("SESSION_TTL_SECONDS", raising=False)

    sm = SessionManager(client=None, chat_factory=lambda c: FakeChat())

    assert sm.session_ttl_seconds == 3600


def test_invalid_ttl_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SESSION_TTL_SECONDS", "not-a-number")

    sm = SessionManager(client=None, chat_factory=lambda c: FakeChat())

    assert sm.session_ttl_seconds == 3600


# ---------------- concurrency (review point 3) ----------------

def test_same_customer_messages_are_serialized(monkeypatch):
    """
    Two concurrent sends from the SAME customer must never overlap and
    must be delivered in call order. Without the per-customer lock both
    fake sends would run simultaneously and the overlap flag would flip.
    """
    import ai.session_manager as sm_module

    sm = make_manager()
    sm.create_session("CUST_A")

    order = []
    state = {"inside": 0, "overlap": False}
    first_started = threading.Event()

    def fake_send(chat, message):
        state["inside"] += 1
        if state["inside"] > 1:
            state["overlap"] = True
        if message == "first":
            first_started.set()
            time.sleep(0.1)
        order.append(message)
        state["inside"] -= 1
        return FakeResponse(f"echo:{message}")

    monkeypatch.setattr(sm_module, "send_message", fake_send)

    sender = threading.Thread(
        target=sm.send_customer_message,
        args=("CUST_A", "first")
    )
    sender.start()
    assert first_started.wait(timeout=2)

    sm.send_customer_message("CUST_A", "second")
    sender.join()

    assert state["overlap"] is False
    assert order == ["first", "second"]


def test_different_customers_are_not_serialized(monkeypatch):
    """
    Different customers must be able to send concurrently (no global
    serialization across customers).
    """
    import ai.session_manager as sm_module

    sm = make_manager()
    sm.create_session("CUST_A")
    sm.create_session("CUST_B")

    state = {"inside": 0, "overlap": False}
    entered = threading.Event()

    def fake_send(chat, message):
        state["inside"] += 1
        if state["inside"] > 1:
            state["overlap"] = True
        entered.set()
        time.sleep(0.05)
        state["inside"] -= 1
        return FakeResponse(f"echo:{message}")

    monkeypatch.setattr(sm_module, "send_message", fake_send)

    sender = threading.Thread(
        target=sm.send_customer_message,
        args=("CUST_B", "from B")
    )
    sender.start()
    assert entered.wait(timeout=2)

    sm.send_customer_message("CUST_A", "from A")
    sender.join()

    assert state["overlap"] is True


# ------------- deletion race / lock persistence (review point 1) -------------

def test_customer_lock_persists_after_end_session():
    """
    end_session must NOT pop the customer lock: a new lock created while
    an old operation still holds the old one would break same-customer
    serialization. The same lock object must be returned before and after
    deletion.
    """
    sm = make_manager()

    sm.create_session("CUST_A")
    lock_before = sm._get_customer_lock("CUST_A")

    sm.end_session("CUST_A")

    lock_after = sm._get_customer_lock("CUST_A")

    assert lock_before is lock_after


def test_end_session_waits_for_inflight_message(monkeypatch):
    """
    A concurrent end_session must wait until the in-flight message for
    the same customer has finished before deleting the session.
    """
    import ai.session_manager as sm_module

    sm = make_manager()
    sm.create_session("CUST_A")

    send_started = threading.Event()
    release_send = threading.Event()

    def fake_send(chat, message):
        send_started.set()
        release_send.wait(timeout=2)
        return FakeResponse("done")

    monkeypatch.setattr(sm_module, "send_message", fake_send)

    sender = threading.Thread(
        target=sm.send_customer_message,
        args=("CUST_A", "inflight")
    )
    sender.start()
    assert send_started.wait(timeout=2)

    end_error = []

    def do_end():
        try:
            sm.end_session("CUST_A")
        except Exception as exc:  # pragma: no cover - should not happen
            end_error.append(exc)

    ender = threading.Thread(target=do_end)
    ender.start()
    time.sleep(0.05)

    # While the message is in flight the session must still exist.
    assert "CUST_A" in sm.sessions

    release_send.set()
    sender.join(timeout=2)
    ender.join(timeout=2)

    assert end_error == []
    assert "CUST_A" not in sm.sessions


def test_send_end_create_send_lifecycle_is_ordered(monkeypatch):
    """
    Simultaneous message sending, session deletion and session recreation:
    with an in-flight send, a waiting end_session and a waiting
    create_session, everything must run through the customer's single
    lock in call order, and the recreated session must get a fresh chat.
    """
    import ai.session_manager as sm_module

    sm = make_manager()
    sm.create_session("CUST_A")
    old_chat = sm.sessions["CUST_A"]["chat"]

    first_started = threading.Event()
    release_first = threading.Event()
    events = []

    def fake_send(chat, message):
        if message == "first":
            first_started.set()
            release_first.wait(timeout=2)
        events.append(message)
        return FakeResponse("ok")

    monkeypatch.setattr(sm_module, "send_message", fake_send)

    sender = threading.Thread(
        target=sm.send_customer_message,
        args=("CUST_A", "first")
    )
    sender.start()
    assert first_started.wait(timeout=2)

    ended = threading.Event()

    def do_end():
        sm.end_session("CUST_A")
        ended.set()

    ender = threading.Thread(target=do_end)
    ender.start()
    time.sleep(0.05)

    # This create_session must block until the in-flight send AND the
    # waiting end_session have both completed (single customer lock).
    sm.create_session("CUST_A")
    sm.send_customer_message("CUST_A", "second")

    release_first.set()
    sender.join(timeout=2)
    ender.join(timeout=2)

    assert ended.is_set()
    assert events == ["first", "second"]

    new_chat = sm.sessions["CUST_A"]["chat"]
    assert new_chat is not old_chat


# ---------------- deletion ----------------

def test_end_session_deletes():
    sm = make_manager()

    sm.create_session("CUST_A")
    sm.end_session("CUST_A")

    assert "CUST_A" not in sm.sessions
    assert sm.active_session_count() == 0


def test_end_session_twice_raises():
    sm = make_manager()

    sm.create_session("CUST_A")
    sm.end_session("CUST_A")

    with pytest.raises(SessionNotFoundError):
        sm.end_session("CUST_A")


def test_end_session_unknown_customer_raises():
    sm = make_manager()

    with pytest.raises(SessionNotFoundError):
        sm.end_session("NOBODY")

# ================= Phase 4: conversation context =================

from ai.session_manager import MAX_CONTEXT_MATCHED_FOODS


def test_new_session_starts_with_empty_context():
    sm = make_manager()
    sm.create_session("C_CTX1")

    context = sm.get_context("C_CTX1")

    assert context == {"preferences": {}, "matched_foods": []}


def test_get_context_unknown_customer_returns_empty():
    sm = make_manager()

    # Contract: get_context ALWAYS returns the full context shape.
    assert sm.get_context("NOBODY") == {
        "preferences": {},
        "matched_foods": []
    }


def test_get_context_does_not_create_session():
    sm = make_manager()

    sm.get_context("NOBODY")

    assert "NOBODY" not in sm.sessions


def test_update_context_stores_preferences_and_matched_foods():
    sm = make_manager()
    sm.create_session("C_CTX2")

    stored = sm.update_context(
        "C_CTX2",
        {
            "preferences": {"food_type": "non-vegetarian", "taste": "spicy"},
            "matched_foods": ["Chicken Biryani", "Chicken 65"],
        }
    )

    assert stored is True
    assert sm.get_context("C_CTX2") == {
        "preferences": {"food_type": "non-vegetarian", "taste": "spicy"},
        "matched_foods": ["Chicken Biryani", "Chicken 65"],
    }


def test_context_is_isolated_between_customers():
    sm = make_manager()
    sm.create_session("C_A")
    sm.create_session("C_B")

    sm.update_context(
        "C_A",
        {"preferences": {"taste": "spicy"}, "matched_foods": ["Chicken Biryani"]}
    )

    assert sm.get_context("C_A")["matched_foods"] == ["Chicken Biryani"]
    assert sm.get_context("C_B") == {"preferences": {}, "matched_foods": []}


def test_update_context_missing_session_returns_false_and_drops():
    sm = make_manager()

    stored = sm.update_context(
        "NOBODY",
        {"preferences": {"taste": "spicy"}, "matched_foods": ["Chicken Biryani"]}
    )

    assert stored is False
    assert "NOBODY" not in sm.sessions


def test_expired_session_context_is_reset_not_resurrected():
    sm = make_manager(session_ttl_seconds=1)
    sm.create_session("C_EXPIRED")

    sm.update_context(
        "C_EXPIRED",
        {"preferences": {"taste": "spicy"}, "matched_foods": ["Chicken Biryani"]}
    )

    time.sleep(1.2)

    # Expiry alone: get_context reports empty (session is stale).
    assert sm.get_context("C_EXPIRED") == {"preferences": {}, "matched_foods": []}

    # update_context must NOT resurrect the expired session.
    stored = sm.update_context(
        "C_EXPIRED",
        {"preferences": {"taste": "mild"}, "matched_foods": ["Paneer Tikka"]}
    )
    assert stored is False


def test_end_session_clears_context():
    sm = make_manager()
    sm.create_session("C_ENDCTX")

    sm.update_context(
        "C_ENDCTX",
        {"preferences": {"taste": "spicy"}, "matched_foods": ["Chicken Biryani"]}
    )
    sm.end_session("C_ENDCTX")

    assert sm.get_context("C_ENDCTX") == {"preferences": {}, "matched_foods": []}


def test_matched_foods_list_is_capped():
    sm = make_manager()
    sm.create_session("C_CAP")

    many_foods = [f"Food {i}" for i in range(MAX_CONTEXT_MATCHED_FOODS + 10)]

    sm.update_context("C_CAP", {"preferences": {}, "matched_foods": many_foods})

    stored = sm.get_context("C_CAP")["matched_foods"]
    assert len(stored) == MAX_CONTEXT_MATCHED_FOODS
    # The MOST RECENT entries are retained (newest recommendations are
    # appended to the end of the list by the chat pipeline).
    assert stored == many_foods[-MAX_CONTEXT_MATCHED_FOODS:]


def test_update_context_waits_for_inflight_message(monkeypatch):
    """
    update_context coordinates on the same persistent per-customer lock
    as send_customer_message: while a message is in flight for the same
    customer, the context update must BLOCK until the message completes.
    Without the lock the update would complete immediately while the
    message is still being sent (overlapping), which this test detects.
    """
    import ai.session_manager as sm_module

    sm = make_manager()
    sm.create_session("C_SERIAL")

    send_started = threading.Event()
    release_send = threading.Event()
    send_finished = threading.Event()
    update_done = threading.Event()

    def fake_send(chat, message):
        send_started.set()
        release_send.wait(timeout=2)
        send_finished.set()
        return FakeResponse(f"echo:{message}")

    monkeypatch.setattr(sm_module, "send_message", fake_send)

    sender = threading.Thread(
        target=sm.send_customer_message,
        args=("C_SERIAL", "in-flight message")
    )
    sender.start()
    assert send_started.wait(timeout=2)

    updater = threading.Thread(
        target=lambda: (
            sm.update_context(
                "C_SERIAL",
                {"preferences": {"taste": "spicy"}, "matched_foods": ["X"]}
            ),
            update_done.set()
        )
    )
    updater.start()

    # The send is still in flight: the context update must be blocked,
    # NOT completed.
    time.sleep(0.15)
    assert not update_done.is_set()

    # Let the message finish: the blocked update must then complete.
    release_send.set()
    sender.join(timeout=2)
    updater.join(timeout=2)

    assert send_finished.is_set()
    assert update_done.is_set()
    assert sm.get_context("C_SERIAL") == {
        "preferences": {"taste": "spicy"},
        "matched_foods": ["X"]
    }


# ---------------- cart state (Phase 7 Step 6.2) ----------------

def _seeded_cart(sm, customer_id, qty=1):
    """ensure_cart + one add_item line, returning the store."""
    cart = sm.ensure_cart(customer_id)
    cart.add_item(customer_id, "d1", "Hyderabadi Chicken Biryani", 299, qty)
    return cart


def test_ensure_cart_creates_and_get_cart_returns_same_instance():
    sm = make_manager()

    cart = sm.ensure_cart("C_CART")

    assert isinstance(cart, CartStore)
    assert sm.get_cart("C_CART") is cart


def test_ensure_cart_is_idempotent_one_store_per_session():
    sm = make_manager()
    sm.create_session("C_CART")

    first = sm.ensure_cart("C_CART")
    second = sm.ensure_cart("C_CART")

    assert first is second


def test_ensure_cart_store_is_usable_with_full_validation():
    sm = make_manager()

    cart = sm.ensure_cart("C_CART")
    result = cart.add_item("C_CART", "d1", "Dish", 100, 2)

    assert result["ok"] is True
    assert sm.get_cart("C_CART").get_cart("C_CART")["cart"]["subtotal"] == 200


def test_carts_are_isolated_between_customers():
    sm = make_manager()

    cart_a = sm.ensure_cart("C_A")
    cart_b = sm.ensure_cart("C_B")

    assert cart_a is not cart_b

    cart_a.add_item("C_A", "d1", "Dish A", 100, 1)

    assert cart_a.get_cart("C_A")["cart"]["subtotal"] == 100
    assert cart_b.get_cart("C_B")["cart"]["items"] == []
    assert sm.get_cart("C_B") is cart_b


def test_get_cart_is_readonly_no_session_no_lock():
    sm = make_manager()

    assert sm.get_cart("NOBODY") is None
    assert "NOBODY" not in sm.sessions
    assert "NOBODY" not in sm._customer_locks


def test_get_cart_returns_none_for_live_session_without_cart():
    sm = make_manager()
    sm.create_session("C_NO_CART")

    assert sm.get_cart("C_NO_CART") is None


def test_get_cart_returns_none_for_unknown_customer():
    sm = make_manager()

    assert sm.get_cart("GHOST") is None


def test_cart_removed_after_end_session():
    sm = make_manager()
    old_cart = _seeded_cart(sm, "C_END")
    assert old_cart.get_cart("C_END")["cart"]["subtotal"] == 299

    sm.end_session("C_END")

    assert sm.get_cart("C_END") is None


def test_fresh_empty_cart_after_end_session_recreate():
    sm = make_manager()
    old_cart = _seeded_cart(sm, "C_END")
    sm.end_session("C_END")

    new_cart = sm.ensure_cart("C_END")

    assert new_cart is not old_cart
    assert new_cart.get_cart("C_END")["cart"]["items"] == []
    assert sm.get_cart("C_END") is new_cart


def test_cart_removed_after_ttl_expiry():
    sm = make_manager(session_ttl_seconds=60)
    _seeded_cart(sm, "C_TTL")

    # Force the session to look expired without waiting.
    sm.sessions["C_TTL"]["created_at"] -= 61

    # Expired: read returns None and does NOT resurrect the session.
    assert sm.get_cart("C_TTL") is None
    assert sm.sessions["C_TTL"]["cart"] is not None  # stale entry until touched


def test_fresh_empty_cart_after_session_recreation():
    sm = make_manager(session_ttl_seconds=60)
    old_cart = _seeded_cart(sm, "C_TTL")
    old_cart.add_item("C_TTL", "d2", "Second Dish", 50, 1)

    sm.sessions["C_TTL"]["created_at"] -= 61

    new_cart = sm.ensure_cart("C_TTL")  # recreates fresh

    assert new_cart is not old_cart
    assert new_cart.get_cart("C_TTL")["cart"]["items"] == []
    assert new_cart.get_cart("C_TTL")["cart"]["subtotal"] == 0
    assert sm.get_cart("C_TTL") is new_cart


def test_cart_discarded_on_lazy_recreate_by_send():
    sm = make_manager(session_ttl_seconds=60)
    _seeded_cart(sm, "C_LAZY")

    sm.sessions["C_LAZY"]["created_at"] -= 61
    sm.send_customer_message("C_LAZY", "hello")  # lazy recreation

    # New session exists but carries no cart yet.
    assert sm.get_cart("C_LAZY") is None


def test_ensure_cart_respects_max_sessions():
    sm = make_manager(max_sessions=2)
    sm.ensure_cart("C1")
    sm.ensure_cart("C2")

    with pytest.raises(SessionCapacityError):
        sm.ensure_cart("C3")

    # Ordinary capacity rejection creates neither a session nor a lock.
    assert "C3" not in sm.sessions
    assert "C3" not in sm._customer_locks


def test_existing_customer_can_ensure_cart_at_capacity():
    sm = make_manager(max_sessions=1)
    sm.ensure_cart("C1")

    assert sm.ensure_cart("C1") is not None

    with pytest.raises(SessionCapacityError):
        sm.ensure_cart("C2")


def test_capacity_sweep_drops_expired_customers_cart():
    sm = make_manager(max_sessions=2, session_ttl_seconds=60)
    _seeded_cart(sm, "C1")
    sm.ensure_cart("C2")

    # C1 expires; C2 stays live. A new customer triggers the sweep.
    sm.sessions["C1"]["created_at"] -= 61
    sm.ensure_cart("C3")

    assert "C1" not in sm.sessions
    assert sm.get_cart("C1") is None
    # C3 was admitted and has its own empty cart.
    assert sm.get_cart("C3").get_cart("C3")["cart"]["items"] == []


def test_menu_state_survives_ensure_cart():
    sm = make_manager()
    sm.create_session("C_MENU")
    sm.set_menu_state("C_MENU", {"items": [{"id": 1, "name": "Dish"}]})

    sm.ensure_cart("C_MENU")

    state = sm.get_menu_state("C_MENU")
    assert state["items"][0]["id"] == 1


def test_cart_survives_set_menu_state_on_live_session():
    sm = make_manager()
    cart = _seeded_cart(sm, "C_BOTH")

    sm.set_menu_state("C_BOTH", {"items": [{"id": 1, "name": "Dish"}]})

    assert sm.get_cart("C_BOTH") is cart
    assert cart.get_cart("C_BOTH")["cart"]["subtotal"] == 299


def test_set_menu_state_recreation_discards_expired_cart():
    sm = make_manager(session_ttl_seconds=60)
    _seeded_cart(sm, "C_X")
    sm.sessions["C_X"]["created_at"] -= 61

    sm.set_menu_state("C_X", {"items": []})  # recreates expired session

    assert sm.get_cart("C_X") is None


def test_injected_cart_store_factory_is_used():
    created = []

    class Marker:
        pass

    def factory():
        marker = Marker()
        created.append(marker)
        return marker

    sm = SessionManager(
        client=None,
        chat_factory=lambda client: FakeChat(),
        cart_store_factory=factory,
    )

    cart = sm.ensure_cart("C_FAKE")

    assert isinstance(cart, Marker)
    assert sm.get_cart("C_FAKE") is cart
    assert len(created) == 1

    # Same session: no second store is created.
    sm.ensure_cart("C_FAKE")
    assert len(created) == 1

    # New session (after end) creates exactly one more.
    sm.end_session("C_FAKE")
    sm.ensure_cart("C_FAKE")
    assert len(created) == 2


def test_default_factory_produces_real_cart_store():
    sm = make_manager()

    assert sm.cart_store_factory is CartStore
    assert isinstance(sm.ensure_cart("C_DEFAULT"), CartStore)
