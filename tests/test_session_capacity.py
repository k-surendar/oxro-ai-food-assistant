"""
Phase 6 Step 3 tests: MAX_SESSIONS capacity gate, safe expired-session
sweep, lock-creation guarantees and concurrency at the capacity
boundary.

Covers:
    - MAX_SESSIONS configuration (default 500, env override, invalid
      fallback, injectable parameter).
    - New customers rejected at capacity WITHOUT creating a session or
      a customer lock (ordinary rejection path).
    - Existing customers keep working at capacity: send, refinement,
      update_context, end_session and their own expired-session
      recreation / reset.
    - Expired sessions free capacity (sweep at capacity) but are never
      removed while their operation is in flight (customer lock held).
    - Atomicity: concurrent new-customer creation at the boundary can
      never exceed MAX_SESSIONS.
    - get_context creates no customer lock and is torn-read-free.
    - API level: static 503 session_capacity_reached, greetings and
      stateless /recommend-food unaffected at capacity.

All Gemini interaction is faked - no network, no API keys.
"""

import threading

import pytest
from fastapi.testclient import TestClient

import main as main_module
from ai.session_manager import (
    DEFAULT_MAX_SESSIONS,
    SessionCapacityError,
    SessionManager,
    SessionNotFoundError,
)


API_KEY = "test-sandbox-key"  # matches conftest default
H = {"X-API-Key": API_KEY}


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeChat:
    def __init__(self, client=None):
        self.history = []

    def send_message(self, message):
        self.history.append(message)
        return FakeResponse(f"echo:{message}")


def make_manager(**kwargs):
    return SessionManager(
        client=None,
        chat_factory=lambda client: FakeChat(),
        **kwargs
    )


# ---------------------------------------------------------------------------
# MAX_SESSIONS configuration
# ---------------------------------------------------------------------------

def test_max_sessions_defaults_to_500(monkeypatch):
    monkeypatch.delenv("MAX_SESSIONS", raising=False)

    sm = make_manager()

    assert sm.max_sessions == DEFAULT_MAX_SESSIONS == 500


def test_max_sessions_env_override(monkeypatch):
    monkeypatch.setenv("MAX_SESSIONS", "250")

    sm = SessionManager(client=None, chat_factory=lambda c: FakeChat())

    assert sm.max_sessions == 250


@pytest.mark.parametrize("raw", ["abc", "-5", "0", " "])
def test_max_sessions_invalid_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setenv("MAX_SESSIONS", raw)

    sm = SessionManager(client=None, chat_factory=lambda c: FakeChat())

    assert sm.max_sessions == DEFAULT_MAX_SESSIONS


def test_max_sessions_injectable_parameter():
    assert make_manager(max_sessions=3).max_sessions == 3


# ---------------------------------------------------------------------------
# Rejection below/at capacity
# ---------------------------------------------------------------------------

def test_creation_below_capacity_works():
    sm = make_manager(max_sessions=3)

    sm.create_session("C1")
    sm.create_session("C2")

    sm.send_customer_message("C1", "I want chicken biryani")
    sm.send_customer_message("C2", "I want veg pizza")

    assert sm.sessions["C1"]["chat"].history == ["I want chicken biryani"]
    assert sm.sessions["C2"]["chat"].history == ["I want veg pizza"]


def test_new_customer_rejected_at_capacity_without_session_or_lock():
    sm = make_manager(max_sessions=2)

    sm.create_session("C1")
    sm.create_session("C2")

    with pytest.raises(SessionCapacityError):
        sm.create_session("C3")

    assert "C3" not in sm.sessions
    assert "C3" not in sm._customer_locks


def test_new_customer_send_rejected_at_capacity_without_session_or_lock():
    sm = make_manager(max_sessions=2)

    sm.create_session("C1")
    sm.create_session("C2")

    with pytest.raises(SessionCapacityError):
        sm.send_customer_message("C_NEW", "I want biryani")

    assert "C_NEW" not in sm.sessions
    assert "C_NEW" not in sm._customer_locks


def test_rejected_customer_get_context_is_empty_and_creates_no_lock():
    sm = make_manager(max_sessions=1)

    sm.create_session("C1")

    with pytest.raises(SessionCapacityError):
        sm.create_session("C_FULL")

    assert sm.get_context("C_FULL") == {
        "preferences": {},
        "matched_foods": []
    }
    assert "C_FULL" not in sm._customer_locks


# ---------------------------------------------------------------------------
# Existing customers at capacity
# ---------------------------------------------------------------------------

def test_existing_customer_send_update_end_still_work_at_capacity():
    sm = make_manager(max_sessions=2)

    sm.create_session("C1")
    sm.create_session("C2")
    sm.send_customer_message("C1", "I want chicken biryani")

    # Capacity is full: a new customer is rejected...
    with pytest.raises(SessionCapacityError):
        sm.create_session("C3")

    # ...but C1 keeps working end to end.
    sm.send_customer_message("C1", "make it spicy")

    chat = sm.sessions["C1"]["chat"]
    assert chat.history == [
        "I want chicken biryani",
        "make it spicy"
    ]

    assert sm.update_context(
        "C1",
        {"preferences": {"food_type": "non-vegetarian"},
         "matched_foods": []}
    ) is True

    sm.end_session("C1")
    assert "C1" not in sm.sessions


def test_expired_existing_customer_recreates_own_session_at_capacity():
    sm = make_manager(max_sessions=2, session_ttl_seconds=60)

    sm.create_session("C1")
    sm.create_session("C2")  # capacity full

    # Force C1's session to look expired (its slot is still there).
    sm.sessions["C1"]["created_at"] -= 61

    sm.send_customer_message("C1", "fresh message")

    assert sm.sessions["C1"]["chat"].history == ["fresh message"]
    assert len(sm.sessions) == 2  # replaced own slot, no growth


def test_existing_customer_reset_at_capacity():
    sm = make_manager(max_sessions=2)

    sm.create_session("C1")
    sm.create_session("C2")  # capacity full

    # Resetting an own existing session is allowed at full capacity.
    sm.create_session("C1")

    assert "C1" in sm.sessions
    assert len(sm.sessions) == 2


# ---------------------------------------------------------------------------
# Expiry frees capacity (sweep happens only at capacity)
# ---------------------------------------------------------------------------

def test_expired_sessions_free_capacity_when_full():
    sm = make_manager(max_sessions=2, session_ttl_seconds=60)

    sm.create_session("C1")
    sm.create_session("C2")

    sm.sessions["C1"]["created_at"] -= 61
    sm.sessions["C2"]["created_at"] -= 61

    # Capacity was full: the gate sweeps the expired sessions first,
    # so C3 fits.
    sm.create_session("C3")

    assert "C3" in sm.sessions
    assert "C1" not in sm.sessions
    assert "C2" not in sm.sessions


def test_partial_expiry_frees_capacity_and_keeps_live_sessions():
    sm = make_manager(max_sessions=2, session_ttl_seconds=60)

    sm.create_session("C1")
    sm.create_session("C2")
    sm.sessions["C1"]["created_at"] -= 61  # only C1 expired

    # Capacity is reached: the gate sweeps ONLY the expired C1; the
    # live C2 is kept, and C3 is admitted into the freed slot.
    sm.create_session("C3")

    assert "C1" not in sm.sessions
    assert "C2" in sm.sessions
    assert "C3" in sm.sessions


def test_no_sweep_when_below_capacity():
    sm = make_manager(max_sessions=10, session_ttl_seconds=60)

    sm.create_session("C1")
    sm.sessions["C1"]["created_at"] -= 61  # expired, but capacity free

    # A new-customer gate below capacity must NOT sweep (decision:
    # sweep only when capacity is reached).
    with sm._lock:
        sm._gate_new_customer_locked("C_NEW")

    assert "C1" in sm.sessions


# ---------------------------------------------------------------------------
# Sweep safety: in-flight operations are protected
# ---------------------------------------------------------------------------

def test_sweep_never_removes_session_with_held_customer_lock():
    sm = make_manager(max_sessions=1, session_ttl_seconds=60)

    sm.create_session("C1")
    sm.sessions["C1"]["created_at"] -= 61  # expired zombie

    lock_acquired = threading.Event()
    release_lock = threading.Event()
    result = {}

    def holder():
        customer_lock = sm._get_customer_lock("C1")
        with customer_lock:
            lock_acquired.set()
            release_lock.wait()
            result["saw_session"] = "C1" in sm.sessions

    thread = threading.Thread(target=holder)
    thread.start()
    lock_acquired.wait()

    # Full capacity: the sweep runs, but C1's customer lock is held,
    # so the expired session must survive.
    with pytest.raises(SessionCapacityError):
        with sm._lock:
            sm._gate_new_customer_locked("C_NEW")

    assert "C1" in sm.sessions

    release_lock.set()
    thread.join(timeout=5)

    assert result["saw_session"] is True  # never deleted mid-operation

    # Next capacity event sweeps the now-unheld expired session.
    with sm._lock:
        sm._gate_new_customer_locked("C_NEW")

    assert "C1" not in sm.sessions


def test_inflight_send_completes_when_capacity_is_reached():
    send_started = threading.Event()
    release_send = threading.Event()
    result = {}

    class SlowChat:
        def __init__(self, client=None):
            self.history = []

        def send_message(self, message):
            send_started.set()
            release_send.wait()
            self.history.append(message)
            return FakeResponse("reply-during-capacity")

    # The LIVE session is backed by SlowChat so the send can be held
    # in flight from the test.
    sm = SessionManager(
        client=None,
        chat_factory=lambda client: SlowChat(),
        max_sessions=1,
        session_ttl_seconds=60
    )

    sm.create_session("C1")

    def sender():
        result["reply"] = sm.send_customer_message("C1", "in-flight")

    thread = threading.Thread(target=sender)
    thread.start()
    send_started.wait()

    # The TTL passes WHILE the send is in flight (simulated by forcing
    # the timestamp backwards here): the send holds C1's customer lock
    # the whole time.
    sm.sessions["C1"]["created_at"] -= 61

    # Concurrent capacity gate for a new customer: sweep must skip the
    # now-expired C1 because its customer lock is held by the
    # in-flight send, and the gate then rejects the new customer.
    with pytest.raises(SessionCapacityError):
        with sm._lock:
            sm._gate_new_customer_locked("C_NEW")

    release_send.set()
    thread.join(timeout=5)

    assert result["reply"] == "reply-during-capacity"
    assert "C1" in sm.sessions  # not swept while in flight
    assert sm.sessions["C1"]["chat"].history == ["in-flight"]


def test_update_context_after_external_sweep_is_dropped_not_crash():
    sm = make_manager(max_sessions=1, session_ttl_seconds=60)

    sm.create_session("C1")
    sm.sessions["C1"]["created_at"] -= 61

    # Someone else's capacity gate sweeps the (unheld) expired session.
    with sm._lock:
        sm._gate_new_customer_locked("C_NEW")

    assert "C1" not in sm.sessions

    # Existing semantics: update on a removed session is dropped.
    assert sm.update_context("C1", {"preferences": {}, "matched_foods": []}) is False


# ---------------------------------------------------------------------------
# Atomicity / concurrency at the boundary
# ---------------------------------------------------------------------------

def test_concurrent_new_customers_never_exceed_capacity():
    sm = make_manager(max_sessions=2)

    customer_ids = [f"C_T{i}" for i in range(10)]
    barrier = threading.Barrier(len(customer_ids))
    outcomes = []
    outcomes_lock = threading.Lock()

    def worker(customer_id):
        barrier.wait()
        try:
            sm.create_session(customer_id)
            outcome = "ok"
        except SessionCapacityError:
            outcome = "capacity"
        except Exception as exc:  # anything else is a bug
            outcome = f"bug:{exc}"

        with outcomes_lock:
            outcomes.append(outcome)

    threads = [
        threading.Thread(target=worker, args=(cid,))
        for cid in customer_ids
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert sorted(outcomes) == ["capacity"] * 8 + ["ok"] * 2
    assert len(sm.sessions) == 2


def test_mixed_operations_at_capacity_finish_without_deadlock():
    sm = make_manager(max_sessions=2)

    sm.create_session("C1")
    sm.create_session("C2")

    def do_sends(customer_id):
        for _ in range(15):
            try:
                sm.send_customer_message(customer_id, "hello")
            except SessionCapacityError:
                pass

    def do_resets():
        for _ in range(15):
            try:
                sm.create_session("C1")
            except SessionCapacityError:
                pass

    def do_ends():
        for _ in range(15):
            try:
                sm.end_session("C1")
            except SessionNotFoundError:
                pass

    def do_reads():
        for _ in range(15):
            sm.get_context("C2")

    threads = [
        threading.Thread(target=do_sends, args=("C1",)),
        threading.Thread(target=do_sends, args=("C2",)),
        threading.Thread(target=do_resets),
        threading.Thread(target=do_ends),
        threading.Thread(target=do_reads),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive(), "operation deadlocked"

    assert len(sm.sessions) <= 2


# ---------------------------------------------------------------------------
# get_context: lock-free, torn-read-free
# ---------------------------------------------------------------------------

def test_get_context_creates_no_customer_lock():
    sm = make_manager()

    sm.get_context("NOBODY")

    assert "NOBODY" not in sm._customer_locks
    assert sm.get_context("NOBODY") == {
        "preferences": {},
        "matched_foods": []
    }


def test_get_context_reads_are_torn_read_free():
    sm = make_manager()

    sm.create_session("C1")

    snapshots = [
        {"preferences": {}, "matched_foods": ["a"]},
        {"preferences": {}, "matched_foods": ["a", "b"]},
        {"preferences": {}, "matched_foods": ["a", "b", "c"]},
    ]

    bad = []
    stop = threading.Event()

    def writer():
        for i in range(300):
            sm.update_context("C1", snapshots[i % 3])

    def reader():
        while not stop.is_set():
            foods = sm.get_context("C1")["matched_foods"]
            if foods not in ([], ["a"], ["a", "b"], ["a", "b", "c"]):
                bad.append(foods)

    reader_thread = threading.Thread(target=reader)
    writer_thread = threading.Thread(target=writer)

    reader_thread.start()
    writer_thread.start()
    writer_thread.join(timeout=10)
    stop.set()
    reader_thread.join(timeout=10)

    assert bad == []


# ---------------------------------------------------------------------------
# API level
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """
    Runs the real lifespan (dummy keys from conftest), then swaps the
    SessionManager for an offline fake with capacity 1.
    """
    with TestClient(main_module.app) as test_client:
        main_module.app.state.session_manager = SessionManager(
            None,
            chat_factory=lambda cl: FakeChat(),
            max_sessions=1
        )
        yield test_client


def test_api_new_customer_rejected_503_at_capacity(client):
    first = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "I want chicken biryani"}
    )
    assert first.status_code == 200

    second = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C2", "message": "I want biryani"}
    )

    assert second.status_code == 503
    assert second.json() == {
        "detail": {
            "status": "session_capacity_reached",
            "reason": "The service is temporarily at session capacity. Please try again later."
        }
    }


def test_api_existing_customer_still_works_at_capacity(client):
    client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "I want chicken biryani"}
    )

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "Make it spicy."}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_api_greeting_works_at_capacity(client):
    client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "I want chicken biryani"}
    )

    response = client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C9", "message": "hello"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "greeting"


def test_api_recommend_food_works_at_capacity(client, monkeypatch):
    # /recommend-food is stateless (no session manager usage), so it
    # must be unaffected by the capacity gate.
    monkeypatch.setattr(
        main_module,
        "recommend_food_with_ai",
        lambda message, client_arg, max_retries=3: {
            "status": "success",
            "preferences": {},
            "food_results": ["Chicken Biryani"],
            "ai_response": "Here are some options.",
        },
    )

    client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "I want chicken biryani"}
    )

    response = client.post(
        "/recommend-food",
        headers=H,
        json={"customer_id": "C9", "message": "I want biryani"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_api_end_session_404_unaffected_at_capacity(client):
    client.post(
        "/chat",
        headers=H,
        json={"customer_id": "C1", "message": "I want chicken biryani"}
    )

    response = client.post("/session/UNKNOWN/end", headers=H)

    assert response.status_code == 404
    assert response.json()["detail"]["status"] == "session_not_found"
