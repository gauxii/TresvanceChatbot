"""Issue 6: concurrent/duplicate requests to the same session must not
interleave state transitions or create duplicate leads; independent sessions
must still process concurrently."""
import concurrent.futures
import json
import uuid


def _read_leads(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_concurrent_duplicate_name_replies_do_not_corrupt_stage(client, session_id):
    """Regression for the originally demonstrated race: firing the same
    'name' reply concurrently used to split across stages and produce
    contradictory replies (one accepted as name, others rejected as bad
    contact info) because of an unprotected read-modify-write on session
    state. With per-session locking, every reply must be processed against a
    consistent, up-to-date stage — so every one of 6 duplicate submissions
    should get the SAME kind of response."""
    client.post("/api/chat", json={"message": "I want a quote", "session_id": session_id})

    def send(i):
        resp = client.post("/api/chat", json={"message": f"RaceName{i}", "session_id": session_id})
        return resp.json()["answer"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(send, range(6)))

    # Exactly one reply should be the "thanks, <name>, what's your email..."
    # acceptance; the rest — since the lock serializes them AFTER the first
    # one already advanced the stage to "contact" — should consistently be
    # "that doesn't look like a valid email/phone" (NOT a mix of unrelated
    # or contradictory states).
    accepted = [r for r in results if r.lower().startswith("thanks,")]
    rejected_as_contact = [r for r in results if "valid email" in r.lower()]
    assert len(accepted) == 1
    assert len(rejected_as_contact) == 5
    assert len(accepted) + len(rejected_as_contact) == 6


def test_concurrent_submissions_create_at_most_one_lead(client, session_id, isolated_leads_file):
    """No duplicate lead should be created from concurrent/duplicate
    submissions of the final 'project' answer."""
    client.post("/api/chat", json={"message": "I want a quote", "session_id": session_id})
    client.post("/api/chat", json={"message": "Chris", "session_id": session_id})
    client.post("/api/chat", json={"message": "chris@example.com", "session_id": session_id})

    def send(i):
        return client.post("/api/chat", json={"message": "a website", "session_id": session_id}).json()["answer"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(send, range(5)))

    leads = _read_leads(isolated_leads_file)
    assert len(leads) == 1


def test_independent_sessions_process_concurrently_without_blocking_each_other(client):
    """Per-session locks must not become a de facto global lock."""
    session_ids = [str(uuid.uuid4()) for _ in range(8)]

    def send(sid):
        return client.post("/api/chat", json={"message": "book a call", "session_id": sid}).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(send, session_ids))

    assert all(status == 200 for status in results)


def test_session_history_is_not_corrupted_by_concurrent_requests(client, session_id):
    """Every user message sent should end up paired with exactly one
    assistant reply in history — no dropped or duplicated turns."""
    client.post("/api/chat", json={"message": "I want a quote", "session_id": session_id})

    def send(i):
        return client.post("/api/chat", json={"message": f"Name{i}", "session_id": session_id})

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(send, range(4)))

    import conversation

    history = conversation.get_session(session_id)["history"]
    user_turns = [m for m in history if m["role"] == "user"]
    assistant_turns = [m for m in history if m["role"] == "assistant"]
    assert len(user_turns) == len(assistant_turns)
