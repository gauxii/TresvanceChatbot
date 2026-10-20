"""Issue 3 (repeat lead capture per session) and Issue 7 (empty-name
validation). The lead flow is a pure state machine (conversation.py) — it
never calls Ollama/Chroma, so these tests don't need mock_llm_success."""
import json

import crm


def _complete_lead(client, session_id, name="Gowri", contact="gowri@example.com", project="A booking site"):
    client.post("/api/chat", json={"message": "I want to book a call", "session_id": session_id})
    client.post("/api/chat", json={"message": name, "session_id": session_id})
    client.post("/api/chat", json={"message": contact, "session_id": session_id})
    return client.post("/api/chat", json={"message": project, "session_id": session_id})


def _read_leads(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_first_enquiry_is_captured(client, session_id, isolated_leads_file):
    resp = _complete_lead(client, session_id)
    assert resp.status_code == 200
    leads = _read_leads(isolated_leads_file)
    assert len(leads) == 1
    assert leads[0]["name"] == "Gowri"
    assert leads[0]["contact"] == "gowri@example.com"
    assert leads[0]["project"] == "A booking site"
    assert leads[0]["session_id"] == session_id
    assert leads[0]["lead_id"]  # non-empty unique id assigned


def test_second_enquiry_same_session_is_captured_as_a_new_lead(client, session_id, isolated_leads_file):
    _complete_lead(client, session_id, project="A booking site")

    # Returning visitor, same browser/session, new enquiry:
    resp = client.post("/api/chat", json={"message": "I'd like a quote for a second project", "session_id": session_id})
    assert resp.status_code == 200
    assert "name" in resp.json()["answer"].lower()  # re-prompts for details, doesn't just ack

    client.post("/api/chat", json={"message": "Gowri", "session_id": session_id})
    client.post("/api/chat", json={"message": "gowri@example.com", "session_id": session_id})
    client.post("/api/chat", json={"message": "A loyalty app", "session_id": session_id})

    leads = _read_leads(isolated_leads_file)
    assert len(leads) == 2
    projects = {lead["project"] for lead in leads}
    assert projects == {"A booking site", "A loyalty app"}
    # distinct lead_ids — not the same record being overwritten
    assert leads[0]["lead_id"] != leads[1]["lead_id"]


def test_previous_lead_preserved_when_new_enquiry_starts(client, session_id, isolated_leads_file):
    _complete_lead(client, session_id, project="First project")
    leads_before = _read_leads(isolated_leads_file)
    first_lead_id = leads_before[0]["lead_id"]

    client.post("/api/chat", json={"message": "book a call", "session_id": session_id})
    client.post("/api/chat", json={"message": "Gowri", "session_id": session_id})
    client.post("/api/chat", json={"message": "gowri@example.com", "session_id": session_id})
    client.post("/api/chat", json={"message": "Second project", "session_id": session_id})

    leads_after = _read_leads(isolated_leads_file)
    first_lead_now = next(l for l in leads_after if l["lead_id"] == first_lead_id)
    assert first_lead_now["project"] == "First project"  # untouched by the second enquiry


def test_conversation_sync_after_capture_does_not_corrupt_other_lead(client, session_id, isolated_leads_file):
    """Regression for the session_id-based update bug: syncing conversation
    for the SECOND lead must not overwrite the first lead's conversation."""
    _complete_lead(client, session_id, project="First project")
    client.post("/api/chat", json={"message": "book a call", "session_id": session_id})
    client.post("/api/chat", json={"message": "Gowri", "session_id": session_id})
    client.post("/api/chat", json={"message": "gowri@example.com", "session_id": session_id})
    client.post("/api/chat", json={"message": "Second project", "session_id": session_id})

    # one more post-capture turn to trigger update_lead_conversation again
    client.post("/api/chat", json={"message": "thanks for the help", "session_id": session_id})

    leads = _read_leads(isolated_leads_file)
    assert len(leads) == 2
    first, second = (l for l in sorted(leads, key=lambda l: l["project"]))
    # the two conversation transcripts must differ (first should be shorter/
    # frozen at its own handoff point, not clobbered with the second lead's tail)
    assert first["conversation"] != second["conversation"]


def test_cancel_does_not_create_a_lead(client, session_id, isolated_leads_file):
    client.post("/api/chat", json={"message": "I want a quote", "session_id": session_id})
    client.post("/api/chat", json={"message": "Alex", "session_id": session_id})
    resp = client.post("/api/chat", json={"message": "cancel", "session_id": session_id})
    assert resp.status_code == 200
    assert not _read_leads(isolated_leads_file)


def test_independent_sessions_do_not_interfere(client, isolated_leads_file):
    import uuid

    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())
    client.post("/api/chat", json={"message": "book a call", "session_id": session_a})
    resp_b = client.post("/api/chat", json={"message": "Gowri", "session_id": session_b})
    # session B never started a lead flow, so "Gowri" should NOT be treated as a name
    assert "thanks, gowri" not in resp_b.json()["answer"].lower()


# --- Issue 7: empty/whitespace name validation ---------------------------

def test_whitespace_only_name_rejected_at_api_boundary(client, session_id, isolated_leads_file):
    """A whitespace-only reply is rejected by Issue 2's message validator
    before it ever reaches the lead flow — that's a stricter, earlier defense
    than a lead-flow-specific reprompt, and it's the behavior that's actually
    in effect, so assert on it directly rather than the reprompt path."""
    client.post("/api/chat", json={"message": "I want to hire you", "session_id": session_id})
    resp = client.post("/api/chat", json={"message": "   ", "session_id": session_id})
    assert resp.status_code == 422


def test_name_stage_also_has_its_own_empty_check(monkeypatch):
    """Defense-in-depth: advance_lead_flow itself must still refuse an empty
    name if ever called with one (e.g. from a future caller that doesn't go
    through ChatRequest's message validator), not rely solely on Issue 2's
    API-boundary check."""
    import conversation

    session = {"history": [], "lead_stage": "name", "lead": {}, "lead_id": None}
    reply = conversation.advance_lead_flow(session, "   ", "some-session-id")
    assert reply is not None
    assert "name" in reply.lower()
    assert session["lead_stage"] == "name"  # did not advance to "contact"
    assert "name" not in session["lead"]


def test_whitespace_only_name_never_reaches_storage(client, session_id, isolated_leads_file):
    client.post("/api/chat", json={"message": "I want to hire you", "session_id": session_id})
    client.post("/api/chat", json={"message": "   ", "session_id": session_id})  # rejected
    client.post("/api/chat", json={"message": "Real Name", "session_id": session_id})  # accepted
    client.post("/api/chat", json={"message": "real@example.com", "session_id": session_id})
    client.post("/api/chat", json={"message": "a test project", "session_id": session_id})

    leads = _read_leads(isolated_leads_file)
    assert len(leads) == 1
    assert leads[0]["name"] == "Real Name"
    assert leads[0]["name"].strip() != ""


def test_valid_name_accepted(client, session_id, isolated_leads_file):
    resp = _complete_lead(client, session_id, name="Priya Nair")
    leads = _read_leads(isolated_leads_file)
    assert leads[0]["name"] == "Priya Nair"


def test_overlong_name_rejected_with_reprompt(client, session_id, isolated_leads_file):
    import conversation

    client.post("/api/chat", json={"message": "I want to hire you", "session_id": session_id})
    too_long_name = "A" * (conversation.MAX_NAME_LENGTH + 1)
    resp = client.post("/api/chat", json={"message": too_long_name, "session_id": session_id})
    assert resp.status_code == 200
    assert "long" in resp.json()["answer"].lower()
    assert not _read_leads(isolated_leads_file)
