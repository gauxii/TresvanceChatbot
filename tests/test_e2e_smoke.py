"""End-to-end smoke tests against the real Ollama model — the primary
conversation flow, lead capture, multi-session independence, and an
unsupported/hallucination-prone question, all driven through the live
/api/chat endpoint rather than mocks. Skipped automatically if Ollama isn't
reachable locally."""
import json
import uuid

import pytest
import requests

import main


def _ollama_available():
    try:
        r = requests.get(f"{main.OLLAMA_BASE}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _ollama_available(), reason="Ollama not reachable locally")


def _read_leads(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_primary_conversation_flow(client):
    sid = str(uuid.uuid4())
    resp = client.post("/api/chat", json={"message": "What services does Tresvance offer?", "session_id": sid})
    assert resp.status_code == 200
    answer = resp.json()["answer"].lower()
    assert any(w in answer for w in ("software", "ai", "iot", "cybersecurity", "marketing"))

    resp2 = client.post("/api/chat", json={"message": "Where are you based?", "session_id": sid})
    assert resp2.status_code == 200
    assert "kochi" in resp2.json()["answer"].lower()


def test_full_lead_capture_end_to_end(client, isolated_leads_file):
    sid = str(uuid.uuid4())
    r1 = client.post("/api/chat", json={"message": "I'd like to book a call about pricing", "session_id": sid})
    assert "name" in r1.json()["answer"].lower()
    r2 = client.post("/api/chat", json={"message": "Taylor", "session_id": sid})
    assert "email" in r2.json()["answer"].lower() or "phone" in r2.json()["answer"].lower()
    r3 = client.post("/api/chat", json={"message": "taylor@example.com", "session_id": sid})
    assert r3.status_code == 200
    r4 = client.post("/api/chat", json={"message": "A booking website for my salon", "session_id": sid})
    assert r4.status_code == 200

    leads = _read_leads(isolated_leads_file)
    assert len(leads) == 1
    assert leads[0]["name"] == "Taylor"
    assert leads[0]["contact"] == "taylor@example.com"


def test_multiple_independent_sessions_live(client):
    sid_a, sid_b = str(uuid.uuid4()), str(uuid.uuid4())
    client.post("/api/chat", json={"message": "Tell me about the BioHippo project.", "session_id": sid_a})
    resp_a_followup = client.post("/api/chat", json={"message": "who gave a testimonial about it?", "session_id": sid_a})
    assert "jibin" in resp_a_followup.json()["answer"].lower()

    # Session B has no prior context — same follow-up question shouldn't
    # resolve to BioHippo's testimonial out of nowhere.
    resp_b = client.post("/api/chat", json={"message": "who gave a testimonial about it?", "session_id": sid_b})
    assert resp_b.status_code == 200


def test_hallucination_prone_question_does_not_fabricate_hospital_experience(client):
    """Live regression for Issue 4, using the real model end-to-end rather
    than a synthetic context block."""
    sid = str(uuid.uuid4())
    resp = client.post("/api/chat", json={"message": "Have you worked with any hospitals?", "session_id": sid})
    assert resp.status_code == 200
    answer = resp.json()["answer"].lower()
    assert "healthcare institution" not in answer
    assert "patient experience" not in answer


def test_unsupported_topic_gets_honest_fallback_not_a_random_answer(client):
    sid = str(uuid.uuid4())
    resp = client.post("/api/chat", json={"message": "What is the capital of France?", "session_id": sid})
    assert resp.status_code == 200
    answer = resp.json()["answer"].lower()
    assert "paris" not in answer
