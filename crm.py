"""Lead sink. This is the one place that needs to change once the in-house
CRM is ready — replace the body of create_lead() with a call to its API.
Everything upstream (the chat flow) stays the same.
"""
import datetime
import json
from pathlib import Path

LEADS_FILE = Path(__file__).parent / "leads.jsonl"


def create_lead(name: str, contact: str, project: str, session_id: str, history: list[dict]) -> None:
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "name": name,
        "contact": contact,
        "project": project,
        "conversation": history,
    }
    with LEADS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[LEAD CAPTURED] {name} <{contact}> — {project}")


def update_lead_conversation(session_id: str, history: list[dict]) -> None:
    """Visitors often keep chatting after a lead is captured — re-sync the
    stored conversation on every later turn so it reflects what was actually
    said, instead of freezing at the moment create_lead() first ran."""
    if not LEADS_FILE.exists():
        return
    lines = LEADS_FILE.read_text().splitlines()
    updated = False
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("session_id") == session_id:
            record["conversation"] = history
            lines[i] = json.dumps(record)
            updated = True
    if updated:
        LEADS_FILE.write_text("\n".join(lines) + "\n")


def list_leads() -> list[dict]:
    """Most-recent-first. Swap for a query against the real CRM once it exists."""
    if not LEADS_FILE.exists():
        return []
    leads = [json.loads(line) for line in LEADS_FILE.read_text().splitlines() if line.strip()]
    leads.reverse()
    return leads
