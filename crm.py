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


def list_leads() -> list[dict]:
    """Most-recent-first. Swap for a query against the real CRM once it exists."""
    if not LEADS_FILE.exists():
        return []
    leads = [json.loads(line) for line in LEADS_FILE.read_text().splitlines() if line.strip()]
    leads.reverse()
    return leads
