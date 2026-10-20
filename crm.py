"""Lead sink. This is the one place that needs to change once the in-house
CRM is ready — replace the body of create_lead() with a call to its API.
Everything upstream (the chat flow) stays the same.
"""
import datetime
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Optional

LEADS_FILE = Path(__file__).parent / "leads.jsonl"
logger = logging.getLogger("tresvance")

# A visitor can now submit more than one lead per session (see conversation.py),
# so file writes/rewrites need to be serialized — otherwise two near-simultaneous
# appends (or an append racing a rewrite in update_lead_conversation) could
# interleave and corrupt the JSONL file.
_FILE_LOCK = threading.Lock()


def create_lead(
    name: str,
    contact: str,
    project: str,
    session_id: str,
    history: list[dict],
    lead_id: Optional[str] = None,
) -> str:
    lead_id = lead_id or str(uuid.uuid4())
    record = {
        "lead_id": lead_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "name": name,
        "contact": contact,
        "project": project,
        "conversation": history,
    }
    with _FILE_LOCK:
        with LEADS_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")
    logger.info("Lead captured: lead_id=%s session_id=%s", lead_id, session_id)
    return lead_id


def update_lead_conversation(lead_id: Optional[str], history: list[dict]) -> None:
    """Visitors often keep chatting after a lead is captured — re-sync the
    stored conversation on every later turn so it reflects what was actually
    said, instead of freezing at the moment create_lead() first ran.

    Matched by lead_id (unique per enquiry) rather than session_id, since a
    single session can now produce multiple leads — matching on session_id
    alone would overwrite every one of that visitor's leads with the same
    conversation transcript."""
    if not lead_id or not LEADS_FILE.exists():
        return
    with _FILE_LOCK:
        lines = LEADS_FILE.read_text().splitlines()
        updated = False
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("lead_id") == lead_id:
                record["conversation"] = history
                lines[i] = json.dumps(record)
                updated = True
                break  # lead_id is unique — no need to scan further
        if updated:
            LEADS_FILE.write_text("\n".join(lines) + "\n")


def list_leads() -> list[dict]:
    """Most-recent-first. Swap for a query against the real CRM once it exists."""
    if not LEADS_FILE.exists():
        return []
    leads = [json.loads(line) for line in LEADS_FILE.read_text().splitlines() if line.strip()]
    leads.reverse()
    return leads
