"""Per-session chat history and the lead-qualification state machine.

The lead flow is a deterministic state machine (not LLM-parsed) so it stays
reliable on a small local model: a keyword match detects buying intent, then
each visitor reply is taken verbatim as the next field (name -> contact ->
project) rather than asking the model to extract structured data.

Sessions live in memory only — fine for a single-process prototype, but swap
for Redis/a DB before running multiple workers or wanting history to survive
a restart.
"""
import re
from typing import Optional

from crm import create_lead

EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PHONE_RE = re.compile(r"\d{7,}")


def is_valid_contact(text: str) -> bool:
    """A contact answer must contain something that looks like an email or a
    phone number (7+ digits once spaces/dashes/parens are stripped) — otherwise
    we'd happily accept any stray sentence as the visitor's contact info."""
    if EMAIL_RE.search(text):
        return True
    digits_only = re.sub(r"[\s\-()]", "", text)
    return bool(PHONE_RE.search(digits_only))

HISTORY_TURNS = 6  # prior messages (user+assistant) fed back to the model as context

LEAD_KEYWORDS = [
    "quote", "pricing", "price", "cost", "how much",
    "hire you", "hire tresvance", "work with you", "work with tresvance",
    "get started", "start a project", "new project",
    "talk to someone", "talk to a human", "talk to your team", "talk to our team",
    "speak to someone", "speak with someone", "speak to your team",
    "contact me", "reach out", "get in touch",
    "book a call", "schedule a call", "schedule a meeting",
    "interested in working", "sign me up", "request a demo", "demo",
    "develop a website", "build a website", "create a website", "design a website",
    "need a website", "want a website", "website for my business", "website for my",
    "develop an app", "build an app", "create an app", "need an app",
    "develop a platform", "build a platform", "create a platform",
    "build a custom", "develop a custom", "custom software", "custom website",
    "can you help me build", "can you help me develop", "can you help me create",
]

CANCEL_WORDS = {"cancel", "nevermind", "never mind", "stop", "forget it"}

SESSIONS: dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": [], "lead_stage": None, "lead": {}}
    return SESSIONS[session_id]


def detect_lead_intent(message: str) -> bool:
    lower = message.lower()
    return any(kw in lower for kw in LEAD_KEYWORDS)


def advance_lead_flow(session: dict, message: str, session_id: str) -> Optional[str]:
    """Returns a canned reply if this turn is handled by the lead flow,
    or None if the caller should fall through to the normal RAG answer."""
    stage = session["lead_stage"]
    text = message.strip()

    if stage in ("name", "contact", "project") and text.lower() in CANCEL_WORDS:
        session["lead_stage"] = None
        return "No problem — happy to just answer questions. What would you like to know?"

    if stage is None:
        if detect_lead_intent(text):
            session["lead_stage"] = "name"
            return "I'd love to connect you with our team on this. First, what's your name?"
        return None

    if stage == "name":
        session["lead"]["name"] = text
        session["lead_stage"] = "contact"
        return f"Thanks, {text}! What's the best email or phone number to reach you?"

    if stage == "contact":
        if not is_valid_contact(text):
            return "That doesn't look like a valid email or phone number — could you share one so our team can reach you?"
        session["lead"]["contact"] = text
        session["lead_stage"] = "project"
        return "Great — and briefly, what are you looking to build or get help with?"

    if stage == "project":
        session["lead"]["project"] = text
        create_lead(
            name=session["lead"].get("name", ""),
            contact=session["lead"].get("contact", ""),
            project=text,
            session_id=session_id,
            history=session["history"],
        )
        session["lead_stage"] = "captured"
        return (
            "Perfect — I've passed your details to our team and someone will reach out "
            "shortly. Anything else I can help with in the meantime?"
        )

    return None  # stage == "captured" -> normal RAG answers from here on
