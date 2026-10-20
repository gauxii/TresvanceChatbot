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
import threading
import uuid
from typing import Optional

from crm import create_lead

MAX_NAME_LENGTH = 200

EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PHONE_RE = re.compile(r"(?<!\d)\d{10}(?!\d)")
# A run of 7+ digits is clearly an attempted phone number (as opposed to a
# stray short number) — used to catch a malformed phone even when a valid
# email is also present in the same message, e.g. "89034526178,maya07@gmail.com".
PHONE_ATTEMPT_RE = re.compile(r"\d{7,}")


def is_valid_contact(text: str) -> bool:
    """A contact answer must contain either a valid email (has an @ and a
    domain) or a valid phone number (exactly 10 digits once spaces/dashes/
    parens/plus signs are stripped) — otherwise we'd happily accept any stray
    sentence as the visitor's contact info. If a phone-number-shaped digit
    run is present, it must be exactly 10 digits even when an email is also
    given — a valid email shouldn't mask an obviously wrong phone number."""
    digits_only = re.sub(r"[\s\-()+]", "", text)
    phone_attempts = PHONE_ATTEMPT_RE.findall(digits_only)
    if any(len(run) != 10 for run in phone_attempts):
        return False
    if EMAIL_RE.search(text):
        return True
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

# A lock per session_id, so concurrent/duplicate requests for the SAME visitor
# are serialized (no interleaved state transitions, no duplicate leads from a
# double-click or retried request) without blocking unrelated sessions. The
# meta-lock only protects creation of a new per-session lock, so it's held
# very briefly, not for the whole request.
_SESSION_LOCKS: dict[str, threading.Lock] = {}
_SESSION_LOCKS_META_LOCK = threading.Lock()


def get_session(session_id: str) -> dict:
    with _SESSION_LOCKS_META_LOCK:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = {
                "history": [],
                "lead_stage": None,
                "lead": {},
                "lead_id": None,
            }
        return SESSIONS[session_id]


def get_session_lock(session_id: str) -> threading.Lock:
    with _SESSION_LOCKS_META_LOCK:
        if session_id not in _SESSION_LOCKS:
            _SESSION_LOCKS[session_id] = threading.Lock()
        return _SESSION_LOCKS[session_id]


def detect_lead_intent(message: str) -> bool:
    lower = message.lower()
    return any(kw in lower for kw in LEAD_KEYWORDS)


def advance_lead_flow(session: dict, message: str, session_id: str) -> Optional[str]:
    """Returns a canned reply if this turn is handled by the lead flow,
    or None if the caller should fall through to the normal RAG answer.

    Call this only while holding conversation.get_session_lock(session_id) —
    it mutates session state and is not safe to run concurrently for the
    same session."""
    stage = session["lead_stage"]
    text = message.strip()

    if stage in ("name", "contact", "project") and text.lower() in CANCEL_WORDS:
        session["lead_stage"] = None
        session["lead"] = {}
        return "No problem — happy to just answer questions. What would you like to know?"

    if stage is None or stage == "captured":
        if detect_lead_intent(text):
            returning = stage == "captured"
            # Starting a fresh enquiry: the PREVIOUS lead was already written
            # to storage by create_lead() and is never touched again here —
            # this only resets the in-progress staging dict for the new one.
            session["lead"] = {}
            session["lead_id"] = None
            session["lead_stage"] = "name"
            if returning:
                return "Happy to help with another enquiry! What's your name?"
            return "I'd love to connect you with our team on this. First, what's your name?"
        return None

    if stage == "name":
        name = text
        if not name:
            return "I didn't quite catch a name there — what should I call you?"
        if len(name) > MAX_NAME_LENGTH:
            return f"That name's a bit long (max {MAX_NAME_LENGTH} characters) — could you give a shorter version?"
        session["lead"]["name"] = name
        session["lead_stage"] = "contact"
        return f"Thanks, {name}! What's the best email or phone number to reach you?"

    if stage == "contact":
        if not is_valid_contact(text):
            return "That doesn't look like a valid email or phone number — please enter a valid email (e.g. name@example.com) or a 10-digit phone number."
        session["lead"]["contact"] = text
        session["lead_stage"] = "project"
        return "Great — and briefly, what are you looking to build or get help with?"

    if stage == "project":
        lead_id = str(uuid.uuid4())
        session["lead"]["project"] = text
        session["lead_id"] = lead_id
        create_lead(
            name=session["lead"].get("name", ""),
            contact=session["lead"].get("contact", ""),
            project=text,
            session_id=session_id,
            history=session["history"],
            lead_id=lead_id,
        )
        session["lead_stage"] = "captured"
        return (
            "Perfect — I've passed your details to our team and someone will reach out "
            "shortly. Anything else I can help with in the meantime?"
        )

    return None
