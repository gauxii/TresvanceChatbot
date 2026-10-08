"""FastAPI backend for the Tresvance FAQ chatbot.
Retrieves relevant knowledge-base chunks from Chroma, then asks a local
Ollama model (qwen3:4b) to answer grounded in that context.
"""
import re
import requests
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from conversation import HISTORY_TURNS, advance_lead_flow, get_session
from crm import list_leads, update_lead_conversation

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "tresvance_kb"
TOP_K = 6

# Single source of truth for contact details — reused by both the system prompt
# and the grounding-fallback reply so neither ever invents or duplicates it.
CONTACT_EMAIL = "contact@tresvance.com"
CONTACT_PHONE = "+91 8921187643"

SYSTEM_PROMPT = (
    "You are the Tresvance website assistant. Tresvance is a software, AI, IoT and "
    "cybersecurity company based in Kochi, India. Answer the visitor's question using "
    "ONLY the context provided below. Be concise and friendly. If the context does not "
    "contain the answer, say you don't have that information and suggest contacting "
    f"Tresvance at {CONTACT_EMAIL} or {CONTACT_PHONE}. Do not make up facts. "
    "Reply in plain text only — no markdown, no asterisks for bold or italics, no "
    "headings or bullet lists with * or -. If a visitor wants to start a project or "
    "get a quote, never tell them to visit a contact page or fill out a website form — "
    "instead offer to take their name, contact info, and project details right here "
    "in the chat. "
    "The context may include descriptions of Tresvance's PAST client projects and "
    "testimonials (named clients and their businesses) — these describe prior work "
    "Tresvance has completed, never the current visitor's own business. Never adopt a "
    "past client's name, company, or project details as if they belonged to the "
    "visitor, and never design or describe 'their project' using a past client's "
    "details unless the visitor explicitly asked about that specific client. "
    "When asked about past projects, clients, or case studies, mention only clients, "
    "industries, and outcomes explicitly named in the context above. Never invent "
    "additional example projects, clients, or industries to make the answer sound "
    "more complete — if the context supports only one example, give just that one. "
    "If the context doesn't support an answer at all, say so and offer the contact "
    "details instead."
)

POST_CAPTURE_ACK = (
    "Thanks, I've noted that — our team will go over the details when they reach out. "
    "Let me know if you have any other questions about Tresvance!"
)

QUESTION_STARTERS = (
    "what", "how", "when", "where", "who", "why", "which", "can", "could",
    "is", "are", "do", "does", "did", "will", "would", "should",
)


def is_question(text: str) -> bool:
    stripped = text.strip().lower()
    return stripped.endswith("?") or stripped.startswith(QUESTION_STARTERS)

app = FastAPI()
# Widget is meant to be embedded on tresvance.com (a different origin from wherever
# this backend ends up deployed), so the chat endpoint needs CORS enabled.
# Once a production domain is known, replace "*" with that exact origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)


class ChatRequest(BaseModel):
    message: str
    session_id: str


def embed(text: str) -> list[float]:
    resp = requests.post(f"{OLLAMA_BASE}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()["embedding"]


def retrieve(query: str) -> tuple[list[str], list[str]]:
    query_embedding = embed(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    # ids are returned alongside documents so the grounding fallback can tell
    # verified project/testimonial chunks apart from general service chunks
    return results["ids"][0], results["documents"][0]


def generate(query: str, context_chunks: list[str], prior_turns: list[dict]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nVisitor question: {query}\n\nAnswer:"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(prior_turns)
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": False,
        },
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    # qwen3 is a reasoning model and may wrap chain-of-thought in <think> tags
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # widget renders plain text, so strip any markdown emphasis the model still adds
    content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)
    content = re.sub(r"\*(.*?)\*", r"\1", content)
    return content


# --- Post-generation grounding validation -----------------------------------
# llama3.2:3b sometimes answers "give me an example" by inventing plausible but
# fake client projects (e.g. "a pharmaceutical company", fabricated percentage
# metrics) even though the prompt says to only use the provided context. These
# checks catch that *after* generation, deterministically and without a second
# LLM call, before the answer is ever shown to the user.

PROJECT_QUESTION_KEYWORDS = (
    "project", "client", "case stud", "portfolio", "example",
    "worked with", "completed", "outcome", "result", "technolog",
    "built for", "delivered", "previous work", "past work",
    "have you built", "have you worked", "have you done",
)

# Acronyms that are common in this domain and would otherwise trip the
# ALL-CAPS named-entity check even though they aren't a client name.
ACRONYM_ALLOWLIST = {"AI", "IT", "IOT", "SEO", "CRM", "API", "FAQ", "UI", "UX", "QA", "SDK", "HR", "ROI", "KPI"}

# Nouns that signal "the model is describing a specific past client" when
# preceded by an indefinite article (e.g. "a pharmaceutical company").
GENERIC_CLIENT_NOUNS = (
    "company", "client", "firm", "manufacturer", "provider", "organization",
    "institution", "enterprise", "business", "vendor", "corporation", "startup",
)
GENERIC_CLIENT_PATTERN = re.compile(
    rf"\b(?:a|an)\s+((?:[A-Za-z]+\s+){{0,3}})({'|'.join(GENERIC_CLIENT_NOUNS)})\b",
    re.IGNORECASE,
)

# Words too generic/common to count as evidence that a specific industry or
# client descriptor is actually supported by the retrieved context — e.g. the
# word "research" alone shouldn't "verify" an invented claim just because it
# happens to appear somewhere in the KB.
GENERIC_DESCRIPTOR_WORDS = {
    "a", "an", "the", "us", "major", "leading", "large", "small", "global",
    "local", "international", "renowned", "top", "premier", "based", "new",
    "company", "business", "client", "research", "technology", "service",
    "services", "solution", "solutions", "industry", "sector", "organization",
    "firm", "work", "platform", "system", "product", "project",
}

PROJECT_NAME_RE = re.compile(r"^Project:\s*([^.]+)\.")

# AI-related attribution terms. "\bAI\b" alone already matches inside the
# hyphenated forms (the hyphen is a word boundary), but they're spelled out
# explicitly so the full set this check targets is clear at a glance.
AI_TERM_PATTERN = re.compile(
    r"\bAI\b|artificial intelligence|machine learning|\bML\b|"
    r"AI-powered|AI-driven|AI-integrated|AI-based|AI-enabled",
    re.IGNORECASE,
)


def _group_project_chunks(ids: list[str], chunks: list[str]) -> dict[str, list[str]]:
    """Group this turn's retrieved project/testimonial chunks by project key
    (e.g. 'project-sweetbloom' and 'testimonial-sweetbloom' -> 'sweetbloom'),
    so claims about a given project are checked only against ITS OWN chunks."""
    groups: dict[str, list[str]] = {}
    for doc_id, text in zip(ids, chunks):
        if doc_id.startswith("project-") or doc_id.startswith("testimonial-"):
            key = doc_id.split("-", 1)[1]
            groups.setdefault(key, []).append(text)
    return groups


def display_name(key: str, texts: list[str]) -> str:
    for text in texts:
        m = PROJECT_NAME_RE.match(text)
        if m:
            return m.group(1).strip()
    return key.replace("-", " ").title()


def is_project_question(user_message: str) -> bool:
    """Only apply the stricter generic-client check to questions that are
    actually asking about past projects/clients/case studies, so normal
    service-description answers aren't scrutinized as aggressively."""
    lower = user_message.lower()
    return any(kw in lower for kw in PROJECT_QUESTION_KEYWORDS)


# Negation/uncertainty cues — a generic-client phrase sitting in the same
# sentence as one of these is the model correctly saying it DOESN'T have a
# claim (e.g. "I don't have information about a pharmaceutical company"), not
# asserting one. Only skip the match for that sentence, not the whole answer.
NEGATION_PATTERN = re.compile(
    r"\b(don't|do not|doesn't|does not|didn't|did not|haven't|have not|"
    r"hasn't|has not|can't confirm|cannot confirm|can not confirm|"
    r"don't know of|do not know of|no information|not worked with|"
    r"no record|not aware|unfortunately|not sure|no info)\b",
    re.IGNORECASE,
)


def _sentence_containing(text: str, start: int, end: int) -> str:
    """Return the sentence span around [start:end) using '.', '!', '?' as
    boundaries, so negation is checked clause-locally rather than over the
    whole answer (which could hide an unrelated denial elsewhere)."""
    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start))
    right_candidates = [i for i in (text.find(".", end), text.find("!", end), text.find("?", end)) if i != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right + 1]


def extract_unverified_client_claims(answer: str, context_text: str) -> list[str]:
    """Find 'a/an <descriptor> company/client/...' phrases in the answer whose
    descriptor words aren't backed by the retrieved context. Generic words are
    stripped first so they can't trivially "verify" an invented claim."""
    context_lower = context_text.lower()
    unsupported = []
    for match in GENERIC_CLIENT_PATTERN.finditer(answer):
        sentence = _sentence_containing(answer, match.start(), match.end())
        if NEGATION_PATTERN.search(sentence):
            continue  # e.g. "I don't have information about a pharmaceutical company"
        descriptor_words = match.group(1).lower().split()
        content_words = [w for w in descriptor_words if w not in GENERIC_DESCRIPTOR_WORDS]
        if not content_words:
            continue  # e.g. "a company" alone — too vague to check, allow it
        if not any(w in context_lower for w in content_words):
            unsupported.append(match.group(0))
    return unsupported


def extract_unsupported_ai_attributions(answer: str, ids: list[str], chunks: list[str]) -> list[str]:
    """A project may only be described as AI-related if THAT project's own
    retrieved chunks mention AI — not just any AI-related chunk elsewhere in
    context. Catches cross-chunk splicing: e.g. "SweetBloom" (real) +
    "AI" (real, but from an unrelated chunk) combined into a relationship
    neither chunk actually states."""
    groups = _group_project_chunks(ids, chunks)
    unsupported = []
    for key, texts in groups.items():
        if AI_TERM_PATTERN.search(" ".join(texts)):
            continue  # this project's own chunks do support an AI claim
        name = display_name(key, texts)
        for m in re.finditer(re.escape(name), answer, re.IGNORECASE):
            sentence = _sentence_containing(answer, m.start(), m.end())
            if AI_TERM_PATTERN.search(sentence):
                unsupported.append(name)
                break
    return unsupported


def is_grounded(answer: str, ids: list[str], chunks: list[str], user_message: str) -> bool:
    """Retrieved context is the source of truth: every specific numeric claim,
    named organization, or client/industry descriptor in the answer must trace
    back to the chunks actually retrieved for this query."""
    context_text = " ".join(chunks)

    # Percentage/metric claims must appear verbatim in the retrieved context.
    for pct in re.findall(r"\d+(?:\.\d+)?\s?%", answer):
        if pct.replace(" ", "") not in context_text.replace(" ", ""):
            return False

    # ALL-CAPS tokens (how this KB styles real client names, e.g. BIOHIPPO)
    # must also appear in the retrieved context, unless they're a known acronym.
    for token in re.findall(r"\b[A-Z]{3,}\b", answer):
        if token not in ACRONYM_ALLOWLIST and token not in context_text:
            return False

    # Generic "a/an ... company/client/..." claims, only for project-type
    # questions, since that phrasing is how the model invents anonymous clients.
    if is_project_question(user_message):
        if extract_unverified_client_claims(answer, context_text):
            return False

    # A named project can only be called AI-related if ITS OWN retrieved
    # chunks say so — always on, since it only fires when a real project name
    # and an AI term are stitched together in the same sentence.
    if extract_unsupported_ai_attributions(answer, ids, chunks):
        return False

    return True


def safe_fallback(ids: list[str], chunks: list[str]) -> str:
    """Built only from chunks actually retrieved for this query — never pads
    with more examples than were verified, never invents contact details."""
    groups = _group_project_chunks(ids, chunks)

    if not groups:
        return (
            "I don't have a verified example for that in my current information. "
            f"I'd be happy to connect you with our team — reach out at {CONTACT_EMAIL} "
            f"or {CONTACT_PHONE} for more detail."
        )

    names = [display_name(key, texts) for key, texts in groups.items()]
    if len(names) == 1:
        intro = f"We can currently verify the {names[0]} project as an example."
    else:
        intro = f"We can currently verify the following projects: {', '.join(names[:-1])} and {names[-1]}."

    return (
        f"{intro} I don't have additional verified examples available in my current "
        f"information. If you'd like to discuss something similar, reach out at "
        f"{CONTACT_EMAIL} or {CONTACT_PHONE}."
    )


@app.post("/api/chat")
def chat(req: ChatRequest):
    session = get_session(req.session_id)
    session["history"].append({"role": "user", "content": req.message})

    answer = advance_lead_flow(session, req.message, req.session_id)
    if answer is None and session["lead_stage"] == "captured" and not is_question(req.message):
        # lead already handed off — a plain statement here is more project detail,
        # not a question, so skip the LLM rather than let it freelance a design
        answer = POST_CAPTURE_ACK
    if answer is None:
        prior_turns = session["history"][-(HISTORY_TURNS + 1):-1]
        # fold the last exchange into the retrieval query so follow-ups like
        # "who gave a testimonial about it?" still retrieve the right chunk
        last_exchange = " ".join(t["content"] for t in prior_turns[-2:])
        retrieval_query = f"{last_exchange} {req.message}".strip()
        ids, chunks = retrieve(retrieval_query)
        answer = generate(req.message, chunks, prior_turns)
        if not is_grounded(answer, ids, chunks, req.message):
            answer = safe_fallback(ids, chunks)

    session["history"].append({"role": "assistant", "content": answer})

    if session["lead_stage"] == "captured":
        # keep the CRM record in sync with anything said after handoff
        update_lead_conversation(req.session_id, session["history"])

    return {"answer": answer}


@app.get("/api/leads")
def leads():
    return list_leads()


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/leads")
def leads_page():
    return FileResponse("static/leads.html")
