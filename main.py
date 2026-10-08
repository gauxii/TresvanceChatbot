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
from crm import list_leads

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "tresvance_kb"
TOP_K = 6

SYSTEM_PROMPT = (
    "You are the Tresvance website assistant. Tresvance is a software, AI, IoT and "
    "cybersecurity company based in Kochi, India. Answer the visitor's question using "
    "ONLY the context provided below. Be concise and friendly. If the context does not "
    "contain the answer, say you don't have that information and suggest contacting "
    "Tresvance at contact@tresvance.com or +91 8921187643. Do not make up facts. "
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
    "details unless the visitor explicitly asked about that specific client."
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


def retrieve(query: str) -> list[str]:
    query_embedding = embed(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    return results["documents"][0]


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
        chunks = retrieve(retrieval_query)
        answer = generate(req.message, chunks, prior_turns)

    session["history"].append({"role": "assistant", "content": answer})
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
