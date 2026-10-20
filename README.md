# Tresvance Assistant (standalone prototype)

A FAQ chatbot grounded in Tresvance's website content (services, projects, contact info),
running entirely on free, self-hosted models via [Ollama](https://ollama.com) — no per-token
API cost, only your own compute.

## Stack
- **Chat model**: `llama3.2:3b` by default (local, via Ollama) — override with the
  `CHAT_MODEL` env var; `qwen3:4b` also works (the code already strips `<think>` tags
  qwen3 wraps its reasoning in)
- **Embeddings**: `nomic-embed-text` (local, via Ollama) — override with `EMBED_MODEL`
- **Vector store**: ChromaDB (local file-based, `./chroma_db`)
- **Backend**: FastAPI
- **Frontend**: plain HTML/JS chat widget (`static/index.html`) — portable into the real
  site later as an embed

## Setup
```bash
cd ~/tresvance-chatbot
source venv/bin/activate
pip install -r requirements.txt

# make sure Ollama has the models (one-time) — match whatever CHAT_MODEL you configure
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# build the knowledge base (re-run whenever knowledge_base.py changes)
python ingest.py

# copy and fill in required configuration (see "Configuration" below)
cp .env.example .env
```

## Configuration
All configuration is read from environment variables (loaded from a local `.env` file
via `python-dotenv` if present — `.env` is gitignored, never commit real values). See
`.env.example` for the full list. Notably:
- `ADMIN_USERNAME` / `ADMIN_PASSWORD` — **required** to use `/leads` and `/api/leads`
  (see "Checking what's been captured" below). Without both set, those endpoints return
  503 rather than silently allowing access.
- `ALLOWED_ORIGINS` — comma-separated list of origins allowed to call `/api/chat` from a
  browser. Defaults to `http://localhost:8000,http://127.0.0.1:8000` for local dev; set
  this to the real site(s) embedding the widget before deploying (e.g.
  `https://tresvance.com`). Never leave this as a wildcard in production.
- `CHAT_MODEL`, `EMBED_MODEL`, `OLLAMA_BASE` — which models/host to use.
- `MAX_MESSAGE_LENGTH`, `OLLAMA_TIMEOUT_SECONDS` — input-validation and request-timeout
  limits for `/api/chat`.

## Run
```bash
source venv/bin/activate
uvicorn main:app --reload
```
Open http://localhost:8000 in a browser.

## Running tests
```bash
source venv/bin/activate
pip install -r requirements-dev.txt
pytest
```
Tests use FastAPI's `TestClient` and don't require Ollama or a running server — they do
require the `chroma_db/` knowledge base to have been built (`python ingest.py`), since
`main.py` opens that collection at import time.

## Updating the knowledge base
Edit `knowledge_base.py` (one entry per fact/topic), then re-run `python ingest.py`.
Content currently covers: company overview, the 6 services, the TRES AI product, why-choose-us
points, 4 past projects, 6 testimonials, and contact details — all scraped from
tresvance.com on 2026-10-02.

## Conversation memory & lead capture
Each browser gets a persistent `session_id` (generated client-side, stored in
`localStorage`), sent with every `/api/chat` call. The backend (`conversation.py`)
keeps an in-memory history per session and feeds the last few turns back to the model,
so follow-up questions ("who gave a testimonial about **it**?") resolve correctly.

When a visitor's message matches buying-intent keywords (`conversation.py:LEAD_KEYWORDS`
— "quote", "pricing", "hire you", "book a call", etc.) or they click the widget's
"Talk to our team" chip, the bot switches into a short qualification flow: asks for
name, then contact info, then project details, then hands the captured lead to
`crm.py:create_lead()`. Saying "cancel" at any point during the flow drops back to
normal FAQ mode.

**Checking what's been captured**: open `http://localhost:8000/leads` — a simple internal
page listing every captured lead (name, contact, project, timestamp) with an expandable
view of the full conversation that led to it. It reads straight from `leads.jsonl` via
`GET /api/leads`. Both routes require HTTP Basic auth (`ADMIN_USERNAME` / `ADMIN_PASSWORD`
env vars, see "Configuration" above) — your browser will prompt for credentials. If those
env vars aren't set, both routes return 503 rather than opening up.

A visitor can now submit more than one lead in the same browser session (e.g. a repeat
visitor, or a second enquiry) — each gets its own `lead_id` and its own line in
`leads.jsonl`, so later conversation syncing never overwrites a different lead from the
same session.

**Wiring up the real CRM**: `crm.py` is the only file that needs to change. Right now
`create_lead()` appends each lead as a JSON line to `leads.jsonl` and prints it to the
console — replace its body with a `requests.post(...)` to your in-house CRM's API once
it's ready; nothing in `main.py` or `conversation.py` needs to change.

**Known limitations** (fine for a prototype, worth revisiting before production):
- Sessions and history live in memory only — they reset if the server restarts, and
  won't work correctly if you ever run multiple backend workers/processes. Move to
  Redis or a DB for that.
- The lead flow is a simple deterministic state machine: whatever the visitor types
  next is taken as the literal answer to "what's your name / contact / project?". If
  they ask an unrelated question mid-flow instead of answering, it'll be captured as
  if it were that field — typing "cancel" is the current escape hatch.

## Embedding on the real site
`static/widget.js` is a self-contained, dependency-free floating chat button + panel.
Drop one line into tresvance.com once this backend is deployed:
```html
<script src="https://YOUR-DEPLOY-HOST/static/widget.js" data-api-base="https://YOUR-DEPLOY-HOST"></script>
```
- `data-api-base` is optional — if omitted, it defaults to the origin the script was loaded
  from (handy when the widget is served from the same host as the main site).
- The backend has CORS enabled (`main.py`) so the widget can call `/api/chat` from a different
  origin than the one it's hosted on, scoped to the `ALLOWED_ORIGINS` env var (see
  "Configuration" above). Once a production domain is picked, set `ALLOWED_ORIGINS` to
  that exact origin — it is never a wildcard.
- `static/embed-test.html` is a throwaway page (styled like a generic site, not Tresvance's)
  used to verify the widget embeds and talks to the backend correctly — open
  `http://localhost:8000/static/embed-test.html` to try it.

## Next steps (once you have site access)
- Deploy: a small CPU VPS (e.g., Hetzner, ~€15-20/mo) is enough for this model size at low
  traffic. Install Ollama + this app there instead of running locally.
- Expand the knowledge base as new pages/case studies are added to the real site.
