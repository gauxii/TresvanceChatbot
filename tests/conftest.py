"""Shared pytest fixtures.

Critical invariant: no test may ever touch the real leads.jsonl. The
isolated_leads_file fixture is autouse precisely so this is impossible to
forget in an individual test.
"""
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")

import crm  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_leads_file(tmp_path, monkeypatch):
    """Point crm.LEADS_FILE at a throwaway file for the duration of each test.
    main.py and conversation.py both call into crm's functions, which resolve
    LEADS_FILE as a module global at call time, so this redirects every code
    path — never just crm.py's own tests."""
    test_leads_path = tmp_path / "test_leads.jsonl"
    monkeypatch.setattr(crm, "LEADS_FILE", test_leads_path)
    yield test_leads_path


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def session_id():
    return str(uuid.uuid4())


@pytest.fixture
def admin_env(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "testadmin")
    monkeypatch.setenv("ADMIN_PASSWORD", "testpassword123")
    return ("testadmin", "testpassword123")


@pytest.fixture
def mock_llm_success(monkeypatch):
    """Stub out the Ollama/Chroma round trip with a canned, grounded answer so
    tests that just need /api/chat to succeed don't depend on a running
    Ollama instance or take several seconds per call. Tests that actually
    verify model/grounding behavior should NOT use this fixture."""
    ids = ["contact-info"]
    chunks = [
        "Contact Tresvance: Email info@tresvance.com or contact@tresvance.com. "
        "Phone: +91 8921187643."
    ]

    def fake_retrieve(query):
        return ids, chunks

    def fake_generate(query, context_chunks, prior_turns):
        return "Tresvance is a software, AI, IoT and cybersecurity company based in Kochi, India."

    monkeypatch.setattr(main, "retrieve", fake_retrieve)
    monkeypatch.setattr(main, "generate", fake_generate)
