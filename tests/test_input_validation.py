"""Issue 2: invalid input must get a controlled 4xx/5xx response, never an
unhandled crash, and the server must keep working afterward."""
import main


def test_empty_message_rejected(client, session_id):
    resp = client.post("/api/chat", json={"message": "", "session_id": session_id})
    assert resp.status_code == 422


def test_whitespace_only_message_rejected(client, session_id):
    resp = client.post("/api/chat", json={"message": "   \n\t  ", "session_id": session_id})
    assert resp.status_code == 422


def test_very_long_message_rejected(client, session_id):
    huge = "A" * (main.MAX_MESSAGE_LENGTH + 1)
    resp = client.post("/api/chat", json={"message": huge, "session_id": session_id})
    assert resp.status_code == 422


def test_message_at_max_length_accepted(client, session_id, mock_llm_success):
    at_limit = "A" * main.MAX_MESSAGE_LENGTH
    resp = client.post("/api/chat", json={"message": at_limit, "session_id": session_id})
    assert resp.status_code == 200


def test_missing_message_field(client, session_id):
    resp = client.post("/api/chat", json={"session_id": session_id})
    assert resp.status_code == 422


def test_missing_session_id_field(client):
    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 422


def test_empty_session_id_rejected(client):
    resp = client.post("/api/chat", json={"message": "hi", "session_id": ""})
    assert resp.status_code == 422


def test_whitespace_only_session_id_rejected(client):
    resp = client.post("/api/chat", json={"message": "hi", "session_id": "   "})
    assert resp.status_code == 422


def test_too_long_session_id_rejected(client):
    resp = client.post(
        "/api/chat",
        json={"message": "hi", "session_id": "x" * (main.MAX_SESSION_ID_LENGTH + 1)},
    )
    assert resp.status_code == 422


def test_message_wrong_type_rejected(client, session_id):
    resp = client.post("/api/chat", json={"message": 12345, "session_id": session_id})
    assert resp.status_code == 422


def test_session_id_wrong_type_rejected(client):
    resp = client.post("/api/chat", json={"message": "hi", "session_id": 42})
    assert resp.status_code == 422


def test_malformed_json_body_rejected(client):
    resp = client.post(
        "/api/chat",
        data="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_error_responses_do_not_leak_stack_traces(client, session_id):
    huge = "A" * (main.MAX_MESSAGE_LENGTH + 1)
    resp = client.post("/api/chat", json={"message": huge, "session_id": session_id})
    body_text = resp.text.lower()
    assert "traceback" not in body_text
    assert "file \"" not in body_text
    assert ".py\", line" not in body_text


def test_embedding_failure_returns_safe_503(client, session_id, monkeypatch):
    def broken_embed(text):
        raise main.AssistantUnavailableError("embedding request failed")

    monkeypatch.setattr(main, "embed", broken_embed)
    resp = client.post("/api/chat", json={"message": "what services do you offer?", "session_id": session_id})
    assert resp.status_code == 503
    assert "traceback" not in resp.text.lower()


def test_vector_db_failure_returns_safe_503(client, session_id, monkeypatch):
    def broken_query(*args, **kwargs):
        raise ValueError("Expected each embedding in the embeddings to be a non-empty list")

    monkeypatch.setattr(main.collection, "query", broken_query)
    monkeypatch.setattr(main, "embed", lambda text: [0.1, 0.2, 0.3])
    resp = client.post("/api/chat", json={"message": "what services do you offer?", "session_id": session_id})
    assert resp.status_code == 503


def test_llm_generation_failure_returns_safe_503(client, session_id, monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda q: (["contact-info"], ["Contact Tresvance..."]))

    def broken_generate(query, context_chunks, prior_turns):
        raise main.AssistantUnavailableError("chat generation request failed")

    monkeypatch.setattr(main, "generate", broken_generate)
    resp = client.post("/api/chat", json={"message": "what services do you offer?", "session_id": session_id})
    assert resp.status_code == 503


def test_llm_timeout_returns_safe_503(client, session_id, monkeypatch):
    import requests

    monkeypatch.setattr(main, "retrieve", lambda q: (["contact-info"], ["Contact Tresvance..."]))

    def timing_out_generate(query, context_chunks, prior_turns):
        raise main.AssistantUnavailableError("timed out") from requests.exceptions.Timeout()

    monkeypatch.setattr(main, "generate", timing_out_generate)
    resp = client.post("/api/chat", json={"message": "what services do you offer?", "session_id": session_id})
    assert resp.status_code == 503


def test_server_stays_operational_after_a_failed_request(client, session_id, monkeypatch):
    """One bad request must not crash or corrupt the app for the next one."""
    resp1 = client.post("/api/chat", json={"message": "", "session_id": session_id})
    assert resp1.status_code == 422

    def fake_retrieve(q):
        return ["contact-info"], ["Contact Tresvance..."]

    def fake_generate(q, c, p):
        return "We're happy to help with your Tresvance question."

    monkeypatch.setattr(main, "retrieve", fake_retrieve)
    monkeypatch.setattr(main, "generate", fake_generate)
    resp2 = client.post("/api/chat", json={"message": "hello again", "session_id": session_id})
    assert resp2.status_code == 200


def test_ollama_requests_have_a_timeout_configured():
    """Guards against the original bug: requests.post() calls to Ollama had
    no timeout at all, so a hung Ollama could hang a request forever."""
    import inspect

    embed_src = inspect.getsource(main.embed)
    generate_src = inspect.getsource(main.generate)
    assert "timeout=" in embed_src
    assert "timeout=" in generate_src
