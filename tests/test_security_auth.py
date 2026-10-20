"""Issue 1: admin lead endpoints must require authentication; CORS must not
be wide open."""
import base64

import crm


def _basic_auth_header(username, password):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_api_leads_rejects_unauthenticated_request(client, monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct-horse-battery-staple")
    resp = client.get("/api/leads")
    assert resp.status_code in (401, 403)


def test_leads_page_rejects_unauthenticated_request(client, monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct-horse-battery-staple")
    resp = client.get("/leads")
    assert resp.status_code in (401, 403)


def test_api_leads_rejects_wrong_credentials(client, monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct-horse-battery-staple")
    resp = client.get("/api/leads", headers=_basic_auth_header("admin", "wrong-password"))
    assert resp.status_code == 401


def test_api_leads_accepts_correct_credentials(client, admin_env):
    username, password = admin_env
    resp = client.get("/api/leads", headers=_basic_auth_header(username, password))
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_api_leads_fails_closed_when_admin_not_configured(client, monkeypatch):
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    resp = client.get("/api/leads", headers=_basic_auth_header("admin", "anything"))
    assert resp.status_code == 503


def test_authenticated_admin_cannot_see_more_than_leads_data(client, admin_env):
    """Sanity check that the admin endpoint only returns lead records — not
    arbitrary server internals, env vars, etc."""
    username, password = admin_env
    resp = client.get("/api/leads", headers=_basic_auth_header(username, password))
    assert resp.status_code == 200
    for lead in resp.json():
        assert set(lead.keys()) <= {
            "lead_id", "timestamp", "session_id", "name", "contact", "project", "conversation"
        }


def test_public_chat_endpoint_does_not_require_admin_auth(client, session_id, mock_llm_success):
    """The public chatbot endpoint must remain unauthenticated — only the
    admin lead views are protected."""
    resp = client.post("/api/chat", json={"message": "hello", "session_id": session_id})
    assert resp.status_code != 401 and resp.status_code != 403


def test_cors_does_not_allow_arbitrary_origin(client, session_id, mock_llm_success):
    resp = client.post(
        "/api/chat",
        json={"message": "hi", "session_id": session_id},
        headers={"Origin": "https://evil-attacker-site.example"},
    )
    allow_origin = resp.headers.get("access-control-allow-origin")
    assert allow_origin != "*"
    assert allow_origin != "https://evil-attacker-site.example"


def test_cors_allows_configured_localhost_origin(client, session_id, mock_llm_success):
    resp = client.post(
        "/api/chat",
        json={"message": "hi", "session_id": session_id},
        headers={"Origin": "http://localhost:8000"},
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8000"
