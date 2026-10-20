"""Admin authentication for internal, non-public endpoints (lead data).

Minimal HTTP Basic auth, intentionally simple: this app has no user accounts
or sessions anywhere else, so adding a full auth framework would be out of
proportion to what's being protected (one internal leads view). Credentials
come only from environment variables (see .env.example) — never hardcoded,
never committed, and never defaulted to a guessable value. If they aren't
configured, admin endpoints fail closed (503) rather than silently opening.
"""
import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


def require_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    admin_username = os.environ.get("ADMIN_USERNAME")
    admin_password = os.environ.get("ADMIN_PASSWORD")

    if not admin_username or not admin_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin access is not configured on this server.",
        )

    username_ok = secrets.compare_digest(credentials.username, admin_username)
    password_ok = secrets.compare_digest(credentials.password, admin_password)
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
