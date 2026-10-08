"""Login / logout endpoints.

Credentials come from the environment (``ADMIN_USERNAME``/``ADMIN_PASSWORD``,
``TEACHER_USERNAME``/``TEACHER_PASSWORD``) - nothing is hard-coded and no
password is ever returned or logged. A successful login issues a signed
session token that is sent back as the ``X-API-Key`` header, so every existing
``require_admin`` / ``require_teacher`` dependency keeps working unchanged.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Security

from app.core.config import settings
from app.core.errors import UnauthorizedError
from app.core.security import (
    ROLE_ADMIN,
    ROLE_TEACHER,
    api_key_header,
    issue_token,
    verify_token,
)
from app.schemas.auth import LoginRequest, LoginResponse
from app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_UNAUTH = {"model": ErrorResponse, "description": "`unauthorized`."}


def _resolve_role(username: str, password: str) -> str | None:
    """Constant-time credential check against the two configured accounts."""
    accounts = (
        (settings.admin_username.strip(), settings.admin_password, ROLE_ADMIN),
        (settings.teacher_username.strip(), settings.teacher_password, ROLE_TEACHER),
    )
    for expected_user, expected_password, role in accounts:
        if not expected_password:
            continue
        # Compare both fields without early exit to avoid leaking which failed.
        user_ok = secrets.compare_digest(username.strip(), expected_user)
        password_ok = secrets.compare_digest(password, expected_password)
        if user_ok and password_ok:
            return role
    return None


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Sign in and receive a session token",
    responses={
        200: {"description": "Signed session token and the granted role."},
        401: _UNAUTH,
    },
)
async def login(payload: LoginRequest) -> LoginResponse:
    if not settings.login_enabled:
        raise UnauthorizedError(
            "Login is disabled. Set ADMIN_PASSWORD and TEACHER_PASSWORD in .env."
        )

    role = _resolve_role(payload.username, payload.password)
    if role is None:
        # Never say which half was wrong.
        logger.info("Failed login attempt for username %r", payload.username.strip())
        raise UnauthorizedError("Incorrect username or password.")

    token, expires_at = issue_token(role, payload.username.strip())
    logger.info("Login succeeded: role=%s", role)
    return LoginResponse(
        token=token,
        role=role,
        username=payload.username.strip(),
        expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc),
    )


@router.get(
    "/me",
    response_model=LoginResponse,
    summary="Inspect the current session",
    responses={
        200: {"description": "The role carried by the supplied token."},
        401: _UNAUTH,
    },
)
async def me(api_key: str | None = Security(api_key_header)) -> LoginResponse:
    """Return the caller's role. Proves a token is still valid."""
    session = verify_token(api_key)
    if session is None:
        raise UnauthorizedError("No active session. Please sign in again.")
    role, username = session
    # Re-derive expiry from the remaining TTL so the client can refresh its UI.
    _, expires_at = issue_token(role, username)
    return LoginResponse(
        token=api_key or "",
        role=role,
        username=username,
        expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc),
    )