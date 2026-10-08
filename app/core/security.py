"""API-key / session-token authentication and role-based authorization.

Two roles exist:

* ``admin``   - student and face/biometric management (create, update, delete)
* ``teacher`` - classroom recognition and attendance operations

Credentials are supplied in the ``X-API-Key`` header and may be **either**:

1. a static API key (``ADMIN_API_KEY`` / ``TEACHER_API_KEY``) - unchanged, so
   existing clients and tests keep working; or
2. a session token issued by ``POST /api/auth/login``.

Tokens are ``base64(payload).base64(hmac_sha256(payload, AUTH_SECRET))`` where
the payload is ``"<role>:<expiry-unix-seconds>:<username>"``. They are signed
with the standard library only - no extra dependency and no user collection.

If no keys are configured the API runs unauthenticated - that is a
**development-only** convenience and ``ENVIRONMENT=production`` refuses to start
without keys (see ``Settings.validate_runtime``).

Face embeddings and raw face images are biometric data: every management
endpoint is behind :func:`require_admin`, and attendance/reporting endpoints sit
behind :func:`require_teacher`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import time

from fastapi import Security
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.core.errors import ForbiddenError, UnauthorizedError

logger = logging.getLogger(__name__)

ANONYMOUS = "anonymous"
ROLE_ADMIN = "admin"
ROLE_TEACHER = "teacher"

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description=(
        "Session token from POST /api/auth/login, or an admin/teacher API key. "
        "Ignored while authentication is disabled (development only)."
    ),
)


# ---------------------------------------------------------------------------
# Session tokens (stdlib HMAC, no extra dependency)
# ---------------------------------------------------------------------------
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _sign(payload: str) -> str:
    key = settings.auth_secret.strip().encode()
    return _b64e(hmac.new(key, payload.encode(), hashlib.sha256).digest())


def issue_token(role: str, username: str) -> tuple[str, int]:
    """Return ``(token, expires_at_epoch_seconds)`` for a freshly logged-in user."""
    ttl = settings.session_ttl_minutes * 60
    expires_at = int(time.time()) + ttl
    payload = f"{role}:{expires_at}:{username}"
    return f"{_b64e(payload.encode())}.{_sign(payload)}", expires_at


def verify_token(token: str | None) -> tuple[str, str] | None:
    """Validate a session token. Returns ``(role, username)`` or ``None``.

    Every failure path returns ``None`` rather than raising, so a malformed or
    tampered token is simply "not a valid credential".
    """
    if not token or "." not in token:
        return None
    encoded, _, signature = token.partition(".")
    if not encoded or not signature:
        return None

    try:
        payload = _b64d(encoded).decode()
    except Exception:  # noqa: BLE001 - malformed input is simply invalid
        return None

    # Constant-time signature check before trusting any of the payload.
    if not hmac.compare_digest(signature, _sign(payload)):
        return None

    parts = payload.split(":", 2)
    if len(parts) != 3:
        return None
    role, raw_expiry, username = parts
    if role not in (ROLE_ADMIN, ROLE_TEACHER):
        return None
    try:
        if int(raw_expiry) < int(time.time()):
            return None
    except ValueError:
        return None
    return role, username


def _matches(candidate: str | None, *keys: str) -> bool:
    """Constant-time comparison against any of ``keys``."""
    if not candidate:
        return False
    return any(key and secrets.compare_digest(candidate, key) for key in keys)


def _authorize(
    api_key: str | None, *, allow_admin: bool, allow_teacher: bool
) -> str:
    if not settings.auth_enabled and not settings.login_enabled:
        return ANONYMOUS

    # 1) A session token from the login page.
    session = verify_token(api_key)
    if session is not None:
        role, username = session
        if (role == ROLE_ADMIN and allow_admin) or (role == ROLE_TEACHER and allow_teacher):
            return f"{role}:{username}"
        raise ForbiddenError()

    # 2) A static API key (existing behaviour, preserved for compatibility).
    if settings.allow_api_key_auth:
        if allow_admin and _matches(api_key, settings.admin_api_key):
            return ROLE_ADMIN
        if allow_teacher and _matches(api_key, settings.teacher_api_key):
            return ROLE_TEACHER

        # A valid key used on the wrong endpoint is 403; anything else is 401.
        if _matches(api_key, settings.admin_api_key, settings.teacher_api_key):
            raise ForbiddenError()

    raise UnauthorizedError()


async def require_admin(api_key: str | None = Security(api_key_header)) -> str:
    """Dependency: only the admin role/key may call this endpoint."""
    return _authorize(api_key, allow_admin=True, allow_teacher=False)


async def require_teacher(api_key: str | None = Security(api_key_header)) -> str:
    """Dependency: teacher role/key (or admin, as superuser) may call this."""
    return _authorize(api_key, allow_admin=True, allow_teacher=True)


def role_of(actor: str) -> str:
    """Strip the ``"role:username"`` form produced by token auth back to a role."""
    return actor.split(":", 1)[0] if actor else ANONYMOUS
