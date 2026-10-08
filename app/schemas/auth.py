"""Authentication request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

Role = str  # "admin" | "teacher"


class LoginRequest(BaseModel):
    """Username + password for the two development accounts."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"username": "admin", "password": "••••••"}}
    )

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class LoginResponse(BaseModel):
    """A signed session token plus the role it grants."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token": "YWRtaW46MTc2MjAwMDAwMzpv…",
                "role": "admin",
                "username": "admin",
                "expires_at": "2026-10-02T09:00:00Z",
            }
        }
    )

    token: str = Field(description="Send as the `X-API-Key` header on every request.")
    role: Role
    username: str
    expires_at: datetime