"""Shared API schemas: error envelope, validation errors and health."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Standard error envelope returned by every failing endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"detail": "No face detected", "code": "no_face_detected"}
        }
    )

    detail: str = Field(description="Human-readable description of the problem.")
    code: str = Field(description="Stable machine-readable error code, e.g. `invalid_image`.")


class ValidationErrorResponse(BaseModel):
    """Returned with HTTP 422 when the request payload/path is malformed."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": [
                    {
                        "type": "int_parsing",
                        "loc": ["path", "student_id"],
                        "msg": "Input should be a valid integer, unable to parse string as an integer",
                    }
                ],
                "code": "invalid_request",
            }
        }
    )

    detail: list[dict[str, Any]] = Field(description="Per-field validation failures.")
    code: str = Field(default="invalid_request", description="Always `invalid_request`.")


class HealthResponse(BaseModel):
    """Liveness + model-readiness probe payload."""

    status: str = Field(description="`ok` once the face model is loaded, otherwise `starting`.")
    model_loaded: bool
    model: str = Field(description="InsightFace model package in use.")
    providers: list[str] = Field(description="Active ONNX Runtime execution providers.")
    match_threshold: float = Field(
        description="Configured cosine-similarity acceptance threshold (not a probability)."
    )
    match_margin: float = Field(
        default=0.0,
        description=(
            "Configured identification margin (FACE_MATCH_MARGIN): the minimum "
            "lead over the best other student required for acceptance. 0 disables "
            "the gate."
        ),
    )
    environment: str
    storage_backend: str = Field(
        default="unknown", description="`mongodb` or `memory`."
    )
    database_connected: bool = False
    database_info: str = ""
    students: int = 0
    face_embeddings: int = 0
    auth_enabled: bool = False


class MessageResponse(BaseModel):
    """Generic acknowledgement payload."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"message": "Student 101 deleted"}}
    )

    message: str
