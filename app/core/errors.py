"""Application error types and FastAPI exception handlers.

Every failure returned to a client has a stable machine-readable ``code`` and a
short human-readable ``detail``. Internal stack traces are logged server-side
but **never** sent to clients (see the privacy/security requirements).

Response envelope for every error::

    {"detail": "<message>", "code": "<machine_code>"}
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------
class AppError(Exception):
    """Base class for expected, client-facing errors."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "Internal server error"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.message
        super().__init__(self.message)


class InvalidImageError(AppError):
    status_code = 400
    code = "invalid_image"
    message = "Invalid image"


class NoFaceDetectedError(AppError):
    status_code = 400
    code = "no_face_detected"
    message = "No face detected"


class MultipleFacesDetectedError(AppError):
    status_code = 400
    code = "multiple_faces_detected"
    message = "Multiple faces detected during registration"


class FaceTooSmallError(AppError):
    status_code = 400
    code = "face_too_small"
    message = "Detected face is too small or too low quality"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"
    message = "Uploaded image exceeds the maximum allowed size"


class StudentNotFoundError(AppError):
    status_code = 404
    code = "student_not_found"
    message = "Student not found"


class ModelNotReadyError(AppError):
    status_code = 500
    code = "model_unavailable"
    message = "Face recognition model initialization failed"


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "service_unavailable"
    message = "Service temporarily unavailable"


class UnauthorizedError(AppError):
    status_code = 401
    code = "unauthorized"
    message = "Missing or invalid API key"


class ForbiddenError(AppError):
    status_code = 403
    code = "forbidden"
    message = "This API key is not permitted to perform this operation"


class DuplicateRollNumberError(AppError):
    status_code = 409
    code = "duplicate_roll_number"
    message = "A student with this roll number already exists in this class/section"


class DuplicateFaceError(AppError):
    status_code = 409
    code = "duplicate_face"
    message = "This photo is too similar to one already registered for this student"


class TooManyFacesError(AppError):
    status_code = 400
    code = "too_many_faces"
    message = "This student already has the maximum number of face embeddings"


class AttendanceNotFoundError(AppError):
    status_code = 404
    code = "attendance_not_found"
    message = "Attendance record not found"


class InvalidAttendanceStatusError(AppError):
    status_code = 400
    code = "invalid_attendance_status"
    message = "Unsupported attendance status"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _error_response(status_code: int, code: str, detail) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail, "code": code})


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the project-wide error contract to ``app``."""

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        log = logger.error if exc.status_code >= 500 else logger.info
        log("%s %s -> %s (%s): %s", request.method, request.url.path,
            exc.status_code, exc.code, exc.message)
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info("%s %s -> 422: invalid request", request.method, request.url.path)
        return _error_response(422, "invalid_request", jsonable_encoder(exc.errors()))

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        logger.info("%s %s -> %s", request.method, request.url.path, exc.status_code)
        response = _error_response(exc.status_code, "http_error", exc.detail)
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Log the full trace server-side; return an opaque message to the client.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _error_response(500, "internal_error", "Internal server error")
