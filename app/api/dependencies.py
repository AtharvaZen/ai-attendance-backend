"""Reusable FastAPI dependencies: upload validation and model readiness.

Image uploads are size-checked **while streaming**, so an oversized body is
rejected without ever being fully buffered in memory. Clients receive clean
HTTP errors (400 / 413) instead of stack traces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from fastapi import File, UploadFile

from app.core.config import settings
from app.core.errors import InvalidImageError, ModelNotReadyError, PayloadTooLargeError
from app.services.face_service import decode_image_bytes, face_service

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 64 * 1024


@dataclass(slots=True)
class ImageUpload:
    """A validated, decoded image upload."""

    filename: str | None
    content_type: str | None
    size_bytes: int
    image: np.ndarray  # BGR, uint8


async def read_image_upload(
    file: UploadFile = File(..., description="Image file (JPEG/PNG)."),
) -> ImageUpload:
    """Dependency: validate content type, size limit and decodability."""
    content_type = (file.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise InvalidImageError(
            f"Unsupported content type '{file.content_type}'. Upload an image file."
        )

    chunks: list[bytes] = []
    total = 0
    limit = settings.max_upload_size_bytes
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError(
                f"Image exceeds the maximum allowed size of {settings.max_upload_size_mb} MB"
            )
        chunks.append(chunk)

    if total == 0:
        raise InvalidImageError("Empty image payload")

    try:
        image = decode_image_bytes(b"".join(chunks))
    except ValueError as exc:
        raise InvalidImageError(str(exc)) from exc

    logger.info(
        "Received image upload: filename=%s size=%dB content_type=%s shape=%s",
        file.filename,
        total,
        file.content_type,
        image.shape,
    )
    return ImageUpload(
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=total,
        image=image,
    )


def require_face_model() -> None:
    """Dependency: guarantee the pretrained model finished loading."""
    if not face_service.is_loaded:
        raise ModelNotReadyError()
