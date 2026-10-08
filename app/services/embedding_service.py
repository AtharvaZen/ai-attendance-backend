"""Turns images into validated face embeddings.

This module owns the **registration rules**:

* exactly one usable face per registration photo,
* reject zero faces,
* reject multiple faces,
* reject faces that are too small / low quality,

all raised as clean :mod:`app.core.errors` exceptions.

No training happens anywhere in this project - the ArcFace weights are
pretrained and frozen; we only run inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.core.errors import (
    FaceTooSmallError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.services.face_service import DetectedFace, FaceService, face_service

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RegistrationFace:
    """One validated face, ready to be stored as a student embedding."""

    embedding: np.ndarray
    det_score: float
    bbox: tuple[float, float, float, float]
    min_side: float


class EmbeddingService:
    """Thin policy layer over :class:`~app.services.face_service.FaceService`."""

    def __init__(self, faces: FaceService | None = None) -> None:
        self._faces = faces or face_service

    def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect every usable face; raise if the image contains none."""
        faces = self._faces.detect(image)
        if not faces:
            raise NoFaceDetectedError(
                "No face detected. Use a clear, well-lit photo containing a face."
            )
        if len(faces) > settings.max_faces_per_image:
            logger.warning(
                "Image contains %d faces; only the largest %d are processed",
                len(faces),
                settings.max_faces_per_image,
            )
            faces = faces[: settings.max_faces_per_image]
        return faces

    def single_face_embedding(self, image: np.ndarray) -> RegistrationFace:
        """Enforce the registration rules and return exactly one embedding."""
        # min_face_size=0 so that an oversized-but-small face is reported as
        # FaceTooSmallError rather than as "no face".
        candidates = self._faces.detect(image, min_face_size=0)

        if not candidates:
            raise NoFaceDetectedError(
                "No face detected. Use a clear, well-lit portrait photo."
            )
        if len(candidates) > 1:
            raise MultipleFacesDetectedError(
                f"Multiple faces detected ({len(candidates)}) during registration. "
                "Upload a photo containing exactly one person."
            )

        face = candidates[0]
        if face.min_side < settings.face_min_face_size:
            raise FaceTooSmallError(
                f"Detected face is too small ({face.min_side:.0f}px). "
                f"At least {settings.face_min_face_size}px is required - move closer "
                "or use a higher-resolution photo."
            )
        if face.embedding is None:
            raise NoFaceDetectedError("Could not compute an embedding for the face.")

        return RegistrationFace(
            embedding=face.embedding,
            det_score=face.det_score,
            bbox=face.bbox,
            min_side=face.min_side,
        )


embedding_service = EmbeddingService()
