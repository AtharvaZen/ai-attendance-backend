"""Face detection + embedding service.

Wraps a single pretrained **InsightFace** ``FaceAnalysis`` pipeline:
SCRFD face detector + ArcFace-compatible ``buffalo_l`` recognition model.

IMPORTANT
---------
* NO neural network is trained anywhere in this project. We only run
  *inference* with ready-made pretrained weights.
* The model is loaded exactly **once**, when the FastAPI application starts,
  and is reused for every request.
* Embeddings returned by :meth:`FaceService.detect` are always L2-normalised,
  so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class FaceModelError(RuntimeError):
    """Raised when the pretrained face model cannot be initialised or used."""


# ---------------------------------------------------------------------------
# Small, dependency-free helpers (kept here because they belong to the face
# pipeline; they avoid an extra abstraction layer).
# ---------------------------------------------------------------------------
def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` scaled to unit L2 norm (float32, 1-D)."""
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalise a zero-length embedding")
    return (vector / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised embeddings (plain dot product)."""
    return float(
        np.dot(
            np.asarray(a, dtype=np.float32).ravel(),
            np.asarray(b, dtype=np.float32).ravel(),
        )
    )


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (BGR, uint8) or raise :class:`ValueError`."""
    import cv2

    if not data:
        raise ValueError("Empty image payload")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image: unsupported or corrupt format")
    return image


def read_image_file(path: str | Path) -> np.ndarray:
    """Read an image from disk (unicode-safe on Windows) as BGR uint8."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return decode_image_bytes(path.read_bytes())


def write_image_file(path: str | Path, image: np.ndarray) -> Path:
    """Write a BGR image to disk (unicode-safe on Windows)."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Could not encode image for {path}")
    buffer.tofile(str(path))
    return path


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class DetectedFace:
    """One detected face plus its (normalised) embedding."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    det_score: float                          # detector confidence (NOT identity prob.)
    embedding: np.ndarray | None              # L2-normalised, shape (512,)
    raw: object | None = field(default=None, repr=False, compare=False)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class FaceService:
    """Thread-safe wrapper around one shared InsightFace pipeline."""

    def __init__(self) -> None:
        self._app: object | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()  # ONNX sessions: serialise calls
        self.providers: list[str] = []
        self.loaded_at: float | None = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._app is not None

    def load(self) -> None:
        """Load the pretrained model once. Safe to call repeatedly."""
        if self._app is not None:
            return
        with self._load_lock:
            if self._app is not None:  # another thread won the race
                return
            started = time.perf_counter()
            self._app = self._build_pipeline()
            self.loaded_at = time.time()
            logger.info(
                "Pretrained face model '%s' loaded in %.0f ms | providers=%s",
                settings.face_model_name,
                (time.perf_counter() - started) * 1000,
                self.providers,
            )

    def _build_pipeline(self):
        """Instantiate + prepare ``insightface.app.FaceAnalysis``."""
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:  # pragma: no cover - environment problem
            raise FaceModelError(
                "Face recognition model initialization failed: "
                f"InsightFace could not be imported ({exc})"
            ) from exc

        kwargs: dict[str, object] = {
            "name": settings.face_model_name,
            "root": str(settings.face_model_root_path),
        }
        if settings.provider_list:
            kwargs["providers"] = settings.provider_list

        try:
            app = FaceAnalysis(**kwargs)
        except TypeError:
            # Signature without the ``providers`` argument.
            kwargs.pop("providers", None)
            try:
                app = FaceAnalysis(**kwargs)
            except Exception as exc:
                raise FaceModelError(
                    "Face recognition model initialization failed: "
                    f"could not create FaceAnalysis ({exc})"
                ) from exc
        except Exception as exc:
            raise FaceModelError(
                "Face recognition model initialization failed: "
                f"could not create FaceAnalysis ({exc}). The first run downloads "
                "the model weights, so check your network connection and that "
                "FACE_MODEL_ROOT is writable."
            ) from exc

        size = (settings.face_det_size, settings.face_det_size)
        # ``prepare``'s signature differs slightly across InsightFace versions,
        # so try the richest call first and fall back gracefully.
        for attempt in (
            {"ctx_id": 0, "det_size": size, "det_thresh": settings.face_det_thresh},
            {"det_size": size, "det_thresh": settings.face_det_thresh},
            {"ctx_id": 0, "det_size": size},
            {},
        ):
            try:
                app.prepare(**attempt)
                break
            except TypeError:
                continue
            except Exception as exc:
                raise FaceModelError(
                    "Face recognition model initialization failed: "
                    f"prepare() failed ({exc})"
                ) from exc

        self.providers = self._read_providers(app)
        return app

    @staticmethod
    def _read_providers(app) -> list[str]:
        try:
            return list(app.models["recognition"].session.get_providers())
        except Exception:  # pragma: no cover - informational only
            return settings.provider_list or ["unknown"]

    def _require_app(self):
        if self._app is None:
            raise FaceModelError(
                "Face recognition model is not loaded. It is initialised during "
                "application startup."
            )
        return self._app

    # -- detection ---------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        *,
        min_face_size: int | None = None,
        require_embedding: bool = True,
    ) -> list[DetectedFace]:
        """Detect every face in ``image`` and attach its normalised embedding.

        Faces whose smaller side is below ``min_face_size`` (default
        ``FACE_MIN_FACE_SIZE``) are dropped as too small / low quality.
        Results are sorted by face area, largest first.
        """
        app = self._require_app()
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid image")

        threshold_px = (
            settings.face_min_face_size if min_face_size is None else min_face_size
        )

        with self._inference_lock:
            raw_faces = list(app.get(image) or [])

        faces: list[DetectedFace] = []
        skipped_small = 0
        for raw in raw_faces:
            x1, y1, x2, y2 = (float(v) for v in raw.bbox[:4])
            face = DetectedFace(
                bbox=(x1, y1, x2, y2),
                det_score=float(getattr(raw, "det_score", 0.0)),
                embedding=None,
                raw=raw,
            )
            if face.min_side < threshold_px:
                skipped_small += 1
                continue
            vector = getattr(raw, "normed_embedding", None)
            if vector is None:
                vector = getattr(raw, "embedding", None)
            if vector is None:
                if require_embedding:
                    logger.warning("Detected face without embedding - skipped")
                    continue
            else:
                face.embedding = l2_normalize(vector)
            faces.append(face)

        faces.sort(key=lambda f: f.area, reverse=True)
        logger.info(
            "Face detection: detected=%d usable=%d skipped_small=%d",
            len(raw_faces),
            len(faces),
            skipped_small,
        )
        return faces

    def annotate(self, image: np.ndarray, faces: list[DetectedFace]) -> np.ndarray:
        """Return a copy of ``image`` with boxes/landmarks drawn (debug aid)."""
        app = self._require_app()
        raw_faces = [f.raw for f in faces if f.raw is not None]
        return app.draw_on(image.copy(), raw_faces)


# Module-level singleton used by the API and scripts.
face_service = FaceService()
