"""Application configuration and logging.

Every tunable value is read from environment variables (or a local ``.env``
file), so no secret or credential is ever hard-coded in the source tree.

Environment variable names are the upper-case form of the field name, e.g.
``face_match_threshold`` -> ``FACE_MATCH_THRESHOLD``.
"""

from __future__ import annotations

import logging
import logging.config
import sys
import warnings
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# <repo>/backend  (this file lives at backend/app/core/config.py)
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings, validated once at import time."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- general -----------------------------------------------------------
    app_name: str = "AI Attendance System"
    environment: str = "development"
    api_prefix: str = "/api"
    log_level: str = "INFO"

    # --- database (MongoDB) --------------------------------------------------
    # Empty MONGODB_URI => the in-memory repository is used (development / demo
    # only). Never hard-code credentials: the URI always comes from the
    # environment / `.env`.
    mongodb_uri: str = ""
    mongodb_database: str = "attendance"
    mongodb_server_selection_timeout_ms: int = Field(default=5000, ge=100)

    # Identifier of the school whose data this process serves. Every student,
    # face embedding and attendance row is scoped to it, and it is applied as a
    # filter inside the vector search - a school can therefore never match
    # another school's students.
    school_id: str = "default"
    school_name: str = "Default School"

    # Embedding dimension of the configured face model. This is what the current
    # pretrained model (InsightFace buffalo_l / ArcFace) produces; if the face
    # model is ever replaced, change it here and re-run scripts/init_mongo.py so
    # the MongoDB vector index is recreated with the new dimension.
    embedding_dim: int = Field(default=512, ge=1)

    # --- API security ------------------------------------------------------
    # Leave both empty to disable auth (development only - refused when
    # ENVIRONMENT=production).
    admin_api_key: str = ""
    teacher_api_key: str = ""
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- login accounts (single-school, one admin + one teacher) -----------
    # Credentials for POST /api/auth/login. They are development accounts read
    # from the environment; nothing is hard-coded and `.env` is never committed.
    admin_username: str = "admin"
    admin_password: str = ""
    teacher_username: str = "teacher"
    teacher_password: str = ""
    # Signs the session token issued by the login endpoint.
    auth_secret: str = ""
    session_ttl_minutes: int = Field(default=480, ge=1)

    # When true, a session token is also accepted for every protected route.
    # Static API keys always keep working so existing clients/tests are not
    # broken; set this to false to force login-only access.
    allow_api_key_auth: bool = True

    # --- file storage ------------------------------------------------------
    upload_dir: Path = Path("uploads")
    max_upload_size_mb: int = Field(default=10, ge=1)

    # --- face recognition (pretrained, inference only) ---------------------
    face_model_name: str = "buffalo_l"
    face_model_root: str = "~/.insightface"
    # Cosine-similarity threshold. NOT a probability - see README.
    face_match_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    # Identification margin: minimum gap between the best candidate (a student)
    # and the best candidate from a *different* student. When the gap is smaller
    # than this, the face is treated as UNKNOWN even if it clears the threshold -
    # a look-alike scoring e.g. 0.51 vs 0.49 must not be accepted. 0 disables
    # the margin gate (threshold-only behaviour). Starting value 0.05, tune on
    # your own data with scripts/evaluate_recognition.py.
    face_match_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    face_det_size: int = Field(default=640, ge=160)
    face_det_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    face_min_face_size: int = Field(default=60, ge=16)
    # Comma-separated ONNX Runtime providers. Empty => automatic selection.
    face_providers: str = ""

    # --- recognition / registration ---------------------------------------
    # How many candidate embeddings to fetch from the vector index per face.
    # Keep this comfortably above faces-per-student-max so one heavily
    # registered student cannot fill the whole window and hide the runner-up
    # needed for the margin check.
    recognition_candidates: int = Field(default=15, ge=1)
    max_faces_per_image: int = Field(default=300, ge=1)
    faces_per_student_min: int = Field(default=3, ge=1)
    faces_per_student_max: int = Field(default=10, ge=1)
    # A new registration photo at least this similar to one already stored for
    # the same student is rejected as a duplicate. Keep this VERY HIGH: photos
    # taken back-to-back of one still student score 0.85-0.97, and re-uploading
    # the very same file scores ~1.00. This guard exists to catch accidental
    # re-uploads, not to judge whether two sittings were distinct enough.
    duplicate_face_similarity: float = Field(default=0.98, ge=0.0, le=1.0)

    # --- derived values ----------------------------------------------------
    @property
    def upload_path(self) -> Path:
        """Absolute path of the upload directory (created on demand)."""
        path = self.upload_dir
        return path if path.is_absolute() else (BACKEND_DIR / path)

    @property
    def face_model_root_path(self) -> Path:
        """Absolute path of the InsightFace model cache (``~`` expanded)."""
        return Path(self.face_model_root).expanduser()

    @property
    def provider_list(self) -> list[str]:
        return [p.strip() for p in self.face_providers.split(",") if p.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.admin_api_key or self.teacher_api_key)

    @property
    def login_enabled(self) -> bool:
        """True when both login accounts have a password configured."""
        return bool(self.admin_password.strip() and self.teacher_password.strip())

    @property
    def uses_mongodb(self) -> bool:
        return bool(self.mongodb_uri.strip())

    def validate_runtime(self) -> None:
        """Fail fast on an unsafe production configuration."""
        if self.environment == "production" and not self.auth_enabled:
            raise RuntimeError(
                "Refusing to start: ENVIRONMENT=production requires "
                "ADMIN_API_KEY and/or TEACHER_API_KEY to be set."
            )
        if self.environment == "production" and not self.uses_mongodb:
            raise RuntimeError(
                "Refusing to start: ENVIRONMENT=production requires MONGODB_URI "
                "(the in-memory store is development-only)."
            )

    @field_validator("mongodb_database")
    @classmethod
    def _valid_database_name(cls, value: str) -> str:
        """MongoDB forbids spaces, '/', '\\', '.', '\"' and '$' in db names.

        Catching it here turns a deep PyMongo stack trace into one clear line.
        """
        name = (value or "").strip()
        if not name:
            raise ValueError("MONGODB_DATABASE must not be empty")
        bad = set(' /\\."$') & set(name)
        if bad:
            raise ValueError(
                f"MONGODB_DATABASE must not contain {sorted(bad)!r} "
                f"(got {name!r}). Use underscores instead, e.g. 'ai_attendance'."
            )
        return name

    @field_validator("environment")
    @classmethod
    def _normalise_environment(cls, value: str) -> str:
        return (value or "development").strip().lower()

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        return (value or "INFO").strip().upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def configure_logging() -> None:
    """Configure application logging (idempotent)."""
    settings = get_settings()

    # InsightFace's face alignment calls a scikit-image API that emits a
    # cosmetic FutureWarning on every aligned face - keep the logs clean.
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r"insightface(\.|$)"
    )

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "standard",
                }
            },
            "root": {"handlers": ["console"], "level": settings.log_level},
            "loggers": {
                "app": {"level": settings.log_level, "propagate": True},
                # InsightFace is chatty about ONNX sessions; downgrade to WARNING.
                "insightface": {"level": "WARNING", "propagate": True},
            },
        }
    )


settings = get_settings()
