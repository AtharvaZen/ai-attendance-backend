"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8000

Swagger UI: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Make the backend directory (the parent of the `app` package) importable
# before the absolute `app.*` imports below. When this file is executed
# directly - `python app/main.py` - Python puts `backend/app/` on sys.path
# instead of `backend/`, which makes every `app.*` import fail with
# ModuleNotFoundError. This is the same bootstrap used by scripts/*.py.
# It is a no-op when the app is imported as `app.main` from `backend/`.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import __version__  # noqa: E402
from app.api import attendance, auth, health, recognition, students  # noqa: E402
from app.core.config import configure_logging, settings  # noqa: E402
from app.core.errors import register_exception_handlers  # noqa: E402
from app.database.mongo_connection import check_database, close_client, init_db  # noqa: E402
from app.database.repository import get_repository  # noqa: E402
from app.services.face_service import FaceModelError, face_service  # noqa: E402

logger = logging.getLogger(__name__)

TAGS_METADATA = [
    {"name": "health", "description": "Liveness, storage and face-model readiness."},
    {
        "name": "students",
        "description": (
            "Student registry and face registration. Embedding vectors are "
            "sensitive biometric data and are **never** returned by the API."
        ),
    },
    {
        "name": "recognition",
        "description": (
            "Recognise students in a classroom photo. Read-only: never writes "
            "attendance."
        ),
    },
    {
        "name": "attendance",
        "description": (
            "Process classroom photos into attendance and correct records "
            "manually. Every change is audited."
        ),
    },
]

DESCRIPTION = """
Backend for an AI-based student attendance system.

Faces are represented with **pretrained ArcFace embeddings** (InsightFace
`buffalo_l`) - no neural network is trained here, only inference.

### About the scores
Every `confidence` / `similarity` value is a **cosine similarity between
embeddings**. It is **not** a probability of identity: `0.93` does not mean
"93 % likely to be this student". Accept/reject is a single configurable
threshold, `FACE_MATCH_THRESHOLD`.

### Authentication
Send `X-API-Key`. Management endpoints require the **admin** key; recognition
and attendance accept the **teacher** key. With no keys configured the API runs
unauthenticated - development only (`ENVIRONMENT=production` refuses to start).

### Errors
Failures always return `{"detail": ..., "code": ...}`; internal stack traces are
never exposed.
"""


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the pretrained model (once) and prepare storage."""
    configure_logging()
    settings.validate_runtime()
    logger.info("Starting %s (%s)", settings.app_name, settings.environment)
    started = time.perf_counter()

    try:
        face_service.load()  # loaded exactly once, reused for every request
    except FaceModelError:
        logger.critical("Face recognition model initialization failed", exc_info=True)
        raise

    repository = get_repository()  # logs which backend was selected
    if settings.uses_mongodb:
        try:
            init_db()
        except Exception:
            logger.critical("Database initialisation failed", exc_info=True)
            raise

    settings.upload_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Startup complete in %.0f ms | model=%s | providers=%s | storage=%s | auth=%s",
        (time.perf_counter() - started) * 1000,
        settings.face_model_name,
        face_service.providers,
        repository.backend,
        "enabled" if settings.auth_enabled else "DISABLED (development)",
    )
    yield

    if settings.uses_mongodb:
        close_client()
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# The Next.js frontend calls this API directly from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uniform error contract: {"detail": ..., "code": ...}, no leaked tracebacks.
register_exception_handlers(app)

app.include_router(health.router)
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(students.router, prefix=settings.api_prefix)
app.include_router(recognition.router, prefix=settings.api_prefix)
app.include_router(attendance.router, prefix=settings.api_prefix)


if __name__ == "__main__":
    # Convenience entry point: `python app/main.py`. The supported way to run
    # the app is still `uvicorn app.main:app` from the backend directory.
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
