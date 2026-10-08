"""MongoDB connection management, index bootstrap and health probing.

A single :class:`~pymongo.MongoClient` is created lazily and reused for the
whole process - PyMongo maintains its own connection pool, so this is safe and
cheap to share across threads and requests. No route ever creates a client.

Vector search note
------------------
``$vectorSearch`` needs MongoDB Atlas (or self-managed ``mongot``). When the
configured deployment does not support it we log that once and the repository
falls back to exact cosine similarity in Python. Both paths return identical
results, so behaviour never depends on the deployment type.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.core.config import settings

logger = logging.getLogger(__name__)

SCHOOLS = "schools"
STUDENTS = "students"
FACE_EMBEDDINGS = "face_embeddings"
ATTENDANCE = "attendance"
ATTENDANCE_AUDIT = "attendance_audit"
COUNTERS = "counters"

_client: MongoClient | None = None
_lock = threading.Lock()
#: True when the deployment actually supports ``$vectorSearch``.
_vector_search_supported: bool | None = None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def mask_uri(uri: str) -> str:
    """Return the URI with any embedded credentials hidden - safe for logs."""
    if "@" not in uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    credentials, _, host = rest.rpartition("@")
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def get_client() -> MongoClient:
    """Return the process-wide client, creating it on first use."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                if not settings.uses_mongodb:
                    raise RuntimeError("MONGODB_URI is not configured")
                _client = MongoClient(
                    settings.mongodb_uri,
                    serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
                    tz_aware=True,
                )
                logger.info(
                    "MongoDB client created for %s", mask_uri(settings.mongodb_uri)
                )
    return _client


def get_database() -> Database:
    """Return the configured application database."""
    return get_client()[settings.mongodb_database]


def collection(name: str) -> Collection:
    """Return one collection of the application database."""
    return get_database()[name]


# ---------------------------------------------------------------------------
# Schema / indexes (idempotent, never destructive)
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create the collections and every index the repository relies on.

    Safe to run repeatedly - index creation is idempotent, and nothing is ever
    dropped or deleted here.
    """
    database = get_database()

    # Touch each collection so it exists even before the first write.
    for name in (
        SCHOOLS,
        STUDENTS,
        FACE_EMBEDDINGS,
        ATTENDANCE,
        ATTENDANCE_AUDIT,
        COUNTERS,
    ):
        database[name]

    # students: a roll number is only unique *within* a school.
    collection(STUDENTS).create_index(
        [("schoolId", ASCENDING)], name="ix_students_school"
    )
    collection(STUDENTS).create_index(
        [("schoolId", ASCENDING), ("rollNumber", ASCENDING)],
        name="uq_students_school_roll",
        unique=True,
    )

    # face_embeddings: one document per registered photo.
    collection(FACE_EMBEDDINGS).create_index(
        [("schoolId", ASCENDING)], name="ix_face_embeddings_school"
    )
    collection(FACE_EMBEDDINGS).create_index(
        [("studentId", ASCENDING)], name="ix_face_embeddings_student"
    )

    # attendance: at most one row per student / class / section / date.
    collection(ATTENDANCE).create_index(
        [("schoolId", ASCENDING)], name="ix_attendance_school"
    )
    collection(ATTENDANCE).create_index(
        [("studentId", ASCENDING)], name="ix_attendance_student"
    )
    collection(ATTENDANCE).create_index(
        [("attendanceDate", ASCENDING)], name="ix_attendance_date"
    )
    collection(ATTENDANCE).create_index(
        [
            ("schoolId", ASCENDING),
            ("studentId", ASCENDING),
            ("className", ASCENDING),
            ("section", ASCENDING),
            ("attendanceDate", ASCENDING),
        ],
        name="uq_attendance_student_class_date",
        unique=True,
    )

    collection(ATTENDANCE_AUDIT).create_index(
        [("attendanceId", ASCENDING)], name="ix_audit_attendance"
    )
    collection(ATTENDANCE_AUDIT).create_index(
        [("studentId", ASCENDING)], name="ix_audit_student"
    )

    _ensure_vector_index()
    _ensure_school()
    logger.info("MongoDB schema is up to date (database=%r)", settings.mongodb_database)


def _ensure_vector_index() -> None:
    """Create the Atlas vector index when the deployment supports it."""
    global _vector_search_supported
    definitions = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": settings.embedding_dim,
                "similarity": "cosine",
            },
            # Filtering by school inside the vector search keeps every search
            # scoped to a single school's embeddings.
            {"type": "filter", "path": "schoolId"},
        ]
    }

    try:
        collection(FACE_EMBEDDINGS).create_search_index(  # type: ignore[attr-defined]
            {"name": "vector_index", "type": "vectorSearch", "definition": definitions}
        )
    except Exception as exc:  # noqa: BLE001 - purely deployment dependent
        _vector_search_supported = False
        logger.warning(
            "MongoDB vector search is unavailable on this deployment (%s). "
            "Falling back to exact cosine similarity, which returns identical "
            "results at this dataset size.",
            type(exc).__name__,
        )
    else:
        _vector_search_supported = True
        logger.info(
            "MongoDB vector index ready (dim=%d, cosine similarity, filtered by schoolId)",
            settings.embedding_dim,
        )


def _ensure_school() -> None:
    """Make sure the configured school exists (created on first boot only)."""
    collection(SCHOOLS).update_one(
        {"_id": settings.school_id},
        {
            "$setOnInsert": {"name": settings.school_name, "createdAt": utcnow()},
            "$set": {"updatedAt": utcnow()},
        },
        upsert=True,
    )


def vector_search_supported() -> bool:
    """Whether the deployment supports ``$vectorSearch`` (False = fallback)."""
    return bool(_vector_search_supported)


def check_database() -> tuple[bool, str]:
    """Return ``(reachable, human_readable_status)`` for the health endpoint."""
    if not settings.uses_mongodb:
        return False, "in-memory (MONGODB_URI not set)"
    try:
        info = get_client().server_info()
        return True, f"MongoDB {info.get('version', 'unknown')} ({settings.mongodb_database})"
    except PyMongoError as exc:
        return False, f"unreachable: {exc}"


def close_client() -> None:
    """Release the pooled connections (used on shutdown and by tests)."""
    global _client, _vector_search_supported
    if _client is not None:
        _client.close()
        logger.info("MongoDB client closed")
    _client = None
    _vector_search_supported = None