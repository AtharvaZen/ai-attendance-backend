"""Repository interface + backend factory.

Two interchangeable backends:

* :class:`~app.database.mongo_repository.MongoAttendanceRepository` - MongoDB,
  used whenever ``MONGODB_URI`` is set (the persistent deployment).
* :class:`~app.database.memory_repository.InMemoryAttendanceRepository` -
  process-local dictionaries so the API can be exercised without a database
  (development / demo only).

Services and routers only ever see this interface plus the plain dataclasses in
:mod:`app.database.records`, never a database driver or session.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date

import numpy as np

from app.core.config import settings
from app.database.records import (
    AttendanceRecord,
    AuditRecord,
    FaceRecord,
    StudentRecord,
)

logger = logging.getLogger(__name__)


class AttendanceRepository(ABC):
    """Storage operations required by the domain services."""

    #: Short backend name, surfaced by ``/health``.
    backend: str = "abstract"

    # -- students ----------------------------------------------------------
    @abstractmethod
    def create_student(
        self, *, roll_number: str, name: str, class_name: str, section: str
    ) -> StudentRecord: ...

    @abstractmethod
    def get_student(self, student_id: int) -> StudentRecord | None: ...

    @abstractmethod
    def get_student_by_roll(
        self, roll_number: str, class_name: str, section: str
    ) -> StudentRecord | None: ...

    @abstractmethod
    def list_students(
        self,
        *,
        class_name: str | None = None,
        section: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StudentRecord]: ...

    @abstractmethod
    def count_students(
        self, *, class_name: str | None = None, section: str | None = None
    ) -> int: ...

    @abstractmethod
    def update_student(
        self,
        student_id: int,
        *,
        name: str | None = None,
        roll_number: str | None = None,
        class_name: str | None = None,
        section: str | None = None,
    ) -> StudentRecord | None: ...

    @abstractmethod
    def delete_student(self, student_id: int) -> bool: ...

    @abstractmethod
    def list_classes(self) -> list[tuple[str, str, int]]:
        """Return ``(class_name, section, student_count)`` triples."""

    # -- face embeddings ---------------------------------------------------
    @abstractmethod
    def add_face(
        self, student_id: int, embedding: np.ndarray, det_score: float
    ) -> FaceRecord: ...

    @abstractmethod
    def list_faces(self, student_id: int) -> list[FaceRecord]: ...

    @abstractmethod
    def count_faces(self, student_id: int) -> int: ...

    @abstractmethod
    def delete_face(self, face_id: int) -> bool: ...

    @abstractmethod
    def delete_faces(self, student_id: int) -> int:
        """Delete all biometric data for a student. Returns rows removed."""

    @abstractmethod
    def search_similar(
        self,
        embedding: np.ndarray,
        *,
        limit: int = 5,
        exclude_student_id: int | None = None,
        class_name: str | None = None,
        section: str | None = None,
    ) -> list[tuple[FaceRecord, float]]:
        """Return ``(face, cosine_similarity)`` sorted by similarity desc.

        When ``class_name``/``section`` are given the candidate set is restricted
        to that class **inside the query**, so a face can never match a student
        from another class.
        """

    @abstractmethod
    def count_embeddings(self) -> int: ...

    # -- attendance --------------------------------------------------------
    @abstractmethod
    def get_attendance(self, attendance_id: int) -> AttendanceRecord | None: ...

    @abstractmethod
    def find_attendance(
        self,
        *,
        student_id: int,
        class_name: str,
        section: str,
        attendance_date: date,
    ) -> AttendanceRecord | None: ...

    @abstractmethod
    def upsert_attendance(
        self,
        *,
        student_id: int,
        class_name: str,
        section: str,
        attendance_date: date,
        status: str,
        confidence: float | None,
        actor: str,
    ) -> tuple[AttendanceRecord, bool]:
        """Insert if absent; return ``(record, created)``. Never duplicates."""

    @abstractmethod
    def update_attendance(
        self, attendance_id: int, *, status: str, actor: str
    ) -> tuple[AttendanceRecord | None, str | None]:
        """Manually correct a status; return ``(record, previous_status)``."""

    @abstractmethod
    def list_attendance(
        self,
        *,
        class_name: str,
        section: str,
        attendance_date: date | None = None,
    ) -> list[AttendanceRecord]: ...

    @abstractmethod
    def list_audit(
        self, *, attendance_id: int | None = None, limit: int = 100
    ) -> list[AuditRecord]: ...

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def reset(self) -> None:
        """Drop all data (tests / demo only)."""


_repository: AttendanceRepository | None = None


def build_repository() -> AttendanceRepository:
    """Instantiate the backend selected by configuration."""
    if settings.uses_mongodb:
        from app.database.mongo_repository import MongoAttendanceRepository

        logger.info(
            "Storage backend: MongoDB (school=%s)", settings.school_id
        )
        return MongoAttendanceRepository()

    from app.database.memory_repository import InMemoryAttendanceRepository

    logger.warning(
        "MONGODB_URI is not set - using the IN-MEMORY repository. Data is lost "
        "on restart and is not shared between workers. Development/demo only."
    )
    return InMemoryAttendanceRepository()


def get_repository() -> AttendanceRepository:
    """Return the process-wide repository singleton."""
    global _repository
    if _repository is None:
        _repository = build_repository()
    return _repository


def set_repository(repository: AttendanceRepository | None) -> None:
    """Override the singleton (used by tests and scripts)."""
    global _repository
    _repository = repository
