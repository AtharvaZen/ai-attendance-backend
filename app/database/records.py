"""Plain data records exchanged between the storage layer and the services.

Deliberately framework-free (dataclasses, never database-driver objects) so
routers and services never touch a database collection or session.

``FaceRecord.embedding`` holds sensitive biometric data: it must never be
serialised into an API response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np

# Supported attendance statuses (the teacher can correct to any of these).
STATUS_PRESENT = "present"
STATUS_ABSENT = "absent"
STATUS_LATE = "late"
STATUS_EXCUSED = "excused"
ATTENDANCE_STATUSES: tuple[str, ...] = (
    STATUS_PRESENT,
    STATUS_ABSENT,
    STATUS_LATE,
    STATUS_EXCUSED,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class StudentRecord:
    id: int
    roll_number: str
    name: str
    class_name: str
    section: str
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    face_count: int = 0


@dataclass(slots=True)
class FaceRecord:
    id: int
    student_id: int
    embedding: np.ndarray  # L2-normalised float32, shape (512,)
    det_score: float
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class AttendanceRecord:
    id: int
    student_id: int
    class_name: str
    section: str
    attendance_date: date
    status: str
    confidence: float | None
    created_at: datetime
    updated_at: datetime
    roll_number: str = ""
    student_name: str = ""


@dataclass(slots=True)
class AuditRecord:
    id: int
    attendance_id: int | None
    student_id: int
    action: str
    old_status: str | None
    new_status: str | None
    actor: str
    confidence: float | None = None
    created_at: datetime = field(default_factory=utcnow)
