"""Attendance request/response schemas.

JSON key mapping
----------------
The documented contract uses ``"date"`` and ``"class"``. ``date`` and ``class``
are unusable as Python/SQL identifiers, so the fields are named
``attendance_date`` and ``class_name`` internally and exposed under the
documented keys via ``serialization_alias``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AttendanceStatus = Literal["present", "absent", "late", "excused"]


class AttendanceEntry(BaseModel):
    """One student marked in an attendance run."""

    student_id: int
    roll_number: str
    name: str
    status: str
    confidence: float = Field(
        description=(
            "Cosine similarity of the match that produced this record - a "
            "confidence/similarity score, **not** a probability of identity."
        )
    )


class AttendanceProcessResponse(BaseModel):
    """Result of processing a classroom photo for attendance."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "date": "2026-10-01",
                "class": "8",
                "section": "A",
                "total_faces": 32,
                "recognized_students": 30,
                "unknown_faces": 2,
                "duplicates_removed": 0,
                "already_present": 0,
                "skipped_out_of_class": 0,
                "threshold": 0.45,
                "processing_time_ms": 1240.7,
                "attendance": [
                    {
                        "student_id": 101,
                        "roll_number": "101",
                        "name": "Rahul Sharma",
                        "status": "present",
                        "confidence": 0.94,
                    }
                ],
            }
        },
    )

    attendance_date: date = Field(serialization_alias="date")
    class_name: str = Field(serialization_alias="class")
    section: str
    total_faces: int = Field(description="Usable faces detected in the classroom photo.")
    recognized_students: int
    unknown_faces: int
    duplicates_removed: int = Field(
        default=0, description="Extra appearances of a student already recognised."
    )
    already_present: int = Field(
        default=0, description="Students who already had a record for this date."
    )
    skipped_out_of_class: int = Field(
        default=0,
        description="Recognised students who do not belong to this class/section.",
    )
    threshold: float
    processing_time_ms: float
    attendance: list[AttendanceEntry]

    # --- added for the teacher review / confirm workflow --------------------
    # Everything below is additive, so existing clients keep working unchanged.
    preview: bool = Field(
        default=False,
        description="True when nothing was written and the roster awaits confirmation.",
    )
    roster: list["RosterEntryResponse"] = Field(
        default_factory=list,
        description=(
            "Every registered student of this class with their proposed status, so "
            "the teacher can review present **and** absent students before confirming."
        ),
    )
    absent_count: int = Field(
        default=0, description="How many roster entries are not present."
    )
    already_recorded: bool = Field(
        default=False,
        description="True when attendance already exists for this class and date.",
    )


class RosterEntryResponse(BaseModel):
    """One student of the reviewed class and their proposed status."""

    student_id: int
    roll_number: str
    name: str
    status: str = Field(
        description="Proposed attendance status: `present` or `absent`."
    )
    confidence: float | None = Field(
        default=None,
        description=(
            "Cosine similarity when the model recognised this face, otherwise "
            "null. A confidence/similarity score, **not** a probability."
        ),
    )
    recognised: bool = Field(
        description="True when the student's face was detected in the classroom photo."
    )


class ConfirmEntryRequest(BaseModel):
    """One teacher-approved row. Sent back from the review screen."""

    student_id: int = Field(ge=1)
    status: AttendanceStatus
    confidence: float | None = Field(
        default=None,
        description="Similarity of the original match; null for a manual decision.",
    )


class AttendanceConfirmRequest(BaseModel):
    """Body of POST /api/attendance/confirm."""

    class_name: str = Field(min_length=1, max_length=40, description="e.g. `8`.")
    section: str = Field(min_length=1, max_length=10, description="e.g. `A`.")
    attendance_date: date = Field(
        serialization_alias="date", description="Class date, `YYYY-MM-DD`."
    )
    entries: list[ConfirmEntryRequest] = Field(
        min_length=1, description="The teacher-approved rows to persist."
    )

    model_config = ConfigDict(populate_by_name=True)


class AttendanceRecordResponse(BaseModel):
    """A stored attendance row."""

    id: int
    student_id: int
    roll_number: str
    name: str
    class_name: str = Field(serialization_alias="class")
    section: str
    attendance_date: date = Field(serialization_alias="date")
    status: str
    confidence: float | None = Field(
        default=None, description="Similarity score, or null for manually created rows."
    )
    created_at: datetime
    updated_at: datetime


class AttendanceSummary(BaseModel):
    total: int
    present: int
    absent: int
    late: int
    excused: int


class AttendanceListResponse(BaseModel):
    class_name: str = Field(serialization_alias="class")
    section: str
    attendance_date: date | None = Field(default=None, serialization_alias="date")
    summary: AttendanceSummary
    records: list[AttendanceRecordResponse]


class AttendanceUpdateRequest(BaseModel):
    """Manual correction of one attendance row (audited)."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "late"}}
    )

    status: AttendanceStatus


class AuditEntryResponse(BaseModel):
    """One entry from the immutable attendance audit trail."""

    id: int
    attendance_id: int | None
    student_id: int
    action: str = Field(description="`created` or `manual_update`.")
    old_status: str | None
    new_status: str | None
    actor: str = Field(description="API key role that performed the change.")
    created_at: datetime


class AuditListResponse(BaseModel):
    total: int
class AttendanceConfirmResponse(BaseModel):
    """Result of persisting a teacher-approved roster."""

    # `class` is a Python keyword, so it is exposed via an alias - matching the
    # documented wire key used by the other attendance responses.
    model_config = ConfigDict(populate_by_name=True)

    attendance_date: date = Field(serialization_alias="date")
    class_name: str = Field(serialization_alias="class")
    section: str
    saved: int = Field(description="Rows written for this confirmation.")
    created: int = Field(description="Rows newly created.")
    skipped: int = Field(
        description=(
            "Rows not written because they already existed or belonged to "
            "another class."
        )
    )
    already_present: int = Field(
        default=0, description="Rows that already existed for this class and date."
    )
    records: list[AttendanceRecordResponse] = Field(default_factory=list)
    entries: list[AuditEntryResponse]
