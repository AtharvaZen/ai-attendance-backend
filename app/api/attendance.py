"""Attendance endpoints.

Creating attendance is kept separate from recognition: this is the only module
that writes attendance rows, and every creation/correction is written to the
audit trail.
"""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, Depends, Form, Path, Query

from app.api.dependencies import ImageUpload, read_image_upload, require_face_model
from app.core.errors import AttendanceNotFoundError, InvalidAttendanceStatusError
from app.core.security import require_teacher
from app.database.records import ATTENDANCE_STATUSES, AttendanceRecord
from app.database.repository import AttendanceRepository, get_repository
from app.schemas.attendance import (
    AttendanceConfirmRequest,
    AttendanceConfirmResponse,
    AttendanceEntry,
    AttendanceListResponse,
    AttendanceProcessResponse,
    AttendanceRecordResponse,
    AttendanceSummary,
    AttendanceUpdateRequest,
    AuditEntryResponse,
    AuditListResponse,
    RosterEntryResponse,
)
from app.schemas.common import ErrorResponse, ValidationErrorResponse
from app.services.attendance_service import AttendanceService, ConfirmedEntry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/attendance", tags=["attendance"])

_AUTH = {"model": ErrorResponse, "description": "`unauthorized` / `forbidden`."}
_INVALID = {"model": ValidationErrorResponse, "description": "`invalid_request`."}


def _to_response(record: AttendanceRecord) -> AttendanceRecordResponse:
    return AttendanceRecordResponse(
        id=record.id,
        student_id=record.student_id,
        roll_number=record.roll_number,
        name=record.student_name,
        class_name=record.class_name,
        section=record.section,
        attendance_date=record.attendance_date,
        status=record.status,
        confidence=record.confidence,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.post(
    "/process",
    response_model=AttendanceProcessResponse,
    summary="Process a classroom photo and mark attendance",
    description=(
        "Full pipeline: face detection -> ArcFace embeddings -> cosine-similarity "
        "search -> duplicate removal -> attendance rows.\n\n"
        "Idempotent: re-processing the same photo for the same class and date "
        "never creates duplicate rows (a unique constraint on student + class + "
        "section + date guarantees it). Students who already had a record are "
        "reported in `already_present`.\n\n"
        "`confidence` values are cosine similarities between embeddings - "
        "**not probabilities** of identity."
    ),
    responses={
        200: {"description": "Attendance outcome."},
        400: {"model": ErrorResponse, "description": "`invalid_image` or `no_face_detected`."},
        401: _AUTH,
        413: {"model": ErrorResponse, "description": "`payload_too_large`."},
        422: _INVALID,
    },
)
async def process_attendance(
    actor: str = Depends(require_teacher),
    model_ready: None = Depends(require_face_model),
    upload: ImageUpload = Depends(read_image_upload),
    class_name: str = Form(..., min_length=1, max_length=40, description="e.g. `8`."),
    section: str = Form(..., min_length=1, max_length=10, description="e.g. `A`."),
    attendance_date: date = Form(..., description="Class date, `YYYY-MM-DD`."),
    dry_run: bool = Form(
        False,
        description=(
            "When true, recognition runs but **nothing is written** - the full "
            "class roster (present + absent) is returned for review before the "
            "teacher confirms via POST /api/attendance/confirm."
        ),
    ),
    repository: AttendanceRepository = Depends(get_repository),
) -> AttendanceProcessResponse:
    outcome = AttendanceService(repository).process(
        upload.image,
        class_name=class_name.strip(),
        section=section.strip(),
        attendance_date=attendance_date,
        actor=actor,
        dry_run=dry_run,
    )

    return AttendanceProcessResponse(
        attendance_date=outcome.attendance_date,
        class_name=outcome.class_name,
        section=outcome.section,
        total_faces=outcome.total_faces,
        recognized_students=outcome.recognized_students,
        unknown_faces=outcome.unknown_faces,
        duplicates_removed=outcome.duplicates_removed,
        already_present=outcome.already_present,
        skipped_out_of_class=outcome.skipped_out_of_class,
        threshold=outcome.threshold,
        processing_time_ms=outcome.processing_time_ms,
        attendance=[
            AttendanceEntry(
                student_id=entry.student_id,
                roll_number=entry.roll_number,
                name=entry.name,
                status=entry.status,
                confidence=entry.confidence,
            )
            for entry in outcome.entries
        ],
        preview=outcome.preview,
        roster=[
            RosterEntryResponse(
                student_id=item.student_id,
                roll_number=item.roll_number,
                name=item.name,
                status=item.status,
                confidence=item.confidence,
                recognised=item.recognised,
            )
            for item in outcome.roster
        ],
        absent_count=len(outcome.absent_students),
        already_recorded=outcome.existing_records > 0,
    )


@router.post(
    "/confirm",
    response_model=AttendanceConfirmResponse,
    summary="Save the teacher-approved attendance",
    description=(
        "Persists the roster the teacher reviewed. Only rows the model "
        "recognised, or that the teacher explicitly confirmed, are written - "
        "students merely missing from the photo are not stored.\n\n"
        "Safe to call twice: the unique index on (school, student, class, "
        "section, date) prevents duplicate rows and existing rows are kept."
    ),
    responses={
        200: {"description": "Attendance saved."},
        401: _AUTH,
        422: _INVALID,
    },
)
async def confirm_attendance(
    payload: AttendanceConfirmRequest,
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_teacher),
) -> AttendanceConfirmResponse:
    outcome = AttendanceService(repository).confirm(
        class_name=payload.class_name.strip(),
        section=payload.section.strip(),
        attendance_date=payload.attendance_date,
        entries=[
            ConfirmedEntry(
                student_id=entry.student_id,
                status=entry.status,
                confidence=entry.confidence,
            )
            for entry in payload.entries
        ],
        actor=actor,
    )

    return AttendanceConfirmResponse(
        attendance_date=outcome.attendance_date,
        class_name=outcome.class_name,
        section=outcome.section,
        saved=outcome.saved,
        created=outcome.created,
        skipped=outcome.skipped,
        already_present=outcome.already_present,
        records=[_to_response(record) for record in outcome.records],
    )


@router.get(
    "",
    response_model=AttendanceListResponse,
    summary="List attendance for a class (optionally by date)",
    responses={
        200: {"description": "Attendance rows plus a status summary."},
        401: _AUTH,
        422: _INVALID,
    },
)
async def list_attendance(
    class_name: str = Query(..., min_length=1, max_length=40, description="e.g. `8`."),
    section: str = Query(..., min_length=1, max_length=10, description="e.g. `A`."),
    attendance_date: date | None = Query(
        default=None, description="Filter by date. Omit to list every date."
    ),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_teacher),
) -> AttendanceListResponse:
    records = repository.list_attendance(
        class_name=class_name.strip(),
        section=section.strip(),
        attendance_date=attendance_date,
    )
    counts: dict[str, int] = {status: 0 for status in ATTENDANCE_STATUSES}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1

    return AttendanceListResponse(
        class_name=class_name.strip(),
        section=section.strip(),
        attendance_date=attendance_date,
        summary=AttendanceSummary(
            total=len(records),
            present=counts["present"],
            absent=counts["absent"],
            late=counts["late"],
            excused=counts["excused"],
        ),
        records=[_to_response(record) for record in records],
    )


@router.get(
    "/{attendance_id}",
    response_model=AttendanceRecordResponse,
    summary="Get one attendance record",
    responses={
        200: {"description": "The attendance row."},
        401: _AUTH,
        404: {"model": ErrorResponse, "description": "`attendance_not_found`."},
        422: _INVALID,
    },
)
async def get_attendance(
    attendance_id: int = Path(..., ge=1),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_teacher),
) -> AttendanceRecordResponse:
    record = repository.get_attendance(attendance_id)
    if record is None:
        raise AttendanceNotFoundError(f"Attendance {attendance_id} not found")
    return _to_response(record)


@router.patch(
    "/{attendance_id}",
    response_model=AttendanceRecordResponse,
    summary="Manually correct an attendance status",
    description=(
        "Lets the teacher override what the model decided. The change is written "
        "to the audit trail with the previous status and the acting role."
    ),
    responses={
        200: {"description": "The updated attendance row."},
        400: {"model": ErrorResponse, "description": "`invalid_attendance_status`."},
        401: _AUTH,
        404: {"model": ErrorResponse, "description": "`attendance_not_found`."},
        422: _INVALID,
    },
)
async def correct_attendance(
    payload: AttendanceUpdateRequest,
    attendance_id: int = Path(..., ge=1),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_teacher),
) -> AttendanceRecordResponse:
    if payload.status not in ATTENDANCE_STATUSES:
        raise InvalidAttendanceStatusError(
            f"Unsupported status {payload.status!r}. Allowed: {', '.join(ATTENDANCE_STATUSES)}"
        )
    record, previous = repository.update_attendance(
        attendance_id, status=payload.status, actor=actor
    )
    if record is None:
        raise AttendanceNotFoundError(f"Attendance {attendance_id} not found")
    if previous != payload.status:
        logger.warning(
            "Attendance %s corrected by %s: %s -> %s",
            attendance_id,
            actor,
            previous,
            payload.status,
        )
    return _to_response(record)


@router.get(
    "/{attendance_id}/audit",
    response_model=AuditListResponse,
    summary="Audit trail for one attendance record",
    responses={200: {"description": "Audit entries, newest first."}, 401: _AUTH, 422: _INVALID},
)
async def attendance_audit(
    attendance_id: int = Path(..., ge=1),
    limit: int = Query(default=50, ge=1, le=500),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_teacher),
) -> AuditListResponse:
    entries = repository.list_audit(attendance_id=attendance_id, limit=limit)
    return AuditListResponse(
        total=len(entries),
        entries=[
            AuditEntryResponse(
                id=entry.id,
                attendance_id=entry.attendance_id,
                student_id=entry.student_id,
                action=entry.action,
                old_status=entry.old_status,
                new_status=entry.new_status,
                actor=entry.actor,
                created_at=entry.created_at,
            )
            for entry in entries
        ],
    )
