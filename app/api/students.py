"""Student registry and face-registration endpoints.

Face embeddings are **written** here but never returned - only opaque metadata
is exposed. Every mutation requires the admin API key; listing is also available
to teachers.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Path, Query, status

from app.api.dependencies import ImageUpload, read_image_upload, require_face_model
from app.core.config import settings
from app.core.errors import (
    DuplicateFaceError,
    DuplicateRollNumberError,
    StudentNotFoundError,
    TooManyFacesError,
)
from app.core.security import require_admin, require_teacher
from app.database.records import StudentRecord
from app.database.repository import AttendanceRepository, get_repository
from app.schemas.common import ErrorResponse, MessageResponse, ValidationErrorResponse
from app.schemas.student import (
    ClassListResponse,
    ClassSummaryResponse,
    FaceMetadataResponse,
    FaceUploadResponse,
    StudentCreateRequest,
    StudentFacesResponse,
    StudentListResponse,
    StudentResponse,
    StudentUpdateRequest,
)
from app.services.embedding_service import embedding_service
from app.services.face_service import cosine_similarity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/students", tags=["students"])

_NOT_FOUND = {"model": ErrorResponse, "description": "`student_not_found`."}
_INVALID = {"model": ValidationErrorResponse, "description": "`invalid_request`."}
_AUTH = {"model": ErrorResponse, "description": "`unauthorized` / `forbidden`."}


def _to_response(student: StudentRecord) -> StudentResponse:
    return StudentResponse(
        id=student.id,
        roll_number=student.roll_number,
        name=student.name,
        class_name=student.class_name,
        section=student.section,
        face_count=student.face_count,
        created_at=student.created_at,
        updated_at=student.updated_at,
    )


def _require_student(
    repository: AttendanceRepository, student_id: int
) -> StudentRecord:
    student = repository.get_student(student_id)
    if student is None:
        raise StudentNotFoundError(f"Student {student_id} not found")
    return student


@router.post(
    "",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a student",
    description=(
        "Creates the student record. Face photos are uploaded separately with "
        "`POST /api/students/{student_id}/faces` (3-5 photos recommended).\n\n"
        "`roll_number` must be unique within the given `class_name` + `section`."
    ),
    responses={
        201: {"description": "The created student."},
        401: _AUTH,
        409: {"model": ErrorResponse, "description": "`duplicate_roll_number`."},
        422: _INVALID,
    },
)
async def create_student(
    payload: StudentCreateRequest,
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_admin),
) -> StudentResponse:
    existing = repository.get_student_by_roll(
        payload.roll_number, payload.class_name, payload.section
    )
    if existing is not None:
        raise DuplicateRollNumberError(
            f"Roll number {payload.roll_number} already exists in "
            f"{payload.class_name}-{payload.section} (student {existing.id})"
        )

    student = repository.create_student(
        roll_number=payload.roll_number,
        name=payload.name,
        class_name=payload.class_name,
        section=payload.section,
    )
    logger.info(
        "Student %s created by %s: roll=%s name=%r class=%s-%s",
        student.id,
        actor,
        student.roll_number,
        student.name,
        student.class_name,
        student.section,
    )
    return _to_response(student)


@router.get(
    "",
    response_model=StudentListResponse,
    summary="List students",
    description="Paginated listing, optionally filtered by class and section.",
    responses={200: {"description": "A page of students."}, 401: _AUTH, 422: _INVALID},
)
async def list_students(
    class_name: str | None = Query(default=None, max_length=40, description="e.g. `8`."),
    section: str | None = Query(default=None, max_length=10, description="e.g. `A`."),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_teacher),
) -> StudentListResponse:
    students = repository.list_students(
        class_name=class_name, section=section, limit=limit, offset=offset
    )
    total = repository.count_students(class_name=class_name, section=section)
    return StudentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        students=[_to_response(s) for s in students],
    )


@router.get(
    "/classes",
    response_model=ClassListResponse,
    summary="List known classes and sections",
    responses={200: {"description": "Distinct class/section pairs with head-counts."}, 401: _AUTH},
)
async def list_classes(
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_teacher),
) -> ClassListResponse:
    return ClassListResponse(
        classes=[
            ClassSummaryResponse(class_name=c, section=s, student_count=n)
            for c, s, n in repository.list_classes()
        ]
    )


@router.get(
    "/{student_id}",
    response_model=StudentResponse,
    summary="Get one student",
    responses={200: {"description": "The student."}, 401: _AUTH, 404: _NOT_FOUND, 422: _INVALID},
)
async def get_student(
    student_id: int = Path(..., ge=1),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_teacher),
) -> StudentResponse:
    return _to_response(_require_student(repository, student_id))


@router.patch(
    "/{student_id}",
    response_model=StudentResponse,
    summary="Update a student's details",
    responses={
        200: {"description": "The updated student."},
        401: _AUTH,
        404: _NOT_FOUND,
        409: {"model": ErrorResponse, "description": "`duplicate_roll_number`."},
        422: _INVALID,
    },
)
async def update_student(
    payload: StudentUpdateRequest,
    student_id: int = Path(..., ge=1),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_admin),
) -> StudentResponse:
    student = _require_student(repository, student_id)

    updates = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not updates:
        return _to_response(student)

    target_roll = updates.get("roll_number", student.roll_number)
    target_class = updates.get("class_name", student.class_name)
    target_section = updates.get("section", student.section)
    if (target_roll, target_class, target_section) != (
        student.roll_number,
        student.class_name,
        student.section,
    ):
        clash = repository.get_student_by_roll(target_roll, target_class, target_section)
        if clash is not None and clash.id != student_id:
            raise DuplicateRollNumberError(
                f"Roll number {target_roll} already exists in "
                f"{target_class}-{target_section} (student {clash.id})"
            )

    updated = repository.update_student(student_id, **updates)
    if updated is None:  # pragma: no cover - deleted concurrently
        raise StudentNotFoundError(f"Student {student_id} not found")
    logger.info("Student %s updated by %s: %s", student_id, actor, updates)
    return _to_response(updated)


@router.delete(
    "/{student_id}",
    response_model=MessageResponse,
    summary="Delete a student and all their biometric data",
    description=(
        "Removes the student, every stored face embedding (biometric data) and "
        "their attendance rows. This satisfies the 'delete a student's account' "
        "and 'delete a student's biometric data' requirements."
    ),
    responses={200: {"description": "Deletion acknowledged."}, 401: _AUTH, 404: _NOT_FOUND},
)
async def delete_student(
    student_id: int = Path(..., ge=1),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_admin),
) -> MessageResponse:
    if not repository.delete_student(student_id):
        raise StudentNotFoundError(f"Student {student_id} not found")
    logger.warning(
        "Student %s and all associated biometric data deleted by %s", student_id, actor
    )
    return MessageResponse(
        message=f"Student {student_id} and all associated biometric data were deleted"
    )


async def _writable_student(
    student_id: int = Path(..., ge=1, description="Student primary key."),
    repository: AttendanceRepository = Depends(get_repository),
) -> StudentRecord:
    """Validate that the student exists and can accept another embedding."""
    student = _require_student(repository, student_id)
    if student.face_count >= settings.faces_per_student_max:
        raise TooManyFacesError(
            f"Student {student_id} already has {student.face_count} face embeddings "
            f"(maximum {settings.faces_per_student_max}). Delete one before adding another."
        )
    return student


@router.post(
    "/{student_id}/faces",
    response_model=FaceUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register one face photo for a student",
    description=(
        "Uploads a single photo, detects the face, generates a pretrained ArcFace "
        "embedding and stores it for this student.\n\n"
        "Registration rules enforced:\n"
        "* exactly one usable face in the photo;\n"
        "* zero faces -> `no_face_detected`;\n"
        "* more than one face -> `multiple_faces_detected`;\n"
        "* a face smaller than `FACE_MIN_FACE_SIZE` -> `face_too_small`;\n"
        "* a photo too similar to one already stored -> `duplicate_face`;\n"
        "* a photo that is clearly ANOTHER student's face -> `duplicate_face`.\n\n"
        f"Store {settings.faces_per_student_min}-{settings.faces_per_student_max} "
        "photos per student for reliable recognition. The raw image is processed in "
        "memory and is never written to disk."
    ),
    responses={
        201: {"description": "Metadata of the stored embedding."},
        400: {
            "model": ErrorResponse,
            "description": "`invalid_image`, `no_face_detected`, `multiple_faces_detected`, `face_too_small` or `too_many_faces`.",
        },
        401: _AUTH,
        404: _NOT_FOUND,
        409: {"model": ErrorResponse, "description": "`duplicate_face`."},
        413: {"model": ErrorResponse, "description": "`payload_too_large`."},
        422: _INVALID,
    },
)
async def upload_student_face(
    actor: str = Depends(require_admin),
    model_ready: None = Depends(require_face_model),
    student: StudentRecord = Depends(_writable_student),
    upload: ImageUpload = Depends(read_image_upload),
    repository: AttendanceRepository = Depends(get_repository),
) -> FaceUploadResponse:
    # Enforce the single-face registration rule (raises clean 400s).
    registration = embedding_service.single_face_embedding(upload.image)

    # Reject an accidental re-upload of an already-registered photo.
    for existing in repository.list_faces(student.id):
        similarity = cosine_similarity(existing.embedding, registration.embedding)
        if similarity >= settings.duplicate_face_similarity:
            raise DuplicateFaceError(
                f"This photo is {similarity:.2f} similar to an existing registration "
                f"photo (>= {settings.duplicate_face_similarity}). Upload a different "
                "pose, expression, angle or lighting condition."
            )

    # Guard against registering ANOTHER student's face under this record: a
    # near-identical match to a different student's stored embedding means the
    # photo almost certainly belongs to that person, not this student.
    other = repository.search_similar(
        registration.embedding,
        limit=1,
        exclude_student_id=student.id,
    )
    if other:
        other_face, other_similarity = other[0]
        if other_similarity >= settings.duplicate_face_similarity:
            raise DuplicateFaceError(
                f"This photo is {other_similarity:.2f} similar (>= "
                f"{settings.duplicate_face_similarity}) to a photo already "
                f"registered for another student (id {other_face.student_id}). "
                "It appears to be the wrong person - upload the correct "
                "student's photo."
            )

    record = repository.add_face(
        student.id, registration.embedding, registration.det_score
    )
    total = repository.count_faces(student.id)
    is_ready = total >= settings.faces_per_student_min

    logger.info(
        "Face %s registered for student %s by %s (det_score=%.3f, total=%d)",
        record.id,
        student.id,
        actor,
        registration.det_score,
        total,
    )

    if is_ready:
        message = f"Face registered. Student now has {total} embeddings."
    else:
        remaining = settings.faces_per_student_min - total
        message = (
            f"Face registered ({total}/{settings.faces_per_student_min} minimum). "
            f"Upload at least {remaining} more photo(s)."
        )

    return FaceUploadResponse(
        face_id=record.id,
        student_id=student.id,
        det_score=round(registration.det_score, 4),
        total_faces=total,
        is_ready=is_ready,
        message=message,
    )


@router.get(
    "/{student_id}/faces",
    response_model=StudentFacesResponse,
    summary="List metadata for a student's registered face embeddings",
    description=(
        "Returns one entry per stored face embedding, **without** the embedding "
        "vector itself - embeddings are sensitive biometric data."
    ),
    responses={
        200: {"description": "Embedding metadata."},
        401: _AUTH,
        404: _NOT_FOUND,
        422: _INVALID,
    },
)
async def list_student_faces(
    student_id: int = Path(..., ge=1),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_teacher),
) -> StudentFacesResponse:
    student = _require_student(repository, student_id)
    faces = repository.list_faces(student_id)
    return StudentFacesResponse(
        student_id=student.id,
        roll_number=student.roll_number,
        name=student.name,
        total_faces=len(faces),
        faces=[
            FaceMetadataResponse(
                id=face.id,
                student_id=face.student_id,
                det_score=round(face.det_score, 4),
                created_at=face.created_at,
            )
            for face in faces
        ],
    )


@router.delete(
    "/{student_id}/faces",
    response_model=MessageResponse,
    summary="Delete all biometric data for a student",
    description=(
        "Removes every stored face embedding for this student while keeping the "
        "student record. Implements the 'delete a student's biometric data' "
        "requirement."
    ),
    responses={200: {"description": "Deletion acknowledged."}, 401: _AUTH, 404: _NOT_FOUND},
)
async def delete_student_faces(
    student_id: int = Path(..., ge=1),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_admin),
) -> MessageResponse:
    _require_student(repository, student_id)
    removed = repository.delete_faces(student_id)
    logger.warning(
        "All biometric data for student %s deleted by %s (%d embeddings)",
        student_id,
        actor,
        removed,
    )
    return MessageResponse(
        message=f"Deleted {removed} face embedding(s) for student {student_id}"
    )


@router.delete(
    "/{student_id}/faces/{face_id}",
    response_model=MessageResponse,
    summary="Delete one face embedding",
    responses={200: {"description": "Deletion acknowledged."}, 401: _AUTH, 404: _NOT_FOUND},
)
async def delete_student_face(
    student_id: int = Path(..., ge=1),
    face_id: int = Path(..., ge=1),
    repository: AttendanceRepository = Depends(get_repository),
    actor: str = Depends(require_admin),
) -> MessageResponse:
    owned = {face.id for face in repository.list_faces(student_id)}
    if face_id not in owned:
        raise StudentNotFoundError(
            f"Face {face_id} does not belong to student {student_id}"
        )
    repository.delete_face(face_id)
    logger.warning("Face %s of student %s deleted by %s", face_id, student_id, actor)
    return MessageResponse(message=f"Face {face_id} deleted")
