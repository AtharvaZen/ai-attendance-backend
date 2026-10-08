"""Student and face-embedding schemas.

Privacy note: face embeddings are sensitive biometric data and are therefore
**never** included in any API response - only opaque metadata is exposed.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StudentCreateRequest(BaseModel):
    """Register a new student (before any face photos are uploaded)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "roll_number": "101",
                "name": "Rahul Sharma",
                "class_name": "8",
                "section": "A",
            }
        }
    )

    roll_number: str = Field(
        min_length=1, max_length=32, description="Unique within this class + section."
    )
    name: str = Field(min_length=1, max_length=120, description="Student's full name.")
    class_name: str = Field(min_length=1, max_length=40, description="e.g. `8`.")
    section: str = Field(min_length=1, max_length=10, description="e.g. `A`.")

    @field_validator("roll_number", "name", "class_name", "section")
    @classmethod
    def _strip_and_require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class StudentUpdateRequest(BaseModel):
    """Correct a student's details. Only the supplied fields are changed."""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    roll_number: str | None = Field(default=None, min_length=1, max_length=32)
    class_name: str | None = Field(default=None, min_length=1, max_length=40)
    section: str | None = Field(default=None, min_length=1, max_length=10)

    @field_validator("name", "roll_number", "class_name", "section")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class StudentResponse(BaseModel):
    """A registered student. Contains no biometric data."""

    id: int
    roll_number: str
    name: str
    class_name: str
    section: str
    face_count: int = Field(
        default=0,
        description="How many face embeddings are stored (3-5 is the recommended range).",
    )
    created_at: datetime
    updated_at: datetime


class StudentListResponse(BaseModel):
    total: int = Field(description="Total matching students, ignoring pagination.")
    limit: int
    offset: int
    students: list[StudentResponse]


class ClassSummaryResponse(BaseModel):
    class_name: str
    section: str
    student_count: int


class ClassListResponse(BaseModel):
    classes: list[ClassSummaryResponse]


class FaceUploadResponse(BaseModel):
    """Result of registering one face photo."""

    face_id: int
    student_id: int
    det_score: float = Field(
        description="Face-detector confidence of this photo (NOT an identity probability)."
    )
    total_faces: int = Field(
        description="Embeddings now stored for this student."
    )
    is_ready: bool = Field(
        description="True once the student has at least the recommended minimum number of photos."
    )
    message: str


class FaceMetadataResponse(BaseModel):
    """Metadata about one stored face embedding (never the vector itself)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 7,
                "student_id": 101,
                "det_score": 0.93,
                "created_at": "2026-10-01T09:15:00Z",
            }
        }
    )

    id: int
    student_id: int
    det_score: float = Field(
        description="Face-detector confidence of the source photo. "
        "This is NOT a probability that the face belongs to the student."
    )
    created_at: datetime


class StudentFacesResponse(BaseModel):
    """A student together with metadata for each of their stored embeddings."""

    student_id: int
    roll_number: str
    name: str
    total_faces: int = Field(description="Number of stored embeddings for this student.")
    faces: list[FaceMetadataResponse]
