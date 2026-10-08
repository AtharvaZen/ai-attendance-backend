from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class RecognizedStudent(BaseModel):
    """A detected face that was matched to a registered student."""

    student_id: int
    roll_number: str
    name: str
    class_name: str
    section: str
    confidence: float = Field(
        description=(
            "Cosine similarity between the detected face embedding and the "
            "student's registered embeddings, in [0, 1]. This is a "
            "**confidence/similarity score, NOT a probability** that the face "
            "belongs to this student."
        )
    )
    face_bbox: list[float] = Field(
        description="Bounding box `[x1, y1, x2, y2]` of the matched face in the uploaded image."
    )


class FaceMatchDetail(BaseModel):
    """Debug detail for **one** detected face - matched or unknown.

    Lets a reviewer see exactly why a face was accepted or rejected: the best
    similarity, the runner-up from a different student, the resulting margin
    and the thresholds that were applied. Embeddings are never exposed.
    """

    face_index: int = Field(description="0-based index of the face in the uploaded image.")
    bbox: list[float] = Field(
        description="Bounding box `[x1, y1, x2, y2]` of the face in the uploaded image."
    )
    student_id: int | None = Field(
        default=None,
        description="Matched student id, or `null` when the face was rejected/unknown.",
    )
    name: str | None = Field(default=None, description="Matched student name, if any.")
    roll_number: str | None = Field(default=None, description="Matched student roll number, if any.")
    best_similarity: float = Field(
        description="Cosine similarity of the closest registered embedding (not a probability)."
    )
    second_best_similarity: float | None = Field(
        default=None,
        description="Best similarity from a *different* student, or `null` when there was no runner-up.",
    )
    margin: float | None = Field(
        default=None,
        description=(
            "best_similarity - second_best_similarity across different students, "
            "or `null` when there was no runner-up."
        ),
    )
    threshold: float = Field(description="FACE_MATCH_THRESHOLD applied to this run.")
    margin_threshold: float = Field(description="FACE_MATCH_MARGIN applied to this run.")
    decision: str = Field(
        description="`match` (accepted), `duplicate` (accepted but superseded by a better appearance) or `unknown`."
    )


class RecognitionResponse(BaseModel):
    """Result of a pure recognition pass - nothing is written to the database."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_faces": 30,
                "recognized": [
                    {
                        "student_id": 101,
                        "roll_number": "101",
                        "name": "Rahul Sharma",
                        "class_name": "8",
                        "section": "A",
                        "confidence": 0.93,
                        "face_bbox": [120.0, 88.0, 210.0, 205.0],
                    }
                ],
                "unknown_faces": 2,
                "duplicates_removed": 0,
                "threshold": 0.45,
                "margin_threshold": 0.05,
                "processing_time_ms": 812.4,
            }
        }
    )

    total_faces: int = Field(description="Number of usable faces detected in the image.")
    recognized: list[RecognizedStudent]
    unknown_faces: int = Field(description="Detected faces that matched no student above the threshold.")
    duplicates_removed: int = Field(
        default=0,
        description="Extra appearances of a student who was already recognised in this image.",
    )
    threshold: float = Field(description="Cosine-similarity threshold that was applied.")
    margin_threshold: float = Field(
        default=0.0,
        description="Identification margin (FACE_MATCH_MARGIN) that was applied. 0 disables the gate.",
    )
    match_details: list[FaceMatchDetail] = Field(
        default_factory=list,
        description="Per-face scores, runner-up, margin and decision for every detected face.",
    )
    processing_time_ms: float
