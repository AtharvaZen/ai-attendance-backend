"""Cosine-similarity matching of detected faces against registered embeddings.

Read this before interpreting any number this module produces
----------------------------------------------------------------
The returned value is a **cosine similarity between two face embeddings**
produced by a pretrained ArcFace model. It is **NOT a probability** that the
detected face belongs to the matched student.

``0.93`` similarity does not mean "93 % likely to be this student". Similarity
scores are not calibrated probabilities, and the same score means different
things for different cameras, lighting and populations. Call it a
*confidence/similarity score* in any UI or report.

The accept/reject decision has two configurable gates, both tuned on your own
data (see ``scripts/evaluate_recognition.py``):

1. ``FACE_MATCH_THRESHOLD`` - the best cosine similarity must clear it.
2. ``FACE_MATCH_MARGIN`` - the lead of the best candidate over the best
   candidate from a *different* student must clear it, so a look-alike who
   scores 0.51 while the runner-up scores 0.49 is reported as UNKNOWN
   instead of being accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import settings
from app.database.records import StudentRecord
from app.database.repository import AttendanceRepository
from app.services.face_service import DetectedFace

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FaceMatch:
    """The outcome of matching one detected face."""

    face: DetectedFace
    student: StudentRecord | None
    similarity: float
    #: Best similarity belonging to a *different* student (identification margin).
    runner_up_similarity: float | None
    #: True when ``similarity >= threshold``, the margin over the runner-up is
    #: ``>= face_match_margin``, and a student was resolved.
    accepted: bool

    @property
    def margin(self) -> float | None:
        if self.runner_up_similarity is None:
            return None
        return self.similarity - self.runner_up_similarity


@dataclass(slots=True)
class IdentificationResult:
    """Everything one identification pass produced.

    ``matches`` keeps its historical meaning: the accepted faces after
    same-student deduplication, best-scoring first. ``all_matches`` carries
    **every** detected face in original order (accepted, rejected, unknown)
    so callers can expose per-face scores/margins in responses and logs.
    """

    matches: list[FaceMatch]
    all_matches: list[FaceMatch]
    unknown: int
    duplicates: int


class MatchingService:
    """Nearest-neighbour matching with a configurable similarity threshold."""

    def __init__(
        self,
        repository: AttendanceRepository,
        threshold: float | None = None,
        class_name: str | None = None,
        section: str | None = None,
        margin: float | None = None,
    ) -> None:
        self._repository = repository
        self._threshold = (
            settings.face_match_threshold if threshold is None else threshold
        )
        # Minimum lead over the best *other* student. 0 = gate disabled.
        self._margin = settings.face_match_margin if margin is None else margin
        # When set, candidates are restricted to that class *inside* the search,
        # so a face can never be matched against another class's students.
        self._class_name = class_name
        self._section = section

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def margin_threshold(self) -> float:
        return self._margin

    def match_face(self, face: DetectedFace) -> FaceMatch:
        """Find the closest registered student for a single detected face."""
        if face.embedding is None:
            return FaceMatch(face, None, 0.0, None, False)

        candidates = self._repository.search_similar(
            face.embedding,
            limit=settings.recognition_candidates,
            class_name=self._class_name,
            section=self._section,
        )
        if not candidates:
            return FaceMatch(face, None, 0.0, None, False)

        best_face, best_similarity = candidates[0]
        # The runner-up must come from a different student to be meaningful.
        runner_up = next(
            (
                similarity
                for candidate, similarity in candidates[1:]
                if candidate.student_id != best_face.student_id
            ),
            None,
        )

        student = self._repository.get_student(best_face.student_id)
        # Two gates: (1) absolute similarity threshold, (2) identification
        # margin over the best candidate from a *different* student. Without
        # gate (2) a look-alike can pass the threshold alone (e.g. 0.51 vs a
        # runner-up of 0.49) and be wrongly accepted as a registered student.
        margin_ok = runner_up is None or (
            best_similarity - runner_up >= self._margin
        ) or self._margin <= 0.0
        accepted = (
            student is not None
            and best_similarity >= self._threshold
            and margin_ok
        )

        margin_value = (
            None if runner_up is None else best_similarity - runner_up
        )
        if student is not None and best_similarity >= self._threshold and not margin_ok:
            reason = "below_margin"
        elif student is None:
            reason = "student_unavailable"
        elif best_similarity < self._threshold:
            reason = "below_threshold"
        else:
            reason = "match"

        logger.info(
            "Face match: similarity=%.4f threshold=%.2f runner_up=%s margin=%s "
            "margin_threshold=%.2f decision=%s student_id=%s",
            best_similarity,
            self._threshold,
            "n/a" if runner_up is None else f"{runner_up:.4f}",
            "n/a" if margin_value is None else f"{margin_value:.4f}",
            self._margin,
            reason,
            student.id if accepted and student else None,
        )
        return FaceMatch(face, student if accepted else None, best_similarity, runner_up, accepted)

    def match_faces(self, faces: list[DetectedFace]) -> list[FaceMatch]:
        return [self.match_face(face) for face in faces]

    def identify(self, faces: list[DetectedFace]) -> IdentificationResult:
        """Run matching for a photo and split the results.

        A student detected more than once keeps only their best-scoring
        occurrence; the extra occurrences are counted as duplicates. Faces
        rejected by the threshold or margin gates are counted as unknown.
        ``all_matches`` preserves every per-face outcome for responses/logs.
        """
        matches = self.match_faces(faces)

        best_by_student: dict[int, FaceMatch] = {}
        unknown = 0
        duplicates = 0
        for match in matches:
            if not match.accepted or match.student is None:
                unknown += 1
                continue
            current = best_by_student.get(match.student.id)
            if current is None:
                best_by_student[match.student.id] = match
            elif match.similarity > current.similarity:
                best_by_student[match.student.id] = match
                duplicates += 1
            else:
                duplicates += 1

        ordered = sorted(best_by_student.values(), key=lambda m: m.similarity, reverse=True)
        return IdentificationResult(
            matches=ordered,
            all_matches=matches,
            unknown=unknown,
            duplicates=duplicates,
        )
