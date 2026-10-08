"""Classroom photo -> attendance records.

Recognition and attendance are deliberately separate: the matching service never
writes to the database, and this service is the only place that creates or
updates attendance rows (always through the audited repository methods).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from app.database.records import STATUS_ABSENT, STATUS_PRESENT, AttendanceRecord
from app.database.repository import AttendanceRepository
from app.services.embedding_service import EmbeddingService, embedding_service
from app.services.matching_service import MatchingService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RecognizedEntry:
    student_id: int
    roll_number: str
    name: str
    status: str
    confidence: float
    created: bool


@dataclass(slots=True)
class RosterEntry:
    """One student of the selected class and their proposed attendance status.

    ``confidence`` is the cosine similarity when the student was recognised by
    the model, and ``None`` for an absent student (either auto-derived from a
    missing face or set by the teacher).
    """

    student_id: int
    roll_number: str
    name: str
    status: str
    confidence: float | None
    recognised: bool
    #: True when a teacher explicitly set this status during review. Auto-derived
    #: absents are not persisted; explicit decisions are.
    teacher_confirmed: bool = False


@dataclass(slots=True)
class ConfirmedEntry:
    """One teacher-approved row submitted to ``confirm``."""

    student_id: int
    status: str
    confidence: float | None = None


@dataclass(slots=True)
class ConfirmOutcome:
    """Result of persisting a teacher-approved roster."""

    attendance_date: date
    class_name: str
    section: str
    saved: int
    created: int
    skipped: int
    already_present: int
    records: list[AttendanceRecord] = field(default_factory=list)


@dataclass(slots=True)
class AttendanceOutcome:
    attendance_date: date
    class_name: str
    section: str
    total_faces: int
    unknown_faces: int
    duplicates_removed: int
    already_present: int
    skipped_out_of_class: int
    threshold: float
    processing_time_ms: float
    entries: list[RecognizedEntry] = field(default_factory=list)
    #: Full class roster with proposed statuses (present/absent). Always filled.
    roster: list[RosterEntry] = field(default_factory=list)
    #: True when nothing was written (dry run / preview).
    preview: bool = False
    #: Rows already stored for this class + date, if any.
    existing_records: int = 0

    @property
    def recognized_students(self) -> int:
        return len(self.entries)

    @property
    def absent_students(self) -> list[RosterEntry]:
        return [r for r in self.roster if r.status != STATUS_PRESENT]

    @property
    def present_count(self) -> int:
        return sum(1 for r in self.roster if r.status == STATUS_PRESENT)


class AttendanceService:
    def __init__(
        self,
        repository: AttendanceRepository,
        embedder: EmbeddingService | None = None,
        matcher: MatchingService | None = None,
    ) -> None:
        self._repository = repository
        self._embedder = embedder or embedding_service
        self._matcher = matcher or MatchingService(repository)

    def process(
        self,
        image: np.ndarray,
        *,
        class_name: str,
        section: str,
        attendance_date: date,
        actor: str = "anonymous",
        status: str = STATUS_PRESENT,
        dry_run: bool = False,
    ) -> AttendanceOutcome:
        """Recognise a classroom photo for one class.

        With ``dry_run=True`` nothing is written: the outcome carries the full
        class roster (present + absent) so the teacher can review and confirm.
        With ``dry_run=False`` the historical behaviour applies - recognised
        students are written immediately.
        """
        started = time.perf_counter()

        faces = self._embedder.detect_faces(image)  # raises NoFaceDetectedError

        # Scope matching to the selected class so a face can never be matched
        # against another class's students.
        matcher = MatchingService(
            self._repository, class_name=class_name, section=section
        )
        identification = matcher.identify(faces)
        matches = identification.matches
        unknown = identification.unknown
        duplicates = identification.duplicates

        entries: list[RecognizedEntry] = []
        already_present = 0
        skipped_out_of_class = 0
        matched: dict[int, float] = {}

        for match in matches:
            student = match.student
            if student is None:
                continue
            # Defensive: never mark a student who is not part of this class.
            if student.class_name != class_name or student.section != section:
                skipped_out_of_class += 1
                logger.info(
                    "Skipping student %s (%s-%s): not part of %s-%s",
                    student.id,
                    student.class_name,
                    student.section,
                    class_name,
                    section,
                )
                continue

            confidence = round(match.similarity, 4)
            matched[student.id] = confidence

            created = False
            if not dry_run:
                record, created = self._repository.upsert_attendance(
                    student_id=student.id,
                    class_name=class_name,
                    section=section,
                    attendance_date=attendance_date,
                    status=status,
                    confidence=confidence,
                    actor=actor,
                )
                if not created:
                    already_present += 1
                record_status = record.status
            else:
                record_status = status

            entries.append(
                RecognizedEntry(
                    student_id=student.id,
                    roll_number=student.roll_number,
                    name=student.name,
                    status=record_status,
                    confidence=confidence,
                    created=created,
                )
            )

        entries.sort(key=lambda e: e.roll_number)

        # Full class roster: everyone registered for this class, recognised or not.
        roster = self._build_roster(
            class_name=class_name, section=section, matched=matched
        )
        existing = len(
            self._repository.list_attendance(
                class_name=class_name, section=section, attendance_date=attendance_date
            )
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "Attendance %s: date=%s class=%s-%s total_faces=%d recognized=%d "
            "absent=%d unknown=%d duplicates_removed=%d already_present=%d "
            "out_of_class=%d threshold=%.2f elapsed_ms=%.1f",
            "PREVIEW" if dry_run else "SAVED",
            attendance_date,
            class_name,
            section,
            len(faces),
            len(entries),
            sum(1 for r in roster if r.status != STATUS_PRESENT),
            unknown,
            duplicates,
            already_present,
            skipped_out_of_class,
            matcher.threshold,
            elapsed_ms,
        )

        return AttendanceOutcome(
            attendance_date=attendance_date,
            class_name=class_name,
            section=section,
            total_faces=len(faces),
            unknown_faces=unknown,
            duplicates_removed=duplicates,
            already_present=already_present,
            skipped_out_of_class=skipped_out_of_class,
            threshold=matcher.threshold,
            processing_time_ms=round(elapsed_ms, 1),
            entries=entries,
            roster=roster,
            preview=dry_run,
            existing_records=existing,
        )

    def _build_roster(
        self, *, class_name: str, section: str, matched: dict[int, float]
    ) -> list[RosterEntry]:
        """Every registered student of the class, marked present or absent.

        Auto-derived absents carry ``teacher_confirmed=False`` so the confirm
        step knows not to persist them - only recognised students and explicit
        teacher decisions are written.
        """
        students = self._repository.list_students(
            class_name=class_name, section=section, limit=500
        )
        roster: list[RosterEntry] = []
        for student in students:
            similarity = matched.get(student.id)
            roster.append(
                RosterEntry(
                    student_id=student.id,
                    roll_number=student.roll_number,
                    name=student.name,
                    status=STATUS_PRESENT if similarity is not None else STATUS_ABSENT,
                    confidence=similarity,
                    recognised=similarity is not None,
                )
            )
        roster.sort(key=lambda r: r.roll_number)
        return roster

    def confirm(
        self,
        *,
        class_name: str,
        section: str,
        attendance_date: date,
        entries: list[ConfirmedEntry],
        actor: str = "anonymous",
    ) -> ConfirmOutcome:
        """Persist the teacher-approved roster.

        Only entries the model recognised **or** that the teacher explicitly
        confirmed are written. Auto-derived absents (a student merely missing
        from the photo) are intentionally not stored, so the register only
        records attendance that was either observed or decided by a person.

        Re-confirming the same class/date is safe: the repository's unique index
        prevents duplicate rows and existing rows are left untouched.
        """
        students = {
            s.id: s
            for s in self._repository.list_students(
                class_name=class_name, section=section, limit=500
            )
        }

        already = {
            record.student_id
            for record in self._repository.list_attendance(
                class_name=class_name,
                section=section,
                attendance_date=attendance_date,
            )
        }

        written: list[AttendanceRecord] = []
        created = 0
        skipped = 0

        for entry in entries:
            student = students.get(entry.student_id)
            if student is None:
                # Defence in depth: never write a student outside this class.
                logger.warning(
                    "Confirm ignored student %s (not in %s-%s)",
                    entry.student_id,
                    class_name,
                    section,
                )
                skipped += 1
                continue

            if entry.student_id in already:
                skipped += 1
                continue

            record, was_created = self._repository.upsert_attendance(
                student_id=student.id,
                class_name=class_name,
                section=section,
                attendance_date=attendance_date,
                status=entry.status,
                confidence=entry.confidence,
                actor=actor,
            )
            written.append(record)
            if was_created:
                created += 1

        logger.info(
            "Attendance confirmed: class=%s-%s date=%s written=%d created=%d "
            "skipped=%d existing=%d",
            class_name,
            section,
            attendance_date,
            len(written),
            created,
            skipped,
            len(already),
        )

        return ConfirmOutcome(
            attendance_date=attendance_date,
            class_name=class_name,
            section=section,
            saved=len(written),
            created=created,
            skipped=skipped,
            already_present=len(already),
            records=written,
        )
