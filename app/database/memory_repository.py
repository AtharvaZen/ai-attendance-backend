"""In-memory repository - development / demo only.

State lives in process-local dictionaries: it is lost on restart and is not
shared between worker processes. It exists so the whole API can be exercised
without a database; the persistent deployment uses MongoDB.

Search is exact brute-force cosine similarity, which is plenty fast at the scale
a single process can hold anyway.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import date

import numpy as np

from app.database.records import (
    AttendanceRecord,
    AuditRecord,
    FaceRecord,
    StudentRecord,
    utcnow,
)
from app.database.repository import AttendanceRepository
from app.services.face_service import l2_normalize


class InMemoryAttendanceRepository(AttendanceRepository):
    backend = "memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._students: dict[int, StudentRecord] = {}
        self._faces: dict[int, FaceRecord] = {}
        self._attendance: dict[int, AttendanceRecord] = {}
        self._audit: list[AuditRecord] = []
        self._next = {"student": 1, "face": 1, "attendance": 1, "audit": 1}

    def reset(self) -> None:
        with self._lock:
            self._students = {}
            self._faces = {}
            self._attendance = {}
            self._audit = []
            self._next = {"student": 1, "face": 1, "attendance": 1, "audit": 1}

    # -- students ----------------------------------------------------------
    def _with_face_count(self, student: StudentRecord) -> StudentRecord:
        return replace(student, face_count=self._face_count_unlocked(student.id))

    def _face_count_unlocked(self, student_id: int) -> int:
        return sum(1 for f in self._faces.values() if f.student_id == student_id)

    def create_student(
        self, *, roll_number: str, name: str, class_name: str, section: str
    ) -> StudentRecord:
        with self._lock:
            record = StudentRecord(
                id=self._next["student"],
                roll_number=roll_number,
                name=name,
                class_name=class_name,
                section=section,
            )
            self._next["student"] += 1
            self._students[record.id] = record
            return replace(record)

    def get_student(self, student_id: int) -> StudentRecord | None:
        with self._lock:
            record = self._students.get(student_id)
            return self._with_face_count(record) if record else None

    def get_student_by_roll(
        self, roll_number: str, class_name: str, section: str
    ) -> StudentRecord | None:
        with self._lock:
            for record in self._students.values():
                if (
                    record.roll_number == roll_number
                    and record.class_name == class_name
                    and record.section == section
                ):
                    return replace(record)
        return None

    def list_students(
        self,
        *,
        class_name: str | None = None,
        section: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StudentRecord]:
        with self._lock:
            rows = [
                s
                for s in self._students.values()
                if (class_name is None or s.class_name == class_name)
                and (section is None or s.section == section)
            ]
            rows.sort(key=lambda s: (s.class_name, s.section, s.id))
            return [
                self._with_face_count(s) for s in rows[offset : offset + limit]
            ]

    def count_students(
        self, *, class_name: str | None = None, section: str | None = None
    ) -> int:
        with self._lock:
            return sum(
                1
                for s in self._students.values()
                if (class_name is None or s.class_name == class_name)
                and (section is None or s.section == section)
            )

    def update_student(
        self,
        student_id: int,
        *,
        name: str | None = None,
        roll_number: str | None = None,
        class_name: str | None = None,
        section: str | None = None,
    ) -> StudentRecord | None:
        with self._lock:
            record = self._students.get(student_id)
            if record is None:
                return None
            if name is not None:
                record.name = name
            if roll_number is not None:
                record.roll_number = roll_number
            if class_name is not None:
                record.class_name = class_name
            if section is not None:
                record.section = section
            record.updated_at = utcnow()
            return self._with_face_count(record)

    def delete_student(self, student_id: int) -> bool:
        with self._lock:
            if student_id not in self._students:
                return False
            del self._students[student_id]
            self.delete_faces(student_id)
            for key in [
                key for key, a in self._attendance.items() if a.student_id == student_id
            ]:
                del self._attendance[key]
            return True

    def list_classes(self) -> list[tuple[str, str, int]]:
        with self._lock:
            buckets: dict[tuple[str, str], int] = {}
            for student in self._students.values():
                key = (student.class_name, student.section)
                buckets[key] = buckets.get(key, 0) + 1
            return sorted(
                ((c, s, n) for (c, s), n in buckets.items()), key=lambda t: (t[0], t[1])
            )

    # -- face embeddings ---------------------------------------------------
    def add_face(
        self, student_id: int, embedding: np.ndarray, det_score: float
    ) -> FaceRecord:
        with self._lock:
            record = FaceRecord(
                id=self._next["face"],
                student_id=student_id,
                embedding=l2_normalize(embedding),
                det_score=float(det_score),
            )
            self._next["face"] += 1
            self._faces[record.id] = record
            return replace(record, embedding=record.embedding.copy())

    def list_faces(self, student_id: int) -> list[FaceRecord]:
        with self._lock:
            rows = sorted(
                (f for f in self._faces.values() if f.student_id == student_id),
                key=lambda f: f.id,
            )
            return [replace(f, embedding=f.embedding.copy()) for f in rows]

    def count_faces(self, student_id: int) -> int:
        with self._lock:
            return self._face_count_unlocked(student_id)

    def delete_face(self, face_id: int) -> bool:
        with self._lock:
            return self._faces.pop(face_id, None) is not None

    def delete_faces(self, student_id: int) -> int:
        with self._lock:
            doomed = [
                fid for fid, f in self._faces.items() if f.student_id == student_id
            ]
            for fid in doomed:
                del self._faces[fid]
            return len(doomed)

    def search_similar(
        self,
        embedding: np.ndarray,
        *,
        limit: int = 5,
        exclude_student_id: int | None = None,
        class_name: str | None = None,
        section: str | None = None,
    ) -> list[tuple[FaceRecord, float]]:
        """Exact cosine similarity; restricted to one class when one is given."""
        query = l2_normalize(embedding)
        allowed: set[int] | None = None
        with self._lock:
            if class_name is not None:
                allowed = {
                    s.id
                    for s in self._students.values()
                    if s.class_name == class_name
                    and (section is None or s.section == section)
                }
            rows = [
                f
                for f in self._faces.values()
                if f.student_id != exclude_student_id
                and (allowed is None or f.student_id in allowed)
            ]
            if not rows:
                return []
            matrix = np.stack([f.embedding for f in rows]).astype(np.float32)

        similarities = matrix @ query  # both sides are L2-normalised
        order = np.argsort(-similarities)[: max(1, limit)]
        return [
            (replace(rows[i], embedding=rows[i].embedding.copy()), float(similarities[i]))
            for i in order
        ]

    def count_embeddings(self) -> int:
        with self._lock:
            return len(self._faces)

    # -- attendance --------------------------------------------------------
    def _decorate(self, record: AttendanceRecord) -> AttendanceRecord:
        student = self._students.get(record.student_id)
        return replace(
            record,
            roll_number=student.roll_number if student else "",
            student_name=student.name if student else "",
        )

    def _find_unlocked(
        self,
        student_id: int,
        class_name: str,
        section: str,
        attendance_date: date,
    ) -> AttendanceRecord | None:
        for record in self._attendance.values():
            if (
                record.student_id == student_id
                and record.class_name == class_name
                and record.section == section
                and record.attendance_date == attendance_date
            ):
                return record
        return None

    def get_attendance(self, attendance_id: int) -> AttendanceRecord | None:
        with self._lock:
            record = self._attendance.get(attendance_id)
            return self._decorate(record) if record else None

    def find_attendance(
        self,
        *,
        student_id: int,
        class_name: str,
        section: str,
        attendance_date: date,
    ) -> AttendanceRecord | None:
        with self._lock:
            record = self._find_unlocked(
                student_id, class_name, section, attendance_date
            )
            return self._decorate(record) if record else None

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
        with self._lock:
            existing = self._find_unlocked(
                student_id, class_name, section, attendance_date
            )
            if existing is not None:
                return self._decorate(existing), False

            record = AttendanceRecord(
                id=self._next["attendance"],
                student_id=student_id,
                class_name=class_name,
                section=section,
                attendance_date=attendance_date,
                status=status,
                confidence=confidence,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
            self._next["attendance"] += 1
            self._attendance[record.id] = record
            self._audit_log(
                attendance_id=record.id,
                student_id=student_id,
                action="created",
                old_status=None,
                new_status=status,
                actor=actor,
                confidence=confidence,
            )
            return self._decorate(record), True

    def update_attendance(
        self, attendance_id: int, *, status: str, actor: str
    ) -> tuple[AttendanceRecord | None, str | None]:
        with self._lock:
            record = self._attendance.get(attendance_id)
            if record is None:
                return None, None
            previous = record.status
            if previous != status:
                record.status = status
                record.updated_at = utcnow()
                self._audit_log(
                    attendance_id=record.id,
                    student_id=record.student_id,
                    action="manual_update",
                    old_status=previous,
                    new_status=status,
                    actor=actor,
                )
            return self._decorate(record), previous

    def list_attendance(
        self,
        *,
        class_name: str,
        section: str,
        attendance_date: date | None = None,
    ) -> list[AttendanceRecord]:
        with self._lock:
            rows = [
                a
                for a in self._attendance.values()
                if a.class_name == class_name
                and a.section == section
                and (attendance_date is None or a.attendance_date == attendance_date)
            ]
            rows.sort(key=lambda a: (a.attendance_date, a.student_id))
            return [self._decorate(a) for a in rows]

    # -- audit -------------------------------------------------------------
    def _audit_log(
        self,
        *,
        attendance_id: int | None,
        student_id: int,
        action: str,
        old_status: str | None,
        new_status: str | None,
        actor: str,
        confidence: float | None = None,
    ) -> None:
        entry = AuditRecord(
            id=self._next["audit"],
            attendance_id=attendance_id,
            student_id=student_id,
            action=action,
            old_status=old_status,
            new_status=new_status,
            actor=actor,
            confidence=confidence,
        )
        self._next["audit"] += 1
        self._audit.append(entry)

    def list_audit(
        self, *, attendance_id: int | None = None, limit: int = 100
    ) -> list[AuditRecord]:
        with self._lock:
            rows = [
                e
                for e in self._audit
                if attendance_id is None or e.attendance_id == attendance_id
            ]
            rows.sort(key=lambda e: e.id, reverse=True)
            return [replace(e) for e in rows[:limit]]
