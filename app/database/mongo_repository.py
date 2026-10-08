"""MongoDB repository - the persistent storage backend.

Documents (school-scoped):

* ``schools``         - one document per school, ``_id`` is the school id.
* ``students``        - ``schoolId`` + ``rollNumber`` is the logical identity;
  roll numbers are deliberately **not** globally unique.
* ``face_embeddings`` - one document per registered photo (many per student).
* ``attendance``      - unique on school + student + class + section + date.
* ``attendance_audit``- immutable trail of every status change.

Similarity search uses MongoDB's native ``$vectorSearch`` (cosine, filtered by
``schoolId``) when the deployment supports it, and otherwise exact cosine
similarity in Python. Only the storage concern lives here: no face-recognition
logic, no business rules.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import numpy as np
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.core.config import settings
from app.database.mongo_connection import (
    ATTENDANCE,
    ATTENDANCE_AUDIT,
    COUNTERS,
    FACE_EMBEDDINGS,
    STUDENTS,
    collection,
    utcnow,
    vector_search_supported,
)
from app.database.records import (
    AttendanceRecord,
    AuditRecord,
    FaceRecord,
    StudentRecord,
)
from app.database.repository import AttendanceRepository
from app.services.face_service import l2_normalize

logger = logging.getLogger(__name__)


def _next_id(key: str) -> int:
    """Atomically allocate the next integer id for ``key``.

    The public API exposes integer ids (``/api/students/{student_id}``), so ids
    are generated here rather than using MongoDB ``ObjectId``.
    """
    document = collection(COUNTERS).find_one_and_update(
        {"_id": key}, {"$inc": {"value": 1}}, upsert=True, return_document=ReturnDocument.AFTER
    )
    return int(document["value"])


def _as_datetime(value: datetime | None) -> datetime:
    """Normalise a Mongo datetime to an aware UTC datetime."""
    if value is None:
        return utcnow()
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _as_date(value) -> date:
    """Mongo stores ``datetime``; the domain uses ``date``."""
    return value.date() if isinstance(value, datetime) else value


class MongoAttendanceRepository(AttendanceRepository):
    backend = "mongodb"

    def __init__(self, school_id: str | None = None) -> None:
        #: Every query is scoped to this school - the isolation guarantee.
        self.school_id = school_id or settings.school_id

    # ------------------------------------------------------------------
    # Document -> record mappers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_student(doc: dict, face_count: int = 0) -> StudentRecord:
        return StudentRecord(
            id=int(doc["studentId"]),
            roll_number=doc.get("rollNumber", ""),
            name=doc.get("name", ""),
            class_name=doc.get("className", ""),
            section=doc.get("section", ""),
            created_at=_as_datetime(doc.get("createdAt")),
            updated_at=_as_datetime(doc.get("updatedAt")),
            face_count=face_count,
        )

    @staticmethod
    def _to_face(doc: dict) -> FaceRecord:
        return FaceRecord(
            id=int(doc["faceId"]),
            student_id=int(doc["studentId"]),
            embedding=np.asarray(doc.get("embedding", []), dtype=np.float32).ravel(),
            det_score=float(doc.get("detScore") or 0.0),
            created_at=_as_datetime(doc.get("createdAt")),
        )

    def _to_attendance(self, doc: dict) -> AttendanceRecord:
        student = self.get_student(int(doc["studentId"]))
        return AttendanceRecord(
            id=int(doc["attendanceId"]),
            student_id=int(doc["studentId"]),
            class_name=doc.get("className", ""),
            section=doc.get("section", ""),
            attendance_date=_as_date(doc.get("attendanceDate")),
            status=doc.get("status", "present"),
            confidence=doc.get("confidence"),
            created_at=_as_datetime(doc.get("createdAt")),
            updated_at=_as_datetime(doc.get("updatedAt")),
            roll_number=student.roll_number if student else "",
            student_name=student.name if student else "",
        )

    @staticmethod
    def _to_audit(doc: dict) -> AuditRecord:
        return AuditRecord(
            id=int(doc["auditId"]),
            attendance_id=doc.get("attendanceId"),
            student_id=int(doc["studentId"]),
            action=doc.get("action", ""),
            old_status=doc.get("oldStatus"),
            new_status=doc.get("newStatus"),
            actor=doc.get("actor", "anonymous"),
            confidence=doc.get("confidence"),
            created_at=_as_datetime(doc.get("createdAt")),
        )

    # -- students ----------------------------------------------------------
    def create_student(
        self, *, roll_number: str, name: str, class_name: str, section: str
    ) -> StudentRecord:
        now = utcnow()
        document = {
            "schoolId": self.school_id,
            "studentId": _next_id("student"),
            "rollNumber": roll_number,
            "name": name,
            "className": class_name,
            "section": section,
            "createdAt": now,
            "updatedAt": now,
        }
        try:
            collection(STUDENTS).insert_one(document)
        except DuplicateKeyError as exc:
            # The unique index on (schoolId, rollNumber) is the source of truth.
            logger.warning("Duplicate roll number %r in school %s", roll_number, self.school_id)
            raise DuplicateKeyError(
                f"Roll number {roll_number} already exists in school {self.school_id}"
            ) from exc
        return self._to_student(document)

    def get_student(self, student_id: int) -> StudentRecord | None:
        doc = collection(STUDENTS).find_one(
            {"schoolId": self.school_id, "studentId": int(student_id)}
        )
        if doc is None:
            return None
        return self._to_student(doc, self.count_faces(student_id))

    def get_student_by_roll(
        self, roll_number: str, class_name: str, section: str
    ) -> StudentRecord | None:
        doc = collection(STUDENTS).find_one(
            {
                "schoolId": self.school_id,
                "rollNumber": roll_number,
                "className": class_name,
                "section": section,
            }
        )
        return self._to_student(doc) if doc else None

    def _student_filter(
        self, class_name: str | None, section: str | None
    ) -> dict:
        criteria: dict = {"schoolId": self.school_id}
        if class_name is not None:
            criteria["className"] = class_name
        if section is not None:
            criteria["section"] = section
        return criteria

    def list_students(
        self,
        *,
        class_name: str | None = None,
        section: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StudentRecord]:
        cursor = (
            collection(STUDENTS)
            .find(self._student_filter(class_name, section))
            .sort([("className", 1), ("section", 1), ("studentId", 1)])
            .skip(max(0, offset))
            .limit(limit)
        )
        return [self._to_student(doc, self.count_faces(int(doc["studentId"]))) for doc in cursor]

    def count_students(
        self, *, class_name: str | None = None, section: str | None = None
    ) -> int:
        return int(
            collection(STUDENTS).count_documents(
                self._student_filter(class_name, section)
            )
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
        updates: dict = {}
        if name is not None:
            updates["name"] = name
        if roll_number is not None:
            updates["rollNumber"] = roll_number
        if class_name is not None:
            updates["className"] = class_name
        if section is not None:
            updates["section"] = section
        if not updates:
            return self.get_student(student_id)

        updates["updatedAt"] = utcnow()
        doc = collection(STUDENTS).find_one_and_update(
            {"schoolId": self.school_id, "studentId": int(student_id)},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            return None
        return self._to_student(doc, self.count_faces(student_id))

    def delete_student(self, student_id: int) -> bool:
        criteria = {"schoolId": self.school_id, "studentId": int(student_id)}
        if collection(STUDENTS).delete_one(criteria).deleted_count == 0:
            return False
        collection(FACE_EMBEDDINGS).delete_many(criteria)
        collection(ATTENDANCE).delete_many(criteria)
        collection(ATTENDANCE_AUDIT).delete_many(criteria)
        return True

    def list_classes(self) -> list[tuple[str, str, int]]:
        """Aggregate class/section head-counts straight in MongoDB."""
        pipeline = [
            {"$match": {"schoolId": self.school_id}},
            {
                "$group": {
                    "_id": {"className": "$className", "section": "$section"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id.className": 1, "_id.section": 1}},
        ]
        return [
            (
                row["_id"].get("className", ""),
                row["_id"].get("section", ""),
                int(row["count"]),
            )
            for row in collection(STUDENTS).aggregate(pipeline)
        ]

    # -- face embeddings ---------------------------------------------------
    def add_face(
        self, student_id: int, embedding: np.ndarray, det_score: float
    ) -> FaceRecord:
        document = {
            "schoolId": self.school_id,
            "faceId": _next_id("face"),
            "studentId": int(student_id),
            # Stored L2-normalised so cosine similarity is a plain dot product.
            "embedding": l2_normalize(embedding).astype(float).tolist(),
            "detScore": float(det_score),
            "createdAt": utcnow(),
        }
        collection(FACE_EMBEDDINGS).insert_one(document)
        return self._to_face(document)

    def list_faces(self, student_id: int) -> list[FaceRecord]:
        cursor = collection(FACE_EMBEDDINGS).find(
            {"schoolId": self.school_id, "studentId": int(student_id)}
        ).sort("faceId", 1)
        return [self._to_face(doc) for doc in cursor]

    def count_faces(self, student_id: int) -> int:
        return int(
            collection(FACE_EMBEDDINGS).count_documents(
                {"schoolId": self.school_id, "studentId": int(student_id)}
            )
        )

    def delete_face(self, face_id: int) -> bool:
        return (
            collection(FACE_EMBEDDINGS).delete_one(
                {"schoolId": self.school_id, "faceId": int(face_id)}
            ).deleted_count
            > 0
        )

    def delete_faces(self, student_id: int) -> int:
        return int(
            collection(FACE_EMBEDDINGS).delete_many(
                {"schoolId": self.school_id, "studentId": int(student_id)}
            ).deleted_count
        )

    def search_similar(
        self,
        embedding: np.ndarray,
        *,
        limit: int = 5,
        exclude_student_id: int | None = None,
        class_name: str | None = None,
        section: str | None = None,
    ) -> list[tuple[FaceRecord, float]]:
        """Nearest neighbours by cosine similarity, scoped to this school.

        When a class is given, the class's student ids are resolved first and
        passed into the vector-search filter, so the candidate set is restricted
        to that class *during* the search rather than filtered afterwards.
        """
        query = l2_normalize(embedding).astype(float).tolist()
        criteria: dict = {"schoolId": self.school_id}

        if class_name is not None:
            student_ids = self._class_student_ids(class_name, section)
            if not student_ids:
                return []
            criteria["studentId"] = {"$in": student_ids}

        if exclude_student_id is not None:
            criteria["studentId"] = (
                {"$ne": int(exclude_student_id)}
                if "studentId" not in criteria
                else {"$in": [i for i in criteria["studentId"]["$in"]
                              if i != int(exclude_student_id)]}
            )

        if vector_search_supported():
            results = self._vector_search(query, criteria, limit)
            if results is not None:
                return results

        return self._exact_search(query, criteria, limit)

    def _class_student_ids(
        self, class_name: str, section: str | None
    ) -> list[int]:
        """Ids of every student in the given class/section of this school."""
        criteria: dict = {"schoolId": self.school_id, "className": class_name}
        if section is not None:
            criteria["section"] = section
        return [
            int(doc["studentId"])
            for doc in collection(STUDENTS).find(criteria, {"studentId": 1})
        ]

    def _vector_search(
        self, query: list[float], criteria: dict, limit: int
    ) -> list[tuple[FaceRecord, float]] | None:
        """MongoDB ``$vectorSearch``. Returns None when unsupported/unavailable."""
        # $vectorSearch is a top-level stage: the school filter is passed in as
        # `filter`, never as a preceding $match.
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query,
                    "numCandidates": max(limit * 10, 100),
                    "limit": max(1, limit),
                    "similarity": "cosine",
                    "filter": criteria,
                }
            },
            {"$project": {"_id": 0, "embedding": 0}},
        ]
        try:
            docs = list(collection(FACE_EMBEDDINGS).aggregate(pipeline))
        except Exception as exc:  # noqa: BLE001 - degrade instead of failing
            logger.warning("$vectorSearch failed (%s); using exact search", exc)
            return None

        results: list[tuple[FaceRecord, float]] = []
        for doc in docs:
            # For `"similarity": "cosine"` MongoDB Atlas returns `score` as the
            # cosine SIMILARITY (1.0 = identical, higher = more similar), NOT a
            # distance - so use it directly. Verified against Atlas docs; the
            # old `1.0 - score` inversion rejected the best matches outright.
            similarity = float(doc.get("score", 0.0))
            results.append((self._to_face(doc), similarity))
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results

    def _exact_search(
        self, query: list[float], criteria: dict, limit: int
    ) -> list[tuple[FaceRecord, float]]:
        """Exact cosine similarity over this school's embeddings (fallback)."""
        cursor = collection(FACE_EMBEDDINGS).find(criteria, {"_id": 0})
        rows: list[tuple[FaceRecord, np.ndarray]] = []
        for doc in cursor:
            vector = np.asarray(doc.get("embedding", []), dtype=np.float32)
            if vector.size:
                rows.append((self._to_face(doc), vector))
        if not rows:
            return []

        matrix = np.stack([vector for _, vector in rows])
        # Both sides are L2-normalised, so cosine similarity is a dot product.
        similarities = matrix @ np.asarray(query, dtype=np.float32)
        order = np.argsort(-similarities)[: max(1, limit)]
        return [(rows[i][0], float(similarities[i])) for i in order]

    def count_embeddings(self) -> int:
        return int(
            collection(FACE_EMBEDDINGS).count_documents({"schoolId": self.school_id})
        )

    # -- attendance --------------------------------------------------------
    def get_attendance(self, attendance_id: int) -> AttendanceRecord | None:
        doc = collection(ATTENDANCE).find_one(
            {"schoolId": self.school_id, "attendanceId": int(attendance_id)}
        )
        return self._to_attendance(doc) if doc else None

    def _attendance_criteria(
        self,
        student_id: int,
        class_name: str,
        section: str,
        attendance_date: date,
    ) -> dict:
        return {
            "schoolId": self.school_id,
            "studentId": int(student_id),
            "className": class_name,
            "section": section,
            "attendanceDate": datetime.combine(attendance_date, datetime.min.time()),
        }

    def find_attendance(
        self,
        *,
        student_id: int,
        class_name: str,
        section: str,
        attendance_date: date,
    ) -> AttendanceRecord | None:
        doc = collection(ATTENDANCE).find_one(
            self._attendance_criteria(student_id, class_name, section, attendance_date)
        )
        return self._to_attendance(doc) if doc else None

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
        now = utcnow()
        document = {
            "schoolId": self.school_id,
            "attendanceId": _next_id("attendance"),
            "studentId": int(student_id),
            "className": class_name,
            "section": section,
            "attendanceDate": datetime.combine(attendance_date, datetime.min.time()),
            "status": status,
            "confidence": confidence,
            "createdAt": now,
            "updatedAt": now,
        }
        try:
            collection(ATTENDANCE).insert_one(document)
        except DuplicateKeyError:
            # The unique index did its job: the row already exists.
            existing = self.find_attendance(
                student_id=student_id,
                class_name=class_name,
                section=section,
                attendance_date=attendance_date,
            )
            if existing is None:  # pragma: no cover - index says otherwise
                raise
            return existing, False

        self._audit(
            attendance_id=document["attendanceId"],
            student_id=int(student_id),
            action="created",
            old_status=None,
            new_status=status,
            actor=actor,
            confidence=confidence,
        )
        return self._to_attendance(document), True

    def update_attendance(
        self, attendance_id: int, *, status: str, actor: str
    ) -> tuple[AttendanceRecord | None, str | None]:
        doc = collection(ATTENDANCE).find_one_and_update(
            {"schoolId": self.school_id, "attendanceId": int(attendance_id)},
            {"$set": {"status": status, "updatedAt": utcnow()}},
            return_document=ReturnDocument.BEFORE,
        )
        if doc is None:
            return None, None

        previous = doc.get("status")
        if previous != status:
            self._audit(
                attendance_id=int(attendance_id),
                student_id=int(doc["studentId"]),
                action="manual_update",
                old_status=previous,
                new_status=status,
                actor=actor,
            )
        updated = collection(ATTENDANCE).find_one(
            {"schoolId": self.school_id, "attendanceId": int(attendance_id)}
        )
        return (self._to_attendance(updated) if updated else None), previous

    def list_attendance(
        self,
        *,
        class_name: str,
        section: str,
        attendance_date: date | None = None,
    ) -> list[AttendanceRecord]:
        criteria: dict = {
            "schoolId": self.school_id,
            "className": class_name,
            "section": section,
        }
        if attendance_date is not None:
            criteria["attendanceDate"] = datetime.combine(
                attendance_date, datetime.min.time()
            )
        cursor = collection(ATTENDANCE).find(criteria).sort(
            [("attendanceDate", 1), ("studentId", 1)]
        )
        return [self._to_attendance(doc) for doc in cursor]

    # -- audit -------------------------------------------------------------
    def _audit(
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
        collection(ATTENDANCE_AUDIT).insert_one(
            {
                "schoolId": self.school_id,
                "auditId": _next_id("audit"),
                "attendanceId": attendance_id,
                "studentId": int(student_id),
                "action": action,
                "oldStatus": old_status,
                "newStatus": new_status,
                "actor": actor,
                "confidence": confidence,
                "createdAt": utcnow(),
            }
        )

    def list_audit(
        self, *, attendance_id: int | None = None, limit: int = 100
    ) -> list[AuditRecord]:
        criteria: dict = {"schoolId": self.school_id}
        if attendance_id is not None:
            criteria["attendanceId"] = int(attendance_id)
        cursor = (
            collection(ATTENDANCE_AUDIT)
            .find(criteria)
            .sort("auditId", -1)
            .limit(limit)
        )
        return [self._to_audit(doc) for doc in cursor]

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Delete every row of the current school. Destructive: scripts only."""
        criteria = {"schoolId": self.school_id}
        for name in (ATTENDANCE_AUDIT, ATTENDANCE, FACE_EMBEDDINGS, STUDENTS):
            collection(name).delete_many(criteria)
        logger.warning("Cleared all data for school %s", self.school_id)