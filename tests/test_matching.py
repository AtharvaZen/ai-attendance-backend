"""Unit tests for the two-gate recognition decision (threshold + margin).

These tests use synthetic embeddings so they run without the InsightFace
model: unit vectors are crafted so the cosine similarity between them is a
chosen number (cosine similarity of two L2-normalised vectors is their dot
product).

Run from the repository root:

    python -m unittest discover backend/tests
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from app.core.config import settings
from app.database.memory_repository import InMemoryAttendanceRepository
from app.services.attendance_service import AttendanceService
from app.services.face_service import DetectedFace
from app.services.matching_service import MatchingService


def unit(sim_x: float) -> np.ndarray:
    """A 2-D unit vector whose similarity to ``[1, 0]`` is ``sim_x``."""
    return np.array([sim_x, float(np.sqrt(max(0.0, 1.0 - sim_x**2)))], dtype=np.float32)


def face(embedding: np.ndarray | None) -> DetectedFace:
    return DetectedFace(bbox=(0.0, 0.0, 100.0, 100.0), det_score=0.99, embedding=embedding)


class MatchingGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = InMemoryAttendanceRepository()
        self.alice = self.repo.create_student(
            roll_number="101", name="Alice", class_name="8", section="A"
        )
        self.bob = self.repo.create_student(
            roll_number="102", name="Bob", class_name="8", section="A"
        )

    def _register(self, student_id: int, sim_to_query: float) -> None:
        self.repo.add_face(student_id, unit(sim_to_query), det_score=0.99)

    # --- gate 1: absolute threshold -------------------------------------
    def test_below_threshold_is_unknown(self) -> None:
        self._register(self.alice.id, sim_to_query=0.40)  # < 0.45
        match = MatchingService(self.repo).match_face(face(unit(1.0)))
        self.assertFalse(match.accepted)
        self.assertIsNone(match.student)

    def test_above_threshold_and_clear_margin_is_accepted(self) -> None:
        self._register(self.alice.id, sim_to_query=0.90)
        self._register(self.bob.id, sim_to_query=0.40)
        match = MatchingService(self.repo).match_face(face(unit(1.0)))
        self.assertTrue(match.accepted)
        self.assertIsNotNone(match.student)
        self.assertEqual(match.student.id, self.alice.id)
        self.assertAlmostEqual(match.similarity, 0.90, places=3)
        # Best other student must be Bob at 0.40, not Alice's own second face.
        self.assertAlmostEqual(match.runner_up_similarity or 0.0, 0.40, places=3)
        self.assertAlmostEqual(match.margin or 0.0, 0.50, places=3)

    # --- gate 2: the false-positive fix ---------------------------------
    def test_small_margin_is_rejected_even_above_threshold(self) -> None:
        """0.50 vs 0.49 clears the threshold but must NOT be accepted."""
        self._register(self.alice.id, sim_to_query=0.50)
        self._register(self.bob.id, sim_to_query=0.49)
        match = MatchingService(self.repo).match_face(face(unit(1.0)))
        self.assertGreaterEqual(match.similarity, settings.face_match_threshold)
        self.assertFalse(match.accepted)
        self.assertIsNone(match.student)  # rejected -> reported as unknown
        self.assertAlmostEqual(match.margin or 0.0, 0.01, places=3)

    def test_margin_gate_disabled_with_zero(self) -> None:
        self._register(self.alice.id, sim_to_query=0.50)
        self._register(self.bob.id, sim_to_query=0.49)
        match = MatchingService(self.repo, margin=0.0).match_face(face(unit(1.0)))
        self.assertTrue(match.accepted)

    def test_lone_candidate_needs_only_threshold(self) -> None:
        self._register(self.alice.id, sim_to_query=0.60)
        match = MatchingService(self.repo).match_face(face(unit(1.0)))
        self.assertTrue(match.accepted)
        self.assertIsNone(match.runner_up_similarity)  # no rival -> no margin test

    # --- degenerate inputs ----------------------------------------------
    def test_face_without_embedding_is_unknown(self) -> None:
        match = MatchingService(self.repo).match_face(face(None))
        self.assertFalse(match.accepted)
        self.assertEqual(match.similarity, 0.0)

    def test_empty_repository_is_unknown(self) -> None:
        empty = InMemoryAttendanceRepository()
        match = MatchingService(empty).match_face(face(unit(1.0)))
        self.assertFalse(match.accepted)

    # --- class scoping ----------------------------------------------------
    def test_class_scoped_search_ignores_other_classes(self) -> None:
        outsider = self.repo.create_student(
            roll_number="201", name="Zoe", class_name="9", section="B"
        )
        self.repo.add_face(outsider.id, unit(1.0), det_score=0.99)  # perfect match
        match = MatchingService(
            self.repo, class_name="8", section="A"
        ).match_face(face(unit(1.0)))
        self.assertFalse(match.accepted)  # excluded by class scope -> unknown

    # --- identify() contract ---------------------------------------------
    def test_identify_dedupes_and_counts(self) -> None:
        self._register(self.alice.id, sim_to_query=0.90)
        self._register(self.bob.id, sim_to_query=0.30)
        faces = [
            face(unit(1.0)),   # accepted (Alice)
            face(unit(1.0)),   # duplicate of Alice
            face(unit(1.0)),   # duplicate of Alice
            # Orthogonal-ish vector: scores NEGATIVELY against every stored
            # embedding (worst case -0.3), i.e. far below the threshold.
            face(np.array([-1.0, 0.0], dtype=np.float32)),  # -> unknown
        ]
        result = MatchingService(self.repo).identify(faces)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].student.id, self.alice.id)
        self.assertEqual(result.duplicates, 2)
        self.assertEqual(result.unknown, 1)
        # Every detected face is preserved for audit/debug output.
        self.assertEqual(len(result.all_matches), len(faces))
        decisions = [m.accepted for m in result.all_matches]
        self.assertEqual(decisions, [True, True, True, False])

    def test_identify_reports_margin_rejection_as_unknown(self) -> None:
        self._register(self.alice.id, sim_to_query=0.50)
        self._register(self.bob.id, sim_to_query=0.49)
        result = MatchingService(self.repo).identify([face(unit(1.0))])
        self.assertEqual(len(result.matches), 0)
        self.assertEqual(result.unknown, 1)
        self.assertFalse(result.all_matches[0].accepted)

    # --- configuration sanity --------------------------------------------
    def test_default_settings(self) -> None:
        self.assertEqual(settings.face_match_threshold, 0.45)
        self.assertEqual(settings.face_match_margin, 0.05)
        # The candidate window must leave room for a non-best student so the
        # margin gate is not blind to the runner-up.
        self.assertGreaterEqual(
            settings.recognition_candidates, settings.faces_per_student_max + 1
        )


class CrossStudentSearchTests(unittest.TestCase):
    """The registration guard relies on search_similar excluding the owner."""

    def test_exclude_student_id_returns_only_others(self) -> None:
        repo = InMemoryAttendanceRepository()
        alice = repo.create_student(
            roll_number="101", name="Alice", class_name="8", section="A"
        )
        bob = repo.create_student(
            roll_number="102", name="Bob", class_name="8", section="A"
        )
        repo.add_face(alice.id, unit(0.99), det_score=0.99)
        repo.add_face(bob.id, unit(0.30), det_score=0.99)

        results = repo.search_similar(unit(1.0), limit=5, exclude_student_id=alice.id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].student_id, bob.id)
        self.assertLess(results[0][1], settings.duplicate_face_similarity)


class AttendanceServiceContractTests(unittest.TestCase):
    """Regression tests for the bug fixed in attendance_service.

    ``confirm`` must be a method of :class:`AttendanceService` (it was
    accidentally dedented to module level, which silently stripped the class
    of its confirm step) and must persist teacher-approved entries.
    """

    def setUp(self) -> None:
        self.repo = InMemoryAttendanceRepository()
        self.student = self.repo.create_student(
            roll_number="101", name="Alice", class_name="8", section="A"
        )
        self.service = AttendanceService(self.repo)

    def test_confirm_is_a_method(self) -> None:
        sig = inspect.signature(AttendanceService.confirm)
        self.assertIn("self", sig.parameters)
        self.assertTrue(callable(self.service.confirm))

    def test_confirm_persists_entries(self) -> None:
        from datetime import date

        from app.database.records import STATUS_PRESENT
        from app.services.attendance_service import ConfirmedEntry

        outcome = self.service.confirm(
            class_name="8",
            section="A",
            attendance_date=date(2026, 10, 4),
            entries=[
                ConfirmedEntry(
                    student_id=self.student.id, status=STATUS_PRESENT, confidence=0.9
                )
            ],
            actor="test",
        )
        self.assertEqual(outcome.saved, 1)
        self.assertEqual(outcome.created, 1)
        rows = self.repo.list_attendance(
            class_name="8", section="A", attendance_date=date(2026, 10, 4)
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, STATUS_PRESENT)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
