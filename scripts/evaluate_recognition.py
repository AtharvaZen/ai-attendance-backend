#!/usr/bin/env python
"""Threshold sweep over a labelled evaluation set - reports results, chooses nothing.

Layout it expects::

    test_data/students/<roll>_<Name>/*.jpg   -> registration photos (3-5 each)
    test_data/eval/<roll>_<Name>/*.jpg       -> test photos of KNOWN students
    test_data/eval/unknown/*.jpg             -> test photos of UNKNOWN people

For every threshold it reports true/false positives and negatives, accuracy,
precision, recall, false-accept rate and average processing time, so the
threshold can be chosen from evidence instead of a guess.

Definitions (open-set identification)::

    TP  ground truth known   -> predicted as that same student
    FN  ground truth known   -> predicted as another student, or as unknown
    TN  ground truth unknown -> predicted as unknown
    FP  ground truth unknown -> predicted as some student        (false accept)

Embeddings are computed **once** per image; only the thresholded decision is
re-evaluated, so the sweep is fast.

Usage:
    python scripts/evaluate_recognition.py
    python scripts/evaluate_recognition.py --thresholds 0.35,0.40,0.45,0.50,0.55,0.60
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import configure_logging, settings  # noqa: E402
from app.database.repository import AttendanceRepository, get_repository  # noqa: E402
from app.services.embedding_service import embedding_service  # noqa: E402
from app.services.face_service import face_service, read_image_file  # noqa: E402

TEST_DATA = BACKEND_DIR / "test_data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
CLASS_NAME = "8"
SECTION = "A"


@dataclass(slots=True)
class Observation:
    """What the model saw for one single-face test image."""

    truth_roll: str | None  # None => the person is not a registered student
    best_roll: str | None
    similarity: float
    runner_up: float | None
    processing_ms: float


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def register_students(repository: AttendanceRepository, students_dir: Path) -> int:
    count = 0
    for folder in sorted(p for p in students_dir.iterdir() if p.is_dir()):
        roll, _, raw_name = folder.name.partition("_")
        images = _images(folder)
        if not images:
            continue
        student = repository.create_student(
            roll_number=roll,
            name=raw_name.replace("_", " ") or f"Student {roll}",
            class_name=CLASS_NAME,
            section=SECTION,
        )
        for image_path in images:
            face = embedding_service.single_face_embedding(read_image_file(image_path))
            repository.add_face(student.id, face.embedding, face.det_score)
        count += 1
    return count


def collect_observations(
    repository: AttendanceRepository, eval_dir: Path
) -> tuple[list[Observation], int]:
    """Run detection + embedding + search once per image."""
    observations: list[Observation] = []
    skipped = 0

    for folder in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        truth_roll = None if folder.name.lower() == "unknown" else folder.name.partition("_")[0]
        for image_path in _images(folder):
            image = read_image_file(image_path)
            started = time.perf_counter()
            faces = face_service.detect(image)
            if len(faces) != 1:
                # The eval set is single-face by design; multi-face images need
                # bounding-box level ground truth, which this script does not model.
                skipped += 1
                print(f"  ! {folder.name}/{image_path.name}: {len(faces)} faces, skipped")
                continue

            candidates = repository.search_similar(
                faces[0].embedding, limit=settings.recognition_candidates
            )
            if candidates:
                best_face, best_sim = candidates[0]
                best_student = repository.get_student(best_face.student_id)
                runner_up = next(
                    (
                        sim
                        for candidate, sim in candidates[1:]
                        if candidate.student_id != best_face.student_id
                    ),
                    None,
                )
                best_roll = best_student.roll_number if best_student else None
            else:
                best_sim, best_roll, runner_up = 0.0, None, None

            observations.append(
                Observation(
                    truth_roll=truth_roll,
                    best_roll=best_roll,
                    similarity=best_sim,
                    runner_up=runner_up,
                    processing_ms=(time.perf_counter() - started) * 1000,
                )
            )
    return observations, skipped


def evaluate(observations: list[Observation], threshold: float) -> dict[str, float]:
    """Apply one threshold to the pre-computed observations."""
    tp = fn_rejected = fn_wrong = tn = fp = 0
    for obs in observations:
        predicted = (
            obs.best_roll
            if obs.best_roll is not None and obs.similarity >= threshold
            else None
        )
        if obs.truth_roll is None:
            if predicted is None:
                tn += 1
            else:
                fp += 1
        elif predicted == obs.truth_roll:
            tp += 1
        elif predicted is None:
            fn_rejected += 1
        else:
            fn_wrong += 1

    total = len(observations)
    fn = fn_rejected + fn_wrong
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn_rejected": fn_rejected,
        "fn_wrong": fn_wrong,
        "tn": tn,
        "total": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "far": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{t:.2f}" for t in DEFAULT_THRESHOLDS),
        help="comma-separated thresholds to sweep",
    )
    parser.add_argument("--students-dir", default=str(TEST_DATA / "students"))
    parser.add_argument("--eval-dir", default=str(TEST_DATA / "eval"))
    parser.add_argument(
        "--explain",
        action="store_true",
        help="also print every test image with its truth, prediction and similarity",
    )
    args = parser.parse_args(argv)

    thresholds = sorted({float(t) for t in args.thresholds.split(",") if t.strip()})

    configure_logging()
    print(f"\nLoading pretrained model '{settings.face_model_name}' ...")
    face_service.load()

    repository = get_repository()
    print(f"Repository: {repository.backend}")
    repository.reset()

    students_dir, eval_dir = Path(args.students_dir), Path(args.eval_dir)
    if not students_dir.is_dir() or not eval_dir.is_dir():
        print("test_data/ not found. Run: python scripts/make_demo_dataset.py")
        return 1

    print(f"\n--- Registering students from {students_dir.name}/ ---")
    registered = register_students(repository, students_dir)
    print(f"  {registered} students, {repository.count_embeddings()} embeddings")

    print(f"\n--- Evaluating images in {eval_dir.name}/ ---")
    observations, skipped = collect_observations(repository, eval_dir)
    known = sum(1 for o in observations if o.truth_roll is not None)
    unknown = len(observations) - known
    print(f"  {known} known-person images, {unknown} unknown-person images"
          + (f", {skipped} skipped" if skipped else ""))
    if not observations:
        print("Nothing to evaluate.")
        return 1

    if args.explain:
        print("\n--- Per-image detail ---")
        for obs in sorted(observations, key=lambda o: -o.similarity):
            truth = obs.truth_roll or "unknown"
            print(
                f"  truth={truth:<5} best={str(obs.best_roll or '-'):<5} "
                f"sim={obs.similarity:.3f} runner_up="
                f"{'-' if obs.runner_up is None else f'{obs.runner_up:.3f}'}"
            )

    rows = [evaluate(observations, t) for t in thresholds]
    avg_ms = sum(o.processing_ms for o in observations) / len(observations)

    print("\n" + "=" * 100)
    print("Threshold sweep (detection + embedding + search)")
    print("=" * 100)
    print(
        f"{'thr':>5} {'TP':>4} {'FP':>4} {'FN(rej)':>8} {'FN(wrong)':>10} "
        f"{'TN':>4} {'accuracy':>9} {'precision':>10} {'recall':>7} {'FAR':>6}"
    )
    print("-" * 100)
    for row in rows:
        print(
            f"{row['threshold']:>5.2f} {row['tp']:>4.0f} {row['fp']:>4.0f} "
            f"{row['fn_rejected']:>8.0f} {row['fn_wrong']:>10.0f} {row['tn']:>4.0f} "
            f"{row['accuracy']:>9.3f} {row['precision']:>10.3f} "
            f"{row['recall']:>7.3f} {row['far']:>6.3f}"
        )
    print("=" * 100)
    print(f"Average processing time per image: {avg_ms:.1f} ms")
    print(
        "\nNotes:\n"
        "  * FN(rej)  = a known student rejected as unknown (too strict).\n"
        "  * FN(wrong)= a known student matched to the WRONG student (worst case).\n"
        "  * FP       = an unknown person accepted as a student (false accept).\n"
        "  * This script reports results only - it does not pick a threshold for\n"
        "    you. Choose based on which error you can least afford, then set\n"
        "    FACE_MATCH_THRESHOLD in .env.\n"
        "  * 'similarity' is a cosine similarity between embeddings, NOT a\n"
        "    probability that the face belongs to the student."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
