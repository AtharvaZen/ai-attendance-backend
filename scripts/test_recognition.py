#!/usr/bin/env python
"""End-to-end recognition test - no frontend, no HTTP server required.

Registers every student found in ``test_data/students/`` (3+ photos each), then
recognises the faces in every image in ``test_data/attendance/`` and prints the
similarity score for each detected face.

Usage:
    python scripts/test_recognition.py
    python scripts/test_recognition.py --threshold 0.40
    python scripts/test_recognition.py --image path/to/classroom.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import configure_logging, settings  # noqa: E402
from app.database.repository import AttendanceRepository, get_repository  # noqa: E402
from app.services.embedding_service import embedding_service  # noqa: E402
from app.services.face_service import face_service, read_image_file  # noqa: E402
from app.services.matching_service import MatchingService  # noqa: E402

TEST_DATA = BACKEND_DIR / "test_data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAME = "8"
SECTION = "A"


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def register_students(
    repository: AttendanceRepository, students_dir: Path
) -> dict[str, int]:
    """Register every ``<roll>_<Name>`` folder. Returns ``{roll: student_id}``."""
    registered: dict[str, int] = {}
    for folder in sorted(p for p in students_dir.iterdir() if p.is_dir()):
        roll, _, raw_name = folder.name.partition("_")  # 101_Rahul_Sharma
        name = raw_name.replace("_", " ") or f"Student {roll}"
        images = _images(folder)
        if not images:
            print(f"  ! {folder.name}: no images, skipped")
            continue

        student = repository.create_student(
            roll_number=roll, name=name, class_name=CLASS_NAME, section=SECTION
        )
        stored = 0
        for image_path in images:
            try:
                face = embedding_service.single_face_embedding(
                    read_image_file(image_path)
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  ! {image_path.name}: {exc}")
                continue
            repository.add_face(student.id, face.embedding, face.det_score)
            stored += 1

        registered[roll] = student.id
        print(f"  {roll:<4} {name:<18} id={student.id:<3} embeddings={stored}")
    return registered


def recognise_image(
    repository: AttendanceRepository, image_path: Path, threshold: float
) -> tuple[int, int, int]:
    """Print per-face results for one image. Returns (detected, known, unknown)."""
    image = read_image_file(image_path)
    faces = face_service.detect(image)
    matcher = MatchingService(repository, threshold=threshold)
    matches = matcher.match_faces(faces)

    print(f"\n{'=' * 62}")
    print(
        f"Image: {image_path.relative_to(BACKEND_DIR)}  "
        f"({image.shape[1]}x{image.shape[0]}, {len(faces)} faces)"
    )
    print("=" * 62)

    known = 0
    for index, match in enumerate(matches, 1):
        student = match.student
        if student is not None:
            known += 1
        print(f"\nFace #{index}")
        if student is not None:
            print(f"  Matched   : {student.name}")
            print(f"  Roll      : {student.roll_number}")
        else:
            print("  Matched   : Unknown")
        print(f"  Similarity: {match.similarity:.2f}   (threshold {threshold:.2f})")
        if match.runner_up_similarity is not None:
            print(f"  Runner-up : {match.runner_up_similarity:.2f}")

    return len(faces), known, len(faces) - known


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.face_match_threshold,
        help=f"cosine-similarity threshold (default {settings.face_match_threshold})",
    )
    parser.add_argument("--image", help="recognise one image instead of the folder")
    parser.add_argument("--students-dir", default=str(TEST_DATA / "students"))
    parser.add_argument("--attendance-dir", default=str(TEST_DATA / "attendance"))
    args = parser.parse_args(argv)

    configure_logging()
    print(f"\nLoading pretrained model '{settings.face_model_name}' ...")
    face_service.load()
    print(f"Model ready | providers={face_service.providers}")

    repository = get_repository()
    print(f"Repository: {repository.backend}")
    repository.reset()

    print(f"\n--- Registering students from {Path(args.students_dir).name}/ ---")
    registered = register_students(repository, Path(args.students_dir))
    if not registered:
        print("No students registered. Run scripts/make_demo_dataset.py first.")
        return 1
    print(
        f"\nRegistered {len(registered)} students, "
        f"{repository.count_embeddings()} embeddings total"
    )

    # Resolve so `--image test_data/...` (relative) works with the
    # relative_to(BACKEND_DIR) print below - previously it crashed.
    images = [Path(args.image).resolve()] if args.image else _images(Path(args.attendance_dir))
    if not images:
        print(f"No images found in {args.attendance_dir}. Run the demo dataset script.")
        return 1

    total_detected = total_known = total_unknown = 0
    for image_path in images:
        detected, known, unknown = recognise_image(
            repository, image_path, args.threshold
        )
        total_detected += detected
        total_known += known
        total_unknown += unknown

    print(f"\n{'=' * 62}")
    print("Summary:")
    print(f"  Detected  : {total_detected}")
    print(f"  Recognized: {total_known}")
    print(f"  Unknown   : {total_unknown}")
    print(f"  Threshold : {args.threshold:.2f}")
    print("=" * 62)
    print(
        "\nReminder: 'similarity' is the cosine similarity between two embeddings,\n"
        "not a probability that the face belongs to that student.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
