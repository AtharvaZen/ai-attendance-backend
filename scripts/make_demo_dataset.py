#!/usr/bin/env python
"""Build a runnable demo dataset under ``test_data/``.

Real face photos cannot be shipped with the repository (privacy), so this script
derives one from the sample images bundled with InsightFace:

* 6 identities, cropped from InsightFace's group photo ``t1.jpg``
* 1 unrelated identity (``Tom_Hanks_54745.png``) used as the *unknown* person.

It writes::

    test_data/students/<roll>_<Name>/ 01.jpg 02.jpg 03.jpg      (registration)
    test_data/eval/<roll>_<Name>/     01..06.jpg                (accuracy test)
    test_data/eval/unknown/           01..04.jpg
    test_data/attendance/classroom_01.jpg, classroom_02.jpg, classroom_unknown.jpg

.. warning::
   The eval images are synthetic variations (brightness / rotation / scale /
   blur) of the *same* source photos used for registration, so accuracy figures
   produced from them are optimistic. Replace ``test_data/`` with genuinely
   different photos of real people before trusting any threshold.

Usage:
    python scripts/make_demo_dataset.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import configure_logging  # noqa: E402
from app.services.face_service import (  # noqa: E402
    cosine_similarity,
    face_service,
    write_image_file,
)

TEST_DATA = BACKEND_DIR / "test_data"

# Roll number + display name for each face found in the group photo.
IDENTITIES = [
    (101, "Rahul_Sharma"),
    (102, "Aman_Verma"),
    (103, "Priya_Nair"),
    (104, "Sneha_Iyer"),
    (105, "Vikram_Singh"),
    (106, "Kavya_Reddy"),
]

# (brightness, rotation_deg, scale, blur_ksize, dx, dy)
#
# Candidates for the registration photos. The generator keeps only those whose
# embeddings are mutually dissimilar enough (see MAX_REGISTRATION_SIMILARITY),
# because photos derived from a single source image are otherwise near-identical.
CANDIDATE_VARIANTS = [
    (1.00, 0.0, 1.00, 0, 0.00, 0.00),
    (1.35, -12.0, 1.15, 1, 0.03, 0.02),
    (0.70, 14.0, 0.92, 1, -0.04, 0.03),
    (1.50, -20.0, 1.05, 1, 0.05, -0.02),
    (0.55, 20.0, 1.10, 2, -0.05, 0.04),
    (1.20, 8.0, 0.86, 1, 0.02, 0.05),
    (0.85, -16.0, 1.28, 2, -0.02, -0.03),
    (1.40, 6.0, 1.00, 1, -0.04, 0.01),
    (0.65, -8.0, 1.18, 2, 0.04, -0.04),
    (1.15, 24.0, 0.95, 1, -0.03, 0.02),
]

#: Reject a candidate registration photo that is this similar (or more) to one
#: already selected for the same student. Kept below the default
#: ``DUPLICATE_FACE_SIMILARITY`` (0.98) so the API accepts every demo photo.
MAX_REGISTRATION_SIMILARITY = 0.94

EVAL_VARIANTS = [
    (1.20, 7.0, 1.08, 0, 0.03, 0.00),
    (0.75, -8.0, 0.94, 0, -0.02, 0.02),
    (1.00, 0.0, 1.15, 1, 0.00, 0.00),
    (1.35, 12.0, 1.00, 0, 0.02, 0.03),
    (0.62, -12.0, 1.10, 2, -0.03, 0.00),
    (1.00, 18.0, 0.90, 0, 0.00, 0.00),
]

TOM_HANKS = "insightface/data/images/Tom_Hanks_54745.png"


def transform(
    image: np.ndarray,
    brightness: float = 1.0,
    rotation: float = 0.0,
    scale: float = 1.0,
    blur: int = 0,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    """Cheap simulated capture variation (lighting, angle, distance, blur)."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height
    out = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    if brightness != 1.0:
        out = np.clip(out.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    if blur:
        k = blur * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def crop_face(image: np.ndarray, bbox, margin: float = 0.40) -> np.ndarray:
    """Crop around a bounding box with a margin, clamped to the image."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (0.5 + margin)
    nx1 = int(max(0, centre_x - half))
    ny1 = int(max(0, centre_y - half))
    nx2 = int(min(image.shape[1], centre_x + half))
    ny2 = int(min(image.shape[0], centre_y + half))
    return image[ny1:ny2, nx1:nx2].copy()


def select_registration_photos(crop: np.ndarray, wanted: int = 3) -> list[np.ndarray]:
    """Pick ``wanted`` registration photos, preferring non-duplicate ones.

    Photos derived from a single source image are usually 0.95+ similar, which
    the API's duplicate guard would reject. Candidates are therefore scored once
    and then selected with the strictest bound first; the bound is relaxed only
    if the pool cannot satisfy ``wanted``. Every bound tried stays below the
    ``DUPLICATE_FACE_SIMILARITY`` default (0.98), so the API accepts every photo.
    """
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for params in CANDIDATE_VARIANTS:
        photo = transform(crop, *params)
        faces = face_service.detect(photo)
        if faces:
            candidates.append((photo, faces[0].embedding))

    chosen: list[tuple[np.ndarray, np.ndarray]] = []
    for bound in (MAX_REGISTRATION_SIMILARITY, 0.96, 0.975):
        selected: list[tuple[np.ndarray, np.ndarray]] = []
        for photo, vector in candidates:
            if all(
                cosine_similarity(vector, other) < bound for _, other in selected
            ):
                selected.append((photo, vector))
                if len(selected) >= wanted:
                    break
        if len(selected) >= wanted:
            if bound > MAX_REGISTRATION_SIMILARITY:
                print(
                    f"    (relaxed duplicate bound to {bound:.2f} to reach "
                    f"{wanted} photos)"
                )
            return [photo for photo, _ in selected]
        chosen = selected
    print(f"    WARNING: only {len(chosen)}/{wanted} distinct photos available")
    return [photo for photo, _ in chosen]


def load_unknown_person(canvas_size: int = 512, scale: float = 2.0) -> np.ndarray | None:
    """Load the bundled portrait sample as a *detectable* image.

    ``Tom_Hanks_54745.png`` is an already-aligned 112x112 face chip with no
    surrounding context, so a detector cannot localise a face in it. Placing it
    on a larger canvas restores the context a detector expects.
    """
    import insightface

    candidate = (
        Path(insightface.__file__).resolve().parent
        / "data/images/Tom_Hanks_54745.png"
    )
    if not candidate.is_file():
        return None
    chip = cv2.imread(str(candidate))
    if chip is None:
        return None

    enlarged = cv2.resize(chip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    height, width = enlarged.shape[:2]
    if height > canvas_size or width > canvas_size:
        return enlarged

    canvas = np.full((canvas_size, canvas_size, 3), 190, np.uint8)
    top = (canvas_size - height) // 2
    left = (canvas_size - width) // 2
    canvas[top : top + height, left : left + width] = enlarged
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)

    configure_logging()
    face_service.load()

    from insightface.data import get_image

    group = get_image("t1")
    unknown_image = load_unknown_person()
    if unknown_image is None:
        print("! bundled portrait sample not found - skipping the 'unknown person' data")

    faces = face_service.detect(group)
    faces.sort(key=lambda f: f.bbox[0])  # left-to-right, stable naming
    print(f"Detected {len(faces)} faces in the bundled group photo")

    for index, (roll, name) in enumerate(IDENTITIES):
        if index >= len(faces):
            print(f"! no face available for {roll}_{name}, skipping")
            continue
        crop = crop_face(group, faces[index].bbox)

        registration = select_registration_photos(crop, wanted=3)
        for n, photo in enumerate(registration, 1):
            target = TEST_DATA / "students" / f"{roll}_{name}" / f"{n:02d}.jpg"
            if args.force or not target.exists():
                write_image_file(target, photo)

        for n, params in enumerate(EVAL_VARIANTS, 1):
            target = TEST_DATA / "eval" / f"{roll}_{name}" / f"{n:02d}.jpg"
            if args.force or not target.exists():
                write_image_file(target, transform(crop, *params))

        print(
            f"  {roll}_{name}: {len(registration)} registration + "
            f"{len(EVAL_VARIANTS)} eval images"
        )

    if unknown_image is not None:
        unknown_faces = face_service.detect(unknown_image)
        if unknown_faces:
            unknown_crop = crop_face(unknown_image, unknown_faces[0].bbox)
            for n, params in enumerate(EVAL_VARIANTS[:4], 1):
                target = TEST_DATA / "eval" / "unknown" / f"{n:02d}.jpg"
                if args.force or not target.exists():
                    write_image_file(target, transform(unknown_crop, *params))
            print(f"  unknown: {len(EVAL_VARIANTS[:4])} eval images")

    write_image_file(TEST_DATA / "attendance" / "classroom_01.jpg", group)
    write_image_file(
        TEST_DATA / "attendance" / "classroom_02.jpg",
        transform(group, brightness=1.25, rotation=3.0, scale=1.02),
    )
    if unknown_image is not None:
        write_image_file(
            TEST_DATA / "attendance" / "classroom_unknown.jpg",
            transform(unknown_image, brightness=1.1),
        )
    print("  attendance: classroom_01.jpg, classroom_02.jpg, classroom_unknown.jpg")

    print(f"\nDemo dataset ready at {TEST_DATA}")
    print("NOTE: eval images are variations of the registration photos, so the")
    print("      accuracy numbers are optimistic. Replace with real photos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
