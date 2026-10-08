#!/usr/bin/env python
"""Phase 1 / Phase 3 smoke test - verify that face detection works.

It loads the pretrained InsightFace model (once), detects every face in an
image and prints each bounding box, detector score and embedding norm.

Usage
-----
    python scripts/detect_faces.py                        # bundled sample image
    python scripts/detect_faces.py path/to/photo.jpg
    python scripts/detect_faces.py photo.jpg --save out/annotated.jpg
    python scripts/detect_faces.py photo.jpg --min-face-size 40
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Make ``app`` importable when the script is run directly.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import configure_logging, settings  # noqa: E402
from app.services.face_service import (  # noqa: E402
    FaceModelError,
    face_service,
    read_image_file,
    write_image_file,
)

logger = logging.getLogger("detect_faces")


def _load_bundled_sample() -> np.ndarray:
    """Fall back to the sample image that ships with InsightFace."""
    from insightface.data import get_image

    logger.info("No image supplied - using the bundled InsightFace sample image")
    return get_image("t1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", nargs="?", help="path to an image file")
    parser.add_argument("--save", help="write an annotated copy to this path")
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=None,
        help=f"reject faces smaller than N px (default {settings.face_min_face_size})",
    )
    args = parser.parse_args(argv)

    configure_logging()

    # 1. Load the pretrained model.
    logger.info("Loading pretrained face model '%s' ...", settings.face_model_name)
    started = time.perf_counter()
    try:
        face_service.load()
    except FaceModelError as exc:
        logger.critical("%s", exc)
        return 1
    logger.info(
        "Model ready in %.0f ms | providers=%s",
        (time.perf_counter() - started) * 1000,
        face_service.providers,
    )

    # 2. Read the image.
    source = args.image or "insightface sample (t1)"
    try:
        image = _load_bundled_sample() if not args.image else read_image_file(args.image)
    except Exception as exc:
        logger.error("Could not read image '%s': %s", source, exc)
        return 1

    # 3. Detect faces (+ embeddings).
    started = time.perf_counter()
    faces = face_service.detect(image, min_face_size=args.min_face_size)
    elapsed_ms = (time.perf_counter() - started) * 1000

    # 4. Report.
    print()
    print(f"Image          : {source}")
    print(f"Resolution     : {image.shape[1]}x{image.shape[0]}")
    print(f"Detected faces : {len(faces)}")
    print(f"Detection time : {elapsed_ms:.1f} ms")
    print()
    for index, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = face.bbox
        norm = float(np.linalg.norm(face.embedding)) if face.embedding is not None else float("nan")
        shape = face.embedding.shape if face.embedding is not None else None
        print(f"  Face #{index}")
        print(f"    bbox      : ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})  ({face.min_side:.0f}px)")
        print(f"    det score : {face.det_score:.3f}")
        print(f"    embedding : shape={shape}  L2-norm={norm:.4f}")
        print()
    if not faces:
        print("  No usable faces found.\n")

    if args.save:
        try:
            write_image_file(args.save, face_service.annotate(image, faces))
            print(f"Annotated image written to: {args.save}\n")
        except Exception as exc:
            logger.error("Could not save annotated image: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
