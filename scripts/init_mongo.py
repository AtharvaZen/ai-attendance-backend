#!/usr/bin/env python
"""Create the MongoDB collections and indexes (idempotent, never destructive).

Usage:
    python scripts/init_mongo.py

Requires MONGODB_URI in ``backend/.env``. Prints the indexes it created so you
can confirm the unique constraints (school + roll number, school + student +
class + section + date) are actually in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import configure_logging, settings  # noqa: E402
from app.database.mongo_connection import (  # noqa: E402
    ATTENDANCE,
    FACE_EMBEDDINGS,
    SCHOOLS,
    STUDENTS,
    check_database,
    close_client,
    collection,
    init_db,
    mask_uri,
    vector_search_supported,
)


def main() -> int:
    configure_logging()

    if not settings.uses_mongodb:
        print("MONGODB_URI is not set - nothing to initialise.")
        print("The API will use the in-memory repository (development only).")
        print("\nTo use MongoDB:")
        print("  1. set MONGODB_URI in backend/.env (Atlas or a local server)")
        print("  2. python scripts/init_mongo.py")
        return 1

    print(f"Connecting to {mask_uri(settings.mongodb_uri)} ...")
    reachable, info = check_database()
    if not reachable:
        print(f"FAILED: {info}")
        return 1
    print(f"Connected: {info}")

    init_db()

    print("\nCollections:")
    for name in (SCHOOLS, STUDENTS, FACE_EMBEDDINGS, ATTENDANCE):
        print(f"  - {name}")
    print("\nIndexes on students:")
    for index in collection(STUDENTS).list_indexes():
        print(f"  - {index.get('name')}: {index.get('key')}")
    print("\nIndexes on face_embeddings:")
    for index in collection(FACE_EMBEDDINGS).list_indexes():
        print(f"  - {index.get('name')}: {index.get('key')}")
    print("\nIndexes on attendance:")
    for index in collection(ATTENDANCE).list_indexes():
        print(f"  - {index.get('name')}: {index.get('key')}")

    print("\nVector search: $vectorSearch" if vector_search_supported()
          else "\nVector search: unavailable on this deployment - using exact "
               "cosine similarity (same results)")
    close_client()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())