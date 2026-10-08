"""Health / readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.core.config import settings
from app.database.mongo_connection import check_database
from app.database.repository import AttendanceRepository, get_repository
from app.schemas.common import HealthResponse
from app.services.face_service import face_service

router = APIRouter(tags=["health"])


@router.get(
    "/",
    summary="Service banner",
    description="Basic service identification. Safe to call without authentication.",
    responses={200: {"description": "Service metadata."}},
)
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
        "docs": "/docs",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness + model readiness",
    description=(
        "Reports whether the API is up and whether the pretrained face model "
        "has finished loading. The model is loaded once at application startup."
    ),
    responses={
        200: {"description": "Health report."},
        500: {
            "description": "The face recognition model failed to initialise.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Face recognition model initialization failed",
                        "code": "model_unavailable",
                    }
                }
            },
        },
    },
)
async def health(
    repository: AttendanceRepository = Depends(get_repository),
) -> HealthResponse:
    if repository.backend == "memory":
        connected, info = True, "in-memory store (development / demo only)"
    else:
        connected, info = check_database()

    return HealthResponse(
        status="ok" if face_service.is_loaded else "starting",
        model_loaded=face_service.is_loaded,
        model=settings.face_model_name,
        providers=face_service.providers,
        match_threshold=settings.face_match_threshold,
        match_margin=settings.face_match_margin,
        environment=settings.environment,
        storage_backend=repository.backend,
        database_connected=connected,
        database_info=info,
        students=repository.count_students(),
        face_embeddings=repository.count_embeddings(),
        auth_enabled=settings.auth_enabled,
    )
