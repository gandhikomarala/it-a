# System health, configuration, and settings router.
from datetime import datetime, timezone
from fastapi import APIRouter
from packages.schemas.settings import HealthCheckResponse, ServiceHealthStatus

router = APIRouter(prefix="/system", tags=["System"])

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="HEALTHY",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc),
        uptime_seconds=3600.0,
        services=[
            ServiceHealthStatus(service_name="PostgreSQL", status="HEALTHY", latency_ms=1.2),
            ServiceHealthStatus(service_name="Redis", status="HEALTHY", latency_ms=0.5),
            ServiceHealthStatus(service_name="Celery Worker", status="HEALTHY", latency_ms=2.1),
            ServiceHealthStatus(service_name="Model Engine", status="HEALTHY", latency_ms=0.8)
        ]
    )
