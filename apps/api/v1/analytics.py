# Executive Analytics and BI router.
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.session import get_db
from backend.services.analytics_service import AnalyticsService
from packages.schemas.analytics import AnalyticsDashboardResponse
from backend.core.dependencies import require_permissions

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/dashboard", response_model=AnalyticsDashboardResponse)
async def get_dashboard_analytics(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_permissions(["analytics:read"]))
):
    service = AnalyticsService(db)
    return await service.get_dashboard_analytics()
