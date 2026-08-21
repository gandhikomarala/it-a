# Dataset ingestion, profiling, and quality validation router.
from typing import Optional
from fastapi import APIRouter, Depends, Query, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.session import get_db
from backend.services.dataset_service import DatasetService
from packages.schemas.dataset import DatasetResponse, DatasetListResponse, DatasetValidationResult
from backend.core.dependencies import require_permissions

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_permissions(["dataset:read"]))
):
    service = DatasetService(db)
    return await service.list_datasets(page=page, page_size=page_size)
