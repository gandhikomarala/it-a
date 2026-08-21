# Model registry, deployments, and promotion router.
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.session import get_db
from backend.services.model_service import ModelService
from packages.schemas.model import ModelResponse
from backend.core.dependencies import require_permissions

router = APIRouter(prefix="/models", tags=["Models"])

@router.get("", response_model=List[ModelResponse])
async def list_models(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_permissions(["model:read"]))
):
    service = ModelService(db)
    return await service.list_models()
