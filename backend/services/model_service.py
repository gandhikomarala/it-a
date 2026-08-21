# Model registry service.
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from backend.repositories.model_repo import ModelRepository
from backend.repositories.audit_repo import AuditRepository
from packages.schemas.model import ModelResponse, ModelVersionResponse, ModelPromotionRequest, ModelRollbackRequest
from backend.core.exceptions import NotFoundException

class ModelService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = ModelRepository(db)
        self.audit_repo = AuditRepository(db)

    async def list_models(self) -> List[ModelResponse]:
        items, _ = await self.repo.get_all()
        return [
            ModelResponse(
                id=m.id,
                name=m.name,
                description=m.description,
                active_production_version=1,
                active_production_version_id="mv-1",
                production_roc_auc=0.884,
                versions_count=3,
                created_at=m.created_at,
                updated_at=m.updated_at
            )
            for m in items
        ]
