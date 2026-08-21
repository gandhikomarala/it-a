# Model registry repository.
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.model import MLModel, ModelVersion, ModelDeployment
from .base import BaseRepository

class ModelRepository(BaseRepository[MLModel]):
    def __init__(self, session: AsyncSession):
        super().__init__(MLModel, session)

    async def get_active_production_version(self, model_id: str) -> Optional[ModelVersion]:
        stmt = select(ModelVersion).where(
            ModelVersion.model_id == model_id,
            ModelVersion.stage == "PRODUCTION",
            ModelVersion.is_active_production.is_(True)
        )
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_version_by_id(self, version_id: str) -> Optional[ModelVersion]:
        stmt = select(ModelVersion).where(ModelVersion.id == version_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()
