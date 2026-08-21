# FeatureVersionRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.feature import FeatureVersion
from .base import BaseRepository

class FeatureVersionRepository(BaseRepository[FeatureVersion]):
    def __init__(self, session: AsyncSession):
        super().__init__(FeatureVersion, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[FeatureVersion]:
        stmt = select(FeatureVersion).where(FeatureVersion.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(FeatureVersion.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
