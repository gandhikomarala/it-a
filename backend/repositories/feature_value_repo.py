# FeatureValueRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.feature import FeatureValue
from .base import BaseRepository

class FeatureValueRepository(BaseRepository[FeatureValue]):
    def __init__(self, session: AsyncSession):
        super().__init__(FeatureValue, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[FeatureValue]:
        stmt = select(FeatureValue).where(FeatureValue.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(FeatureValue.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
