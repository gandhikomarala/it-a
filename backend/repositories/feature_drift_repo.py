# FeatureDriftMetricRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.monitoring import FeatureDriftMetricRecord
from .base import BaseRepository

class FeatureDriftMetricRepository(BaseRepository[FeatureDriftMetricRecord]):
    def __init__(self, session: AsyncSession):
        super().__init__(FeatureDriftMetricRecord, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[FeatureDriftMetricRecord]:
        stmt = select(FeatureDriftMetricRecord).where(FeatureDriftMetricRecord.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(FeatureDriftMetricRecord.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
