# ModelMetricRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.model import ModelMetric
from .base import BaseRepository

class ModelMetricRepository(BaseRepository[ModelMetric]):
    def __init__(self, session: AsyncSession):
        super().__init__(ModelMetric, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[ModelMetric]:
        stmt = select(ModelMetric).where(ModelMetric.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(ModelMetric.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
