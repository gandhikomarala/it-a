# PerformanceSnapshotRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.monitoring import ModelPerformanceSnapshot
from .base import BaseRepository

class PerformanceSnapshotRepository(BaseRepository[ModelPerformanceSnapshot]):
    def __init__(self, session: AsyncSession):
        super().__init__(ModelPerformanceSnapshot, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[ModelPerformanceSnapshot]:
        stmt = select(ModelPerformanceSnapshot).where(ModelPerformanceSnapshot.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(ModelPerformanceSnapshot.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
