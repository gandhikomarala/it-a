# PredictionBatchRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.prediction import PredictionBatch
from .base import BaseRepository

class PredictionBatchRepository(BaseRepository[PredictionBatch]):
    def __init__(self, session: AsyncSession):
        super().__init__(PredictionBatch, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[PredictionBatch]:
        stmt = select(PredictionBatch).where(PredictionBatch.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(PredictionBatch.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
