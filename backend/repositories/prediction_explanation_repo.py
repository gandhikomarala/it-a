# PredictionExplanationRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.prediction import PredictionExplanation
from .base import BaseRepository

class PredictionExplanationRepository(BaseRepository[PredictionExplanation]):
    def __init__(self, session: AsyncSession):
        super().__init__(PredictionExplanation, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[PredictionExplanation]:
        stmt = select(PredictionExplanation).where(PredictionExplanation.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(PredictionExplanation.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
