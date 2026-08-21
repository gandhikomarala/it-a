# QualityReportRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.dataset import DataQualityReport
from .base import BaseRepository

class QualityReportRepository(BaseRepository[DataQualityReport]):
    def __init__(self, session: AsyncSession):
        super().__init__(DataQualityReport, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[DataQualityReport]:
        stmt = select(DataQualityReport).where(DataQualityReport.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(DataQualityReport.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
