# DatasetVersionRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.dataset import DatasetVersion
from .base import BaseRepository

class DatasetVersionRepository(BaseRepository[DatasetVersion]):
    def __init__(self, session: AsyncSession):
        super().__init__(DatasetVersion, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[DatasetVersion]:
        stmt = select(DatasetVersion).where(DatasetVersion.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(DatasetVersion.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
