# ModelArtifactRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.model import ModelArtifact
from .base import BaseRepository

class ModelArtifactRepository(BaseRepository[ModelArtifact]):
    def __init__(self, session: AsyncSession):
        super().__init__(ModelArtifact, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[ModelArtifact]:
        stmt = select(ModelArtifact).where(ModelArtifact.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(ModelArtifact.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
