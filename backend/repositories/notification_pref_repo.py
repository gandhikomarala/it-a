# NotificationPreferenceRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.notification import NotificationPreference
from .base import BaseRepository

class NotificationPreferenceRepository(BaseRepository[NotificationPreference]):
    def __init__(self, session: AsyncSession):
        super().__init__(NotificationPreference, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[NotificationPreference]:
        stmt = select(NotificationPreference).where(NotificationPreference.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(NotificationPreference.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
