# NotificationTemplateRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.notification import NotificationTemplate
from .base import BaseRepository

class NotificationTemplateRepository(BaseRepository[NotificationTemplate]):
    def __init__(self, session: AsyncSession):
        super().__init__(NotificationTemplate, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[NotificationTemplate]:
        stmt = select(NotificationTemplate).where(NotificationTemplate.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(NotificationTemplate.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
