# SupportTicketRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.customer import CustomerSupport
from .base import BaseRepository

class SupportTicketRepository(BaseRepository[CustomerSupport]):
    def __init__(self, session: AsyncSession):
        super().__init__(CustomerSupport, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[CustomerSupport]:
        stmt = select(CustomerSupport).where(CustomerSupport.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(CustomerSupport.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
