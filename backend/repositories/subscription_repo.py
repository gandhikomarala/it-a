# SubscriptionRepository specialized repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.customer import CustomerSubscription
from .base import BaseRepository

class SubscriptionRepository(BaseRepository[CustomerSubscription]):
    def __init__(self, session: AsyncSession):
        super().__init__(CustomerSubscription, session)

    async def get_by_entity_id(self, entity_id: str) -> Optional[CustomerSubscription]:
        stmt = select(CustomerSubscription).where(CustomerSubscription.id == entity_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def count_total(self) -> int:
        stmt = select(func.count(CustomerSubscription.id))
        res = await self.session.execute(stmt)
        return res.scalar() or 0
