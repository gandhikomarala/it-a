# Customer repository.
from typing import List, Optional, Tuple
from sqlalchemy import func, select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.customer import Customer, CustomerEvent, CustomerNote
from .base import BaseRepository

class CustomerRepository(BaseRepository[Customer]):
    def __init__(self, session: AsyncSession):
        super().__init__(Customer, session)

    async def get_by_customer_id(self, customer_id: str) -> Optional[Customer]:
        stmt = select(Customer).where(Customer.customer_id == customer_id, Customer.is_deleted.is_(False))
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def search_customers(
        self,
        search: Optional[str] = None,
        risk_level: Optional[str] = None,
        subscription_type: Optional[str] = None,
        contract_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[List[Customer], int]:
        stmt = select(Customer).where(Customer.is_deleted.is_(False))

        if search:
            s = f"%{search}%"
            stmt = stmt.where(or_(
                Customer.customer_id.ilike(s),
                Customer.first_name.ilike(s),
                Customer.last_name.ilike(s),
                Customer.email.ilike(s)
            ))

        if risk_level:
            stmt = stmt.where(Customer.latest_risk_level == risk_level)
        if subscription_type:
            stmt = stmt.where(Customer.subscription_type == subscription_type)
        if contract_type:
            stmt = stmt.where(Customer.contract_type == contract_type)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_res = await self.session.execute(count_stmt)
        total = total_res.scalar() or 0

        stmt = stmt.order_by(Customer.created_at.desc()).offset(skip).limit(limit)
        res = await self.session.execute(stmt)
        return list(res.scalars().all()), total
