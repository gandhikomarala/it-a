# Customer service.
from typing import Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from backend.repositories.customer_repo import CustomerRepository
from packages.schemas.customer import (
    CustomerCreate, CustomerUpdate, CustomerResponse, CustomerListResponse, CustomerSegmentationSummary
)
from backend.core.exceptions import NotFoundException, ConflictException

class CustomerService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = CustomerRepository(db)

    async def get_customer(self, id: str) -> CustomerResponse:
        c = await self.repo.get_by_id(id)
        if not c:
            raise NotFoundException(f"Customer with id '{id}' not found.")
        return CustomerResponse.model_validate(c)

    async def list_customers(
        self,
        search: Optional[str] = None,
        risk_level: Optional[str] = None,
        subscription_type: Optional[str] = None,
        contract_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> CustomerListResponse:
        skip = (page - 1) * page_size
        items, total = await self.repo.search_customers(
            search=search,
            risk_level=risk_level,
            subscription_type=subscription_type,
            contract_type=contract_type,
            skip=skip,
            limit=page_size
        )
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        return CustomerListResponse(
            items=[CustomerResponse.model_validate(x) for x in items],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    async def get_segmentation_summary(self) -> CustomerSegmentationSummary:
        items, total = await self.repo.get_all(limit=10000)
        high_risk = sum(1 for x in items if x.latest_risk_level in ["HIGH", "CRITICAL"])
        med_risk = sum(1 for x in items if x.latest_risk_level == "MEDIUM")
        low_risk = sum(1 for x in items if x.latest_risk_level == "LOW")
        rev_at_risk = sum(x.monthly_charge for x in items if x.latest_risk_level in ["HIGH", "CRITICAL"])

        return CustomerSegmentationSummary(
            total_customers=total,
            active_customers=sum(1 for x in items if x.is_active),
            high_risk_count=high_risk,
            medium_risk_count=med_risk,
            low_risk_count=low_risk,
            estimated_revenue_at_risk=round(rev_at_risk, 2),
            average_customer_tenure=round(sum(x.tenure_months for x in items) / total, 1) if total > 0 else 0.0,
            average_monthly_revenue=round(sum(x.monthly_charge for x in items) / total, 2) if total > 0 else 0.0
        )
