# Customer management and segmentation router.
from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.session import get_db
from backend.services.customer_service import CustomerService
from packages.schemas.customer import CustomerResponse, CustomerListResponse, CustomerSegmentationSummary
from backend.core.dependencies import get_current_user, require_permissions

router = APIRouter(prefix="/customers", tags=["Customers"])

@router.get("", response_model=CustomerListResponse)
async def list_customers(
    search: Optional[str] = None,
    risk_level: Optional[str] = None,
    subscription_type: Optional[str] = None,
    contract_type: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_permissions(["customer:read"]))
):
    service = CustomerService(db)
    return await service.list_customers(
        search=search,
        risk_level=risk_level,
        subscription_type=subscription_type,
        contract_type=contract_type,
        page=page,
        page_size=page_size
    )

@router.get("/summary/segmentation", response_model=CustomerSegmentationSummary)
async def get_customer_summary(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_permissions(["customer:read"]))
):
    service = CustomerService(db)
    return await service.get_segmentation_summary()

@router.get("/{id}", response_model=CustomerResponse)
async def get_customer_detail(
    id: str,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_permissions(["customer:read"]))
):
    service = CustomerService(db)
    return await service.get_customer(id)
