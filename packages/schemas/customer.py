"""Customer entities, behavioral metrics, and timeline schemas."""
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field
from packages.shared.enums import ContractType, PaymentMethod, SubscriptionType, RiskLevel

class CustomerBase(BaseModel):
    customer_id: str = Field(..., description="Unique enterprise customer identifier, e.g. CUS-100291")
    first_name: str
    last_name: str
    email: EmailStr
    phone: Optional[str] = None
    age: int = Field(..., ge=18, le=120)
    gender: str = Field(..., pattern="^(Male|Female|Non-Binary|Other)$")
    region: str
    city: str
    occupation: Optional[str] = None
    income: float = Field(..., ge=0.0)
    signup_date: date
    subscription_type: SubscriptionType
    contract_type: ContractType
    payment_method: PaymentMethod
    monthly_charge: float = Field(..., ge=0.0)
    tenure_months: int = Field(..., ge=0)
    total_spend: float = Field(..., ge=0.0)
    is_active: bool = True

class CustomerCreate(CustomerBase):
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class CustomerUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    region: Optional[str] = None
    city: Optional[str] = None
    income: Optional[float] = None
    subscription_type: Optional[SubscriptionType] = None
    contract_type: Optional[ContractType] = None
    payment_method: Optional[PaymentMethod] = None
    monthly_charge: Optional[float] = None
    tenure_months: Optional[int] = None
    total_spend: Optional[float] = None
    is_active: Optional[bool] = None
    tags: Optional[List[str]] = None

class CustomerUsageSchema(BaseModel):
    id: str
    customer_id: str
    daily_usage_hours: float
    weekly_usage_hours: float
    monthly_usage_hours: float
    login_count_monthly: int
    session_count_monthly: int
    average_session_duration_minutes: float
    feature_usage_diversity_score: float
    last_active_at: datetime
    recorded_at: datetime

class CustomerPaymentSchema(BaseModel):
    id: str
    customer_id: str
    payment_amount: float
    payment_status: str
    payment_failures_count: int
    late_payments_count: int
    refunds_count: int
    last_payment_date: datetime

class CustomerSupportTicketSchema(BaseModel):
    id: str
    customer_id: str
    ticket_count: int
    complaint_count: int
    average_resolution_hours: float
    satisfaction_score: float = Field(..., ge=1.0, le=5.0)
    open_tickets_count: int
    last_ticket_date: Optional[datetime] = None

class CustomerSubscriptionSchema(BaseModel):
    id: str
    customer_id: str
    subscription_type: SubscriptionType
    contract_type: ContractType
    start_date: date
    end_date: Optional[date] = None
    auto_renew: bool
    plan_tier_upgrades: int
    plan_tier_downgrades: int
    cancellation_requested: bool

class CustomerTimelineEvent(BaseModel):
    id: str
    customer_id: str
    event_type: str
    title: str
    description: str
    metadata: Dict[str, Any] = {}
    created_at: datetime

class CustomerResponse(CustomerBase):
    id: str
    latest_churn_probability: Optional[float] = None
    latest_risk_level: Optional[RiskLevel] = None
    latest_prediction_date: Optional[datetime] = None
    tags: List[str] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class CustomerFilter(BaseModel):
    search: Optional[str] = None
    risk_level: Optional[RiskLevel] = None
    subscription_type: Optional[SubscriptionType] = None
    contract_type: Optional[ContractType] = None
    region: Optional[str] = None
    is_active: Optional[bool] = None
    min_tenure: Optional[int] = None
    max_tenure: Optional[int] = None
    min_churn_probability: Optional[float] = None
    max_churn_probability: Optional[float] = None
    page: int = 1
    page_size: int = 20
    sort_by: str = "created_at"
    sort_order: str = "desc"

class CustomerListResponse(BaseModel):
    items: List[CustomerResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

class CustomerSegmentationSummary(BaseModel):
    total_customers: int
    active_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    estimated_revenue_at_risk: float
    average_customer_tenure: float
    average_monthly_revenue: float
