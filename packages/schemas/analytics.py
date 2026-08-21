"""Business intelligence and MLOps executive analytics schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class BusinessKPIs(BaseModel):
    total_customers: int
    active_customers: int
    overall_churn_rate_pct: float
    high_risk_customers_count: int
    estimated_revenue_at_risk_monthly: float
    estimated_revenue_at_risk_annual: float
    average_customer_lifetime_value: float
    net_retention_rate_pct: float

class ChurnBySegment(BaseModel):
    segment_name: str
    customer_count: int
    churn_rate_pct: float
    revenue_at_risk: float

class ChurnTrendPoint(BaseModel):
    date: str
    total_customers: int
    predicted_churners: int
    actual_churners: Optional[int] = None
    churn_rate_pct: float

class RevenueAtRiskSummary(BaseModel):
    by_subscription_tier: List[ChurnBySegment]
    by_contract_type: List[ChurnBySegment]
    by_region: List[ChurnBySegment]
    by_tenure_cohort: List[ChurnBySegment]

class CohortRetentionMatrix(BaseModel):
    cohort_months: List[str]
    matrix: List[List[float]]

class AnalyticsDashboardResponse(BaseModel):
    kpis: BusinessKPIs
    churn_trends: List[ChurnTrendPoint]
    revenue_risk: RevenueAtRiskSummary
    cohort_retention: CohortRetentionMatrix
    generated_at: datetime
