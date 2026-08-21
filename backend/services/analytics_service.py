# Analytics service.
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from packages.schemas.analytics import (
    AnalyticsDashboardResponse, BusinessKPIs, ChurnBySegment,
    ChurnTrendPoint, RevenueAtRiskSummary, CohortRetentionMatrix
)

class AnalyticsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_dashboard_analytics(self) -> AnalyticsDashboardResponse:
        kpis = BusinessKPIs(
            total_customers=12540,
            active_customers=11820,
            overall_churn_rate_pct=14.2,
            high_risk_customers_count=1780,
            estimated_revenue_at_risk_monthly=142300.0,
            estimated_revenue_at_risk_annual=1707600.0,
            average_customer_lifetime_value=1240.0,
            net_retention_rate_pct=108.5
        )

        trends = [
            ChurnTrendPoint(date="2026-03", total_customers=10200, predicted_churners=1450, actual_churners=1410, churn_rate_pct=13.8),
            ChurnTrendPoint(date="2026-04", total_customers=10800, predicted_churners=1520, actual_churners=1490, churn_rate_pct=13.8),
            ChurnTrendPoint(date="2026-05", total_customers=11300, predicted_churners=1630, actual_churners=1610, churn_rate_pct=14.2),
            ChurnTrendPoint(date="2026-06", total_customers=11900, predicted_churners=1710, actual_churners=1690, churn_rate_pct=14.2),
            ChurnTrendPoint(date="2026-07", total_customers=12540, predicted_churners=1780, actual_churners=None, churn_rate_pct=14.2)
        ]

        sub_segments = [
            ChurnBySegment(segment_name="Basic", customer_count=4380, churn_rate_pct=22.4, revenue_at_risk=28500.0),
            ChurnBySegment(segment_name="Standard", customer_count=5020, churn_rate_pct=13.1, revenue_at_risk=52100.0),
            ChurnBySegment(segment_name="Premium", customer_count=2260, churn_rate_pct=8.5, revenue_at_risk=28600.0),
            ChurnBySegment(segment_name="Enterprise", customer_count=880, churn_rate_pct=4.2, revenue_at_risk=33100.0)
        ]

        contract_segments = [
            ChurnBySegment(segment_name="Month-to-Month", customer_count=6890, churn_rate_pct=23.8, revenue_at_risk=98400.0),
            ChurnBySegment(segment_name="One-Year", customer_count=3760, churn_rate_pct=6.5, revenue_at_risk=31200.0),
            ChurnBySegment(segment_name="Two-Year", customer_count=1890, churn_rate_pct=2.1, revenue_at_risk=12700.0)
        ]

        return AnalyticsDashboardResponse(
            kpis=kpis,
            churn_trends=trends,
            revenue_risk=RevenueAtRiskSummary(
                by_subscription_tier=sub_segments,
                by_contract_type=contract_segments,
                by_region=[],
                by_tenure_cohort=[]
            ),
            cohort_retention=CohortRetentionMatrix(
                cohort_months=["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"],
                matrix=[
                    [100.0, 94.2, 89.1, 85.4, 82.1],
                    [100.0, 95.1, 90.3, 86.8, 0.0],
                    [100.0, 93.8, 88.5, 0.0, 0.0],
                    [100.0, 94.7, 0.0, 0.0, 0.0],
                    [100.0, 0.0, 0.0, 0.0, 0.0]
                ]
            ),
            generated_at=datetime.now(timezone.utc)
        )
