# Catalog of domain features for customer churn modeling.
from typing import Dict, List, Optional
from packages.schemas.feature import FeatureDefinitionSchema

FEATURE_DEFINITIONS: List[FeatureDefinitionSchema] = [
    FeatureDefinitionSchema(
        name="tenure_months",
        data_type="integer",
        description="Number of months customer has been subscribed to service.",
        transformation_logic="Direct observation from customer subscription history.",
        category="Demographics & Tenure"
    ),
    FeatureDefinitionSchema(
        name="monthly_charge",
        data_type="float",
        description="Current recurring monthly billing charge in USD.",
        transformation_logic="Direct observation.",
        category="Billing & Pricing"
    ),
    FeatureDefinitionSchema(
        name="daily_usage_hours",
        data_type="float",
        description="Average daily active usage in hours over past 30 days.",
        transformation_logic="Mean of daily active session durations.",
        category="Usage & Activity"
    ),
    FeatureDefinitionSchema(
        name="usage_trend",
        data_type="float",
        description="Trend slope of usage over past 90 days (-1.0 to +1.0).",
        transformation_logic="Linear regression slope over rolling 7-day usage windows.",
        category="Usage & Activity"
    ),
    FeatureDefinitionSchema(
        name="days_since_last_login",
        data_type="integer",
        description="Days elapsed since customer last logged into platform.",
        transformation_logic="Days between current timestamp and last_active_at.",
        category="Usage & Activity"
    ),
    FeatureDefinitionSchema(
        name="payment_failures_count",
        data_type="integer",
        description="Total failed payment attempts in past 90 days.",
        transformation_logic="Count of payment records with status=FAILED.",
        category="Billing & Pricing"
    ),
    FeatureDefinitionSchema(
        name="late_payments_count",
        data_type="integer",
        description="Count of invoices paid after due date in past 12 months.",
        transformation_logic="Count of payments settled > 0 days after due date.",
        category="Billing & Pricing"
    ),
    FeatureDefinitionSchema(
        name="complaint_count",
        data_type="integer",
        description="Number of formal complaints logged by customer.",
        transformation_logic="Count of support tickets categorized as COMPLAINT.",
        category="Support & Customer Care"
    ),
    FeatureDefinitionSchema(
        name="satisfaction_score",
        data_type="float",
        description="Customer satisfaction CSAT rating (1.0 - 5.0).",
        transformation_logic="Mean of post-ticket survey responses or NPS translation.",
        category="Support & Customer Care"
    ),
    FeatureDefinitionSchema(
        name="charge_to_income_ratio",
        data_type="float",
        description="Financial burden ratio: monthly charge / (annual income / 12).",
        transformation_logic="monthly_charge / (income / 12).",
        category="Derived Financial"
    ),
    FeatureDefinitionSchema(
        name="spend_per_tenure_month",
        data_type="float",
        description="Customer spend velocity per month of subscription.",
        transformation_logic="total_spend / (tenure_months + 1).",
        category="Derived Financial"
    ),
    FeatureDefinitionSchema(
        name="total_payment_issues",
        data_type="integer",
        description="Weighted composite score of payment failures and late payments.",
        transformation_logic="(payment_failures * 2) + late_payments.",
        category="Billing & Pricing"
    )
]

def get_feature_definition(name: str) -> Optional[FeatureDefinitionSchema]:
    for feat in FEATURE_DEFINITIONS:
        if feat.name == name:
            return feat
    return None
