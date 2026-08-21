# Confusion matrix analysis and business revenue impact calculations.
from typing import Dict, Any

class ConfusionMatrixAnalyzer:
    @staticmethod
    def calculate_business_impact(
        tn: int, fp: int, fn: int, tp: int,
        retention_intervention_cost: float = 25.0,
        average_customer_clv: float = 650.0,
        intervention_success_rate: float = 0.40
    ) -> Dict[str, Any]:
        customers_targeted = tp + fp
        intervention_costs = customers_targeted * retention_intervention_cost
        customers_saved = int(tp * intervention_success_rate)
        revenue_saved = customers_saved * average_customer_clv
        net_profit_generated = revenue_saved - intervention_costs
        roi_percentage = (net_profit_generated / intervention_costs * 100.0) if intervention_costs > 0 else 0.0

        return {
            "customers_targeted": customers_targeted,
            "intervention_cost": intervention_costs,
            "estimated_customers_saved": customers_saved,
            "revenue_saved": revenue_saved,
            "net_financial_benefit": net_profit_generated,
            "campaign_roi_percentage": round(roi_percentage, 1)
        }
