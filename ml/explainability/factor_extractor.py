# Factor attribution extraction and presentation utilities.
from typing import List
from packages.schemas.prediction import SHAPFactorContribution

class ChurnFactorExtractor:
    @staticmethod
    def get_actionable_recommendations(top_factors: List[SHAPFactorContribution]) -> List[str]:
        recommendations: List[str] = []
        for factor in top_factors:
            feat = factor.feature_name.lower()
            if "payment_failure" in feat:
                recommendations.append("Proactively reach out to update payment method or resolve billing gateway errors.")
            elif "complaint" in feat:
                recommendations.append("Assign senior customer success manager to address unresolved support tickets.")
            elif "month-to-month" in feat:
                recommendations.append("Offer annual subscription incentive with 15% promotional discount.")
            elif "days_since_last_login" in feat or "usage" in feat:
                recommendations.append("Trigger re-engagement onboarding campaign with product tutorials.")
        return recommendations or ["Schedule standard customer health check-in."]
