# Categorizes churn probabilities into business risk levels.
from packages.shared.enums import RiskLevel
from packages.shared.constants import DEFAULT_RISK_THRESHOLDS

class RiskCategorizer:
    @staticmethod
    def categorize(
        probability: float,
        low_cutoff: float = DEFAULT_RISK_THRESHOLDS["LOW_CUTOFF"],
        high_cutoff: float = DEFAULT_RISK_THRESHOLDS["HIGH_CUTOFF"],
        critical_cutoff: float = DEFAULT_RISK_THRESHOLDS["CRITICAL_CUTOFF"]
    ) -> RiskLevel:
        if probability >= critical_cutoff:
            return RiskLevel.CRITICAL
        elif probability >= high_cutoff:
            return RiskLevel.HIGH
        elif probability >= low_cutoff:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
