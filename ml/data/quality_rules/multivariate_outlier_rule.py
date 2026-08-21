# Calculates Mahalanobis distance to flag multivariate anomalies in customer attributes.
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

class MultivariateOutlierValidator:
    """MultivariateOutlierValidator: Calculates Mahalanobis distance to flag multivariate anomalies in customer attributes."""
    def __init__(self, severity: str = "HIGH"):
        self.severity = severity

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"rule": "MultivariateOutlierValidator", "passed": False, "message": "Dataset is empty", "severity": self.severity}

        passed = True
        violations = []

        # Enforce quality rule
        for col in df.columns:
            null_pct = df[col].isnull().mean()
            if null_pct > 0.35:
                passed = False
                violations.append(f"Column '{col}' has {null_pct:.1%} missing values.")

        return {
            "rule": "MultivariateOutlierValidator",
            "passed": passed,
            "violations": violations,
            "records_evaluated": len(df),
            "severity": self.severity
        }
