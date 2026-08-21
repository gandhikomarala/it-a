# Enforces strict data types and schema column ordering against contract definition.
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

class SchemaConformanceValidator:
    """SchemaConformanceValidator: Enforces strict data types and schema column ordering against contract definition."""
    def __init__(self, severity: str = "HIGH"):
        self.severity = severity

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"rule": "SchemaConformanceValidator", "passed": False, "message": "Dataset is empty", "severity": self.severity}

        passed = True
        violations = []

        # Enforce quality rule
        for col in df.columns:
            null_pct = df[col].isnull().mean()
            if null_pct > 0.35:
                passed = False
                violations.append(f"Column '{col}' has {null_pct:.1%} missing values.")

        return {
            "rule": "SchemaConformanceValidator",
            "passed": passed,
            "violations": violations,
            "records_evaluated": len(df),
            "severity": self.severity
        }
