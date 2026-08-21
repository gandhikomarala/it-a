# SkillGapDiagnosticDeltaExtractor (EdTech & Corporate Learning SaaS)
# Pre-test vs post-test diagnostic benchmark score improvement.
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SkillGapDiagnosticDeltaExtractor(BaseEstimator, TransformerMixin):
    """SkillGapDiagnosticDeltaExtractor: Pre-test vs post-test diagnostic benchmark score improvement."""
    def __init__(self, prefix: str = "skill_gap_diagnostic_delta"):
        self.prefix = prefix
        self.params_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = pd.DataFrame(X).copy()
        if "monthly_charge" in X.columns:
            self.params_["charge_mean"] = float(X["monthly_charge"].mean())
        if "tenure_months" in X.columns:
            self.params_["tenure_median"] = float(X["tenure_months"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        n = len(X_out)
        
        if "tenure_months" in X_out.columns and "monthly_charge" in X_out.columns:
            tenure = np.maximum(1.0, X_out["tenure_months"].values)
            charge = np.maximum(0.0, X_out["monthly_charge"].values)
            
            # Mathematical domain indicator
            decay = np.exp(-0.04 * tenure)
            norm_charge = charge / (self.params_.get("charge_mean", 80.0) + 1e-5)
            
            X_out[f"{self.prefix}_signal"] = np.log1p(charge) * (1.0 / (np.sqrt(tenure) + 1.0))
            X_out[f"{self.prefix}_decay"] = decay
            X_out[f"{self.prefix}_risk_score"] = np.clip(decay * norm_charge * 1.8, 0.0, 1.0)
        else:
            X_out[f"{self.prefix}_signal"] = np.zeros(n)
            X_out[f"{self.prefix}_risk_score"] = np.ones(n) * 0.5
            
        return X_out
