# CommittedSpendBurndownVelocityExtractor (B2B Cloud Marketplace SaaS)
# Speed of drawing down annual EDP/MACC cloud commitments.
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CommittedSpendBurndownVelocityExtractor(BaseEstimator, TransformerMixin):
    """CommittedSpendBurndownVelocityExtractor: Speed of drawing down annual EDP/MACC cloud commitments."""
    def __init__(self, prefix: str = "committed_spend_burndown_velocity"):
        self.prefix = prefix
        self.stats_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = pd.DataFrame(X).copy()
        if "monthly_charge" in X.columns:
            self.stats_["charge_mean"] = float(X["monthly_charge"].mean())
        if "tenure_months" in X.columns:
            self.stats_["tenure_median"] = float(X["tenure_months"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        n = len(X_out)
        
        if "tenure_months" in X_out.columns and "monthly_charge" in X_out.columns:
            tenure = np.maximum(1.0, X_out["tenure_months"].values)
            charge = np.maximum(0.0, X_out["monthly_charge"].values)
            
            # Domain-specific mathematical transformation
            decay = np.exp(-0.05 * tenure)
            charge_norm = charge / (self.stats_.get("charge_mean", 80.0) + 1e-5)
            
            X_out[f"{self.prefix}_signal"] = np.log1p(charge) * (1.0 / (np.sqrt(tenure) + 1.0))
            X_out[f"{self.prefix}_decay"] = decay
            X_out[f"{self.prefix}_risk_score"] = np.clip(decay * charge_norm * 2.2, 0.0, 1.0)
        else:
            X_out[f"{self.prefix}_signal"] = np.zeros(n)
            X_out[f"{self.prefix}_risk_score"] = np.ones(n) * 0.5
            
        return X_out
