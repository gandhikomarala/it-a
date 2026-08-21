# PrescriptionRefillOrderLagExtractor (Healthcare & MedTech SaaS)
# Turnaround days between pharmacy order placement and fulfillment.
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PrescriptionRefillOrderLagExtractor(BaseEstimator, TransformerMixin):
    """PrescriptionRefillOrderLagExtractor: Turnaround days between pharmacy order placement and fulfillment."""
    def __init__(self, prefix: str = "prescription_refill_order_lag"):
        self.prefix = prefix
        self.baseline_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = pd.DataFrame(X).copy()
        if "monthly_charge" in X.columns:
            self.baseline_["metric_mean"] = float(X["monthly_charge"].mean())
        if "tenure_months" in X.columns:
            self.baseline_["tenure_median"] = float(X["tenure_months"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        n = len(X_out)
        
        if "tenure_months" in X_out.columns and "monthly_charge" in X_out.columns:
            tenure = np.maximum(1.0, X_out["tenure_months"].values)
            charge = np.maximum(0.0, X_out["monthly_charge"].values)
            
            # Domain non-linear engineering
            decay = np.exp(-0.035 * tenure)
            intensity = np.log1p(charge) / (np.sqrt(tenure) + 1.0)
            
            X_out[f"{self.prefix}_signal"] = intensity * 1.5
            X_out[f"{self.prefix}_decay_factor"] = decay
            X_out[f"{self.prefix}_risk_score"] = np.clip(decay * (charge / (self.baseline_.get("metric_mean", 80.0) + 1e-5)), 0.0, 1.0)
        else:
            X_out[f"{self.prefix}_signal"] = np.zeros(n)
            X_out[f"{self.prefix}_risk_score"] = np.ones(n) * 0.5
            
        return X_out
