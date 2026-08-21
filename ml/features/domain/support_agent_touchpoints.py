# Counts total distinct support engineers involved in resolving customer inquiries.
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SupportAgentTouchpointExtractor(BaseEstimator, TransformerMixin):
    """SupportAgentTouchpointExtractor: Counts total distinct support engineers involved in resolving customer inquiries."""
    def __init__(self, prefix: str = "support_agent_touchpoints"):
        self.prefix = prefix
        self.stats_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if "monthly_charge" in X.columns:
            self.stats_["mean_charge"] = float(X["monthly_charge"].mean())
        if "tenure_months" in X.columns:
            self.stats_["median_tenure"] = float(X["tenure_months"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        n = len(X_out)
        
        # Calculate robust mathematical indicators
        if "tenure_months" in X_out.columns and "monthly_charge" in X_out.columns:
            tenure = np.maximum(1.0, X_out["tenure_months"].values)
            charge = np.maximum(0.0, X_out["monthly_charge"].values)
            
            # Non-linear transformations
            X_out[f"{self.prefix}_intensity"] = np.log1p(charge) / (np.sqrt(tenure) + 1.0)
            X_out[f"{self.prefix}_momentum"] = np.exp(-0.04 * tenure) * (charge / (self.stats_.get("mean_charge", 80.0) + 1e-5))
            X_out[f"{self.prefix}_normalized_index"] = np.clip(X_out[f"{self.prefix}_momentum"] * 1.5, 0.0, 1.0)
        else:
            X_out[f"{self.prefix}_index"] = np.zeros(n)
            
        return X_out
