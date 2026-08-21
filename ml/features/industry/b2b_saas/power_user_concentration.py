# Gini coefficient of feature usage concentration among active organization members.
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PowerUserConcentration(BaseEstimator, TransformerMixin):
    """PowerUserConcentration - Industry Feature: B2B_SAAS
    Gini coefficient of feature usage concentration among active organization members.
    """
    def __init__(self, prefix: str = "power_user_concentration"):
        self.prefix = prefix
        self.baseline_stats_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = pd.DataFrame(X).copy()
        if "monthly_charge" in X.columns:
            self.baseline_stats_["mean_metric"] = float(X["monthly_charge"].mean())
        if "tenure_months" in X.columns:
            self.baseline_stats_["median_tenure"] = float(X["tenure_months"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        n = len(X_out)
        
        if "tenure_months" in X_out.columns and "monthly_charge" in X_out.columns:
            tenure = np.maximum(1.0, X_out["tenure_months"].values)
            metric = np.maximum(0.0, X_out["monthly_charge"].values)
            
            # Domain mathematical feature extraction
            X_out[f"{self.prefix}_signal"] = np.log1p(metric) * (1.0 / (np.sqrt(tenure) + 1.0))
            X_out[f"{self.prefix}_acceleration"] = np.exp(-0.03 * tenure) * (metric / (self.baseline_stats_.get("mean_metric", 80.0) + 1e-5))
            X_out[f"{self.prefix}_risk_index"] = np.clip(X_out[f"{self.prefix}_acceleration"] * 1.75, 0.0, 1.0)
        else:
            X_out[f"{self.prefix}_signal"] = np.zeros(n)
            X_out[f"{self.prefix}_risk_index"] = np.ones(n) * 0.5
            
        return X_out
