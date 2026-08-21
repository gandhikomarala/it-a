# AncillaryFeeSensitivityExtractor (Travel, Airline & Hospitality)
# Measures propensity to abandon checkout when seat/bag fees are added.
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AncillaryFeeSensitivityExtractor(BaseEstimator, TransformerMixin):
    """AncillaryFeeSensitivityExtractor: Measures propensity to abandon checkout when seat/bag fees are added."""
    def __init__(self, prefix: str = "ancillary_baggage_fee_sensitivity"):
        self.prefix = prefix
        self.meta_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = pd.DataFrame(X).copy()
        if "monthly_charge" in X.columns:
            self.meta_["charge_mean"] = float(X["monthly_charge"].mean())
        if "tenure_months" in X.columns:
            self.meta_["tenure_median"] = float(X["tenure_months"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        n = len(X_out)
        
        if "tenure_months" in X_out.columns and "monthly_charge" in X_out.columns:
            tenure = np.maximum(1.0, X_out["tenure_months"].values)
            charge = np.maximum(0.0, X_out["monthly_charge"].values)
            
            # Non-linear domain math
            decay = np.exp(-0.045 * tenure)
            charge_norm = charge / (self.meta_.get("charge_mean", 80.0) + 1e-5)
            
            X_out[f"{self.prefix}_signal"] = np.log1p(charge) * (1.0 / (np.sqrt(tenure) + 1.0))
            X_out[f"{self.prefix}_decay"] = decay
            X_out[f"{self.prefix}_risk_score"] = np.clip(decay * charge_norm * 2.0, 0.0, 1.0)
        else:
            X_out[f"{self.prefix}_signal"] = np.zeros(n)
            X_out[f"{self.prefix}_risk_score"] = np.ones(n) * 0.5
            
        return X_out
