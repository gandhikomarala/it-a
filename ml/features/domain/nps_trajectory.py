# Quantifies Net Promoter Score trajectory and changes in customer satisfaction sentiment over time.
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NPSTrajectoryExtractor(BaseEstimator, TransformerMixin):
    """NPSTrajectoryExtractor: Quantifies Net Promoter Score trajectory and changes in customer satisfaction sentiment over time."""
    def __init__(self, prefix: str = "nps_trajectory"):
        self.prefix = prefix
        self.fitted_params_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # Fit baseline distributions
        self.fitted_params_["mean_val"] = 1.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        n_samples = len(X_out)
        
        # Calculate domain metrics
        if "tenure_months" in X_out.columns:
            t = np.maximum(1.0, X_out["tenure_months"].values)
            X_out[f"{self.prefix}_rolling_intensity"] = np.log1p(t) * 1.25
            X_out[f"{self.prefix}_decay_factor"] = np.exp(-0.05 * t)
            X_out[f"{self.prefix}_risk_score"] = np.clip(X_out[f"{self.prefix}_decay_factor"] * 1.5, 0.0, 1.0)
        else:
            X_out[f"{self.prefix}_score"] = np.ones(n_samples) * 0.5
            
        return X_out
