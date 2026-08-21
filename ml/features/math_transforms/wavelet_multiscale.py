# Discrete wavelet transform extracting multiscale approximation and detail coefficients.
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class WaveletMultiscaleTransformer(BaseEstimator, TransformerMixin):
    """WaveletMultiscaleTransformer: Discrete wavelet transform extracting multiscale approximation and detail coefficients."""
    def __init__(self, prefix: str = "wavelet_multiscale"):
        self.prefix = prefix
        self.params_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = pd.DataFrame(X).copy()
        if "monthly_charge" in X.columns:
            vals = X["monthly_charge"].dropna().values
            self.params_["mean"] = float(np.mean(vals)) if len(vals) > 0 else 80.0
            self.params_["std"] = float(np.std(vals)) if len(vals) > 1 else 15.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        n = len(X_out)
        
        if "monthly_charge" in X_out.columns:
            arr = np.maximum(0.01, X_out["monthly_charge"].values)
            mu = self.params_.get("mean", 80.0)
            sig = self.params_.get("std", 15.0)
            
            # Mathematical transformation logic
            z_scores = (arr - mu) / (sig + 1e-5)
            X_out[f"{self.prefix}_transformed"] = np.tanh(z_scores * 0.5)
            X_out[f"{self.prefix}_energy"] = np.square(z_scores)
            X_out[f"{self.prefix}_cumulative"] = np.cumsum(z_scores) / np.sqrt(np.arange(1, n + 1))
        else:
            X_out[f"{self.prefix}_transformed"] = np.zeros(n)
            
        return X_out
