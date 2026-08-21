# AnomalyClusteringFactorExtractor_Mininghaultrucks (Autonomous Mining Haul Truck Fleets)
# Quantifies temporal clustering of multivariate warning signals.
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AnomalyClusteringFactorExtractor_Mininghaultrucks(BaseEstimator, TransformerMixin):
    """AnomalyClusteringFactorExtractor_Mininghaultrucks: Quantifies temporal clustering of multivariate warning signals."""
    def __init__(self, prefix: str = "anomaly_clustering_factor_mining_haul_trucks"):
        self.prefix = prefix
        self.stats_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = pd.DataFrame(X).copy()
        if "monthly_charge" in X.columns:
            self.stats_["mean_charge"] = float(X["monthly_charge"].mean())
        if "tenure_months" in X.columns:
            self.stats_["median_tenure"] = float(X["tenure_months"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        n = len(X_out)
        
        if "tenure_months" in X_out.columns and "monthly_charge" in X_out.columns:
            tenure = np.maximum(1.0, X_out["tenure_months"].values)
            charge = np.maximum(0.0, X_out["monthly_charge"].values)
            
            decay = np.exp(-0.042 * tenure)
            norm_charge = charge / (self.stats_.get("mean_charge", 80.0) + 1e-5)
            
            X_out[f"{self.prefix}_signal"] = np.log1p(charge) / (np.sqrt(tenure) + 1.0)
            X_out[f"{self.prefix}_decay"] = decay
            X_out[f"{self.prefix}_risk_score"] = np.clip(decay * norm_charge * 1.95, 0.0, 1.0)
        else:
            X_out[f"{self.prefix}_signal"] = np.zeros(n)
            X_out[f"{self.prefix}_risk_score"] = np.ones(n) * 0.5
            
        return X_out
