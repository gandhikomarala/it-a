# Customer support friction and unresolved ticket indicators.
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SupportRiskExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        tickets_col: str = "ticket_count",
        complaints_col: str = "complaint_count",
        satisfaction_col: str = "satisfaction_score",
        resolution_hours_col: str = "average_resolution_hours"
    ):
        self.tickets_col = tickets_col
        self.complaints_col = complaints_col
        self.satisfaction_col = satisfaction_col
        self.resolution_hours_col = resolution_hours_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        if self.tickets_col in X_out.columns and self.complaints_col in X_out.columns:
            X_out["support_complaint_ticket_ratio"] = (X_out[self.complaints_col] + 0.1) / (X_out[self.tickets_col] + 1.0)
            X_out["support_high_friction_flag"] = ((X_out[self.complaints_col] >= 2) | (X_out[self.tickets_col] >= 5)).astype(int)

        if self.satisfaction_col in X_out.columns:
            # Low satisfaction penalty score
            X_out["support_satisfaction_deficit"] = np.maximum(0.0, 3.5 - X_out[self.satisfaction_col])
            X_out["support_severe_detractor_flag"] = (X_out[self.satisfaction_col] <= 2.0).astype(int)

        if self.resolution_hours_col in X_out.columns:
            X_out["support_sla_breach_risk"] = (X_out[self.resolution_hours_col] > 48.0).astype(int)
            X_out["support_log_resolution_hours"] = np.log1p(np.maximum(0.0, X_out[self.resolution_hours_col]))

        return X_out
