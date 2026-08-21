# Usage trajectory and activity velocity extraction.
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class UsageVelocityExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        daily_hours_col: str = "daily_usage_hours",
        weekly_hours_col: str = "weekly_usage_hours",
        monthly_hours_col: str = "monthly_usage_hours"
    ):
        self.daily_hours_col = daily_hours_col
        self.weekly_hours_col = weekly_hours_col
        self.monthly_hours_col = monthly_hours_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        if self.daily_hours_col in X_out.columns and self.weekly_hours_col in X_out.columns:
            # Expected weekly based on daily vs actual weekly
            expected_weekly = X_out[self.daily_hours_col] * 7.0
            X_out["usage_weekly_consistency_ratio"] = (X_out[self.weekly_hours_col] + 0.1) / (expected_weekly + 0.1)
            
        if self.weekly_hours_col in X_out.columns and self.monthly_hours_col in X_out.columns:
            expected_monthly = X_out[self.weekly_hours_col] * 4.33
            X_out["usage_monthly_decay_velocity"] = (X_out[self.monthly_hours_col] + 0.1) / (expected_monthly + 0.1)
            X_out["usage_dropoff_flag"] = (X_out["usage_monthly_decay_velocity"] < 0.70).astype(int)

        return X_out
