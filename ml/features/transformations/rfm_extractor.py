# Recency, Frequency, Monetary (RFM) behavioral feature extraction.
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RFMFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        recency_col: str = "days_since_last_login",
        frequency_col: str = "login_count_monthly",
        monetary_col: str = "monthly_charge",
        tenure_col: str = "tenure_months"
    ):
        self.recency_col = recency_col
        self.frequency_col = frequency_col
        self.monetary_col = monetary_col
        self.tenure_col = tenure_col
        self.r_quantiles = {}
        self.f_quantiles = {}
        self.m_quantiles = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.recency_col in X.columns:
            self.r_quantiles = np.percentile(X[self.recency_col].dropna(), [25, 50, 75]).tolist()
        if self.frequency_col in X.columns:
            self.f_quantiles = np.percentile(X[self.frequency_col].dropna(), [25, 50, 75]).tolist()
        if self.monetary_col in X.columns:
            self.m_quantiles = np.percentile(X[self.monetary_col].dropna(), [25, 50, 75]).tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # Calculate composite RFM Score
        if self.recency_col in X_out.columns and self.frequency_col in X_out.columns:
            # Lower recency is better (higher score), higher frequency is better
            r_score = np.digitize(X_out[self.recency_col], bins=self.r_quantiles)
            f_score = 4 - np.digitize(X_out[self.frequency_col], bins=self.f_quantiles)
            m_score = 4 - np.digitize(X_out[self.monetary_col], bins=self.m_quantiles) if self.monetary_col in X_out.columns else 2
            
            X_out["rfm_recency_score"] = 4 - r_score
            X_out["rfm_frequency_score"] = f_score
            X_out["rfm_monetary_score"] = m_score
            X_out["rfm_composite_index"] = (X_out["rfm_recency_score"] * 0.40) + (X_out["rfm_frequency_score"] * 0.35) + (X_out["rfm_monetary_score"] * 0.25)
            X_out["rfm_dormancy_ratio"] = X_out[self.recency_col] / (X_out[self.tenure_col] * 30.0 + 1.0)
            X_out["rfm_usage_intensity"] = X_out[self.frequency_col] / (X_out[self.recency_col] + 1.0)
            
        return X_out
