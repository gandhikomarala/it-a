# Non-linear domain interaction terms.
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DomainInteractionExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # Inactivity x Monthly Charge (High charge customers who stop logging in have extreme churn risk)
        if "days_since_last_login" in X_out.columns and "monthly_charge" in X_out.columns:
            X_out["interaction_inactivity_x_charge"] = X_out["days_since_last_login"] * X_out["monthly_charge"]

        # Complaints x Payment Failures
        if "complaint_count" in X_out.columns and "payment_failures_count" in X_out.columns:
            X_out["interaction_complaints_x_failures"] = X_out["complaint_count"] * X_out["payment_failures_count"]

        # Low CSAT x Month-to-Month
        if "satisfaction_score" in X_out.columns and "contract_type" in X_out.columns:
            m2m = (X_out["contract_type"] == "Month-to-Month").astype(float)
            csat_deficit = np.maximum(0.0, 3.5 - X_out["satisfaction_score"])
            X_out["interaction_m2m_x_low_csat"] = m2m * csat_deficit

        return X_out
