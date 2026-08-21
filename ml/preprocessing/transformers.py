# Feature interaction and non-linear mathematical transformers.
from typing import List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()

        if "monthly_charge" in X.columns:
            X["log_monthly_charge"] = np.log1p(np.maximum(0, X["monthly_charge"]))
        if "total_spend" in X.columns:
            X["log_total_spend"] = np.log1p(np.maximum(0, X["total_spend"]))
        if "income" in X.columns:
            X["log_income"] = np.log1p(np.maximum(0, X["income"]))

        if "monthly_charge" in X.columns and "income" in X.columns:
            monthly_income = (X["income"] / 12.0) + self.epsilon
            X["charge_to_income_ratio"] = (X["monthly_charge"] / monthly_income).clip(0, 1.0)

        if "total_spend" in X.columns and "tenure_months" in X.columns:
            X["spend_per_tenure_month"] = X["total_spend"] / (X["tenure_months"] + 1.0)

        if "complaint_count" in X.columns and "ticket_count" in X.columns:
            X["complaint_ratio"] = X["complaint_count"] / (X["ticket_count"] + 1.0)

        if "payment_failures_count" in X.columns and "late_payments_count" in X.columns:
            X["total_payment_issues"] = X["payment_failures_count"] * 2 + X["late_payments_count"]

        return X
