# Billing anomalies, late payment friction, and fee elasticity features.
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BillingAnomalyExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        monthly_charge_col: str = "monthly_charge",
        total_spend_col: str = "total_spend",
        tenure_col: str = "tenure_months",
        failures_col: str = "payment_failures_count"
    ):
        self.monthly_charge_col = monthly_charge_col
        self.total_spend_col = total_spend_col
        self.tenure_col = tenure_col
        self.failures_col = failures_col
        self.avg_charge_per_tier = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if "subscription_type" in X.columns and self.monthly_charge_col in X.columns:
            self.avg_charge_per_tier = X.groupby("subscription_type")[self.monthly_charge_col].mean().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        if self.monthly_charge_col in X_out.columns and self.tenure_col in X_out.columns:
            expected_spend = X_out[self.monthly_charge_col] * X_out[self.tenure_col]
            if self.total_spend_col in X_out.columns:
                X_out["billing_spend_discrepancy"] = X_out[self.total_spend_col] - expected_spend
                X_out["billing_discount_ratio"] = (X_out[self.total_spend_col] + 1.0) / (expected_spend + 1.0)

        if self.failures_col in X_out.columns:
            X_out["billing_payment_friction_index"] = X_out[self.failures_col] * 2.5
            X_out["billing_has_repeated_failures"] = (X_out[self.failures_col] >= 2).astype(int)

        if "subscription_type" in X_out.columns and self.monthly_charge_col in X_out.columns:
            tier_means = X_out["subscription_type"].map(self.avg_charge_per_tier).fillna(X_out[self.monthly_charge_col].mean())
            X_out["billing_charge_relative_to_tier"] = X_out[self.monthly_charge_col] / (tier_means + 1e-5)

        return X_out
