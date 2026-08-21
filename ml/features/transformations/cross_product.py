# Cross-product elasticity and service tier engagement features.
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CrossProductElasticityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, diversity_col: str = "feature_usage_diversity_score"):
        self.diversity_col = diversity_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        if self.diversity_col in X_out.columns:
            X_out["product_stickiness_score"] = np.sqrt(np.clip(X_out[self.diversity_col], 0.0, 1.0))
            X_out["product_single_feature_risk"] = (X_out[self.diversity_col] < 0.25).astype(int)

        if "subscription_type" in X_out.columns and "contract_type" in X_out.columns:
            # Contract tier stability score
            is_annual = X_out["contract_type"].isin(["One-Year", "Two-Year"]).astype(float)
            is_enterprise = X_out["subscription_type"].isin(["Premium", "Enterprise"]).astype(float)
            X_out["product_enterprise_lock_in"] = is_annual * is_enterprise

        return X_out
