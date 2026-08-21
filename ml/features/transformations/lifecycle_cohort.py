# Customer lifecycle stage and onboarding cohort features.
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LifecycleCohortExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, tenure_col: str = "tenure_months"):
        self.tenure_col = tenure_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        if self.tenure_col in X_out.columns:
            tenure = X_out[self.tenure_col]
            # Lifecycle stages
            X_out["lifecycle_is_onboarding"] = (tenure <= 3).astype(int)
            X_out["lifecycle_is_early_adoption"] = ((tenure > 3) & (tenure <= 12)).astype(int)
            X_out["lifecycle_is_mature"] = ((tenure > 12) & (tenure <= 36)).astype(int)
            X_out["lifecycle_is_champion"] = (tenure > 36).astype(int)
            X_out["lifecycle_hazard_weight"] = 1.0 / (np.sqrt(tenure) + 1.0)

        return X_out
