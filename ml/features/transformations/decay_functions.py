# Mathematical exponential decay and half-life transformers.
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ExponentialDecayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[str]] = None, half_life_days: float = 14.0):
        self.columns = columns or ["days_since_last_login"]
        self.half_life_days = half_life_days
        self.decay_constant = np.log(2) / half_life_days

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.columns:
            if col in X_out.columns:
                X_out[f"{col}_exp_decay"] = np.exp(-self.decay_constant * np.maximum(0.0, X_out[col]))
        return X_out
