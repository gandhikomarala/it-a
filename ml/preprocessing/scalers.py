# Adaptive robust numerical scaling transformers.
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = "robust", exclude_cols: Optional[List[str]] = None):
        self.method = method
        self.exclude_cols = exclude_cols or []
        self.scaler = RobustScaler() if method == "robust" else StandardScaler()
        self.scaled_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        X = pd.DataFrame(X).copy()
        self.scaled_cols_ = [
            c for c in X.select_dtypes(include=[np.number]).columns
            if c not in self.exclude_cols and not c.endswith("_is_missing") and c != "churn"
        ]
        if self.scaled_cols_:
            self.scaler.fit(X[self.scaled_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        if self.scaled_cols_:
            scaled_vals = self.scaler.transform(X[self.scaled_cols_])
            for i, col in enumerate(self.scaled_cols_):
                X[col] = scaled_vals[:, i]
        return X
