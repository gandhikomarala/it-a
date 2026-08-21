# Outlier handling via Winsorization and Tukey IQR bounds.
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, factor: float = 2.5, exclude_cols: Optional[List[str]] = None):
        self.factor = factor
        self.exclude_cols = exclude_cols or []
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        X = pd.DataFrame(X).copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in self.exclude_cols or col.endswith("_is_missing") or col == "churn":
                continue
            series = X[col].dropna()
            if len(series) > 0:
                q25 = float(np.percentile(series, 25))
                q75 = float(np.percentile(series, 75))
                iqr = q75 - q25
                lower = q25 - (self.factor * iqr)
                upper = q75 + (self.factor * iqr)
                self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X
