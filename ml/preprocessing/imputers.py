# Missing value imputation transformers with leakage protection.
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AdaptiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy_numeric: str = "median", strategy_categorical: str = "mode", add_indicator: bool = True):
        self.strategy_numeric = strategy_numeric
        self.strategy_categorical = strategy_categorical
        self.add_indicator = add_indicator
        self.numeric_impute_values_: Dict[str, float] = {}
        self.categorical_impute_values_: Dict[str, str] = {}
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        X = pd.DataFrame(X).copy()
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in self.numeric_cols_:
            series = X[col].dropna()
            if len(series) == 0:
                self.numeric_impute_values_[col] = 0.0
            elif self.strategy_numeric == "median":
                self.numeric_impute_values_[col] = float(series.median())
            else:
                self.numeric_impute_values_[col] = float(series.mean())

        for col in self.categorical_cols_:
            series = X[col].dropna()
            if len(series) == 0:
                self.categorical_impute_values_[col] = "UNKNOWN"
            else:
                self.categorical_impute_values_[col] = str(series.mode().iloc[0])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col in self.numeric_cols_:
            if col in X.columns:
                if self.add_indicator and X[col].isnull().any():
                    X[f"{col}_is_missing"] = X[col].isnull().astype(int)
                val = self.numeric_impute_values_.get(col, 0.0)
                X[col] = X[col].fillna(val)

        for col in self.categorical_cols_:
            if col in X.columns:
                val = self.categorical_impute_values_.get(col, "UNKNOWN")
                X[col] = X[col].fillna(val)

        return X
