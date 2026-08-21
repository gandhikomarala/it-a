# Categorical encoding transformers with handling for unseen categories.
from typing import Dict, List, Optional, Set, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, max_categories: int = 15, drop_first: bool = True):
        self.max_categories = max_categories
        self.drop_first = drop_first
        self.cat_mappings_: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        X = pd.DataFrame(X).copy()
        cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns

        for col in cat_cols:
            if col in ["customer_id", "email", "phone", "first_name", "last_name"]:
                continue
            top_cats = X[col].value_counts().head(self.max_categories).index.tolist()
            self.cat_mappings_[col] = top_cats

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()

        for col, known_cats in self.cat_mappings_.items():
            if col in X.columns:
                clean_series = X[col].apply(lambda v: v if v in known_cats else "Other")
                dummies = pd.get_dummies(clean_series, prefix=col, drop_first=self.drop_first, dtype=int)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(columns=[col])

        return X
