# Gradient Boosting decision tree classifier.
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from .base import BaseModelWrapper

class GradientBoostingModel(BaseModelWrapper):
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__("GradientBoosting", hyperparameters)
        params = self.hyperparameters.copy()
        learning_rate = params.get("learning_rate", 0.08)
        max_iter = params.get("max_iter", 120)
        max_depth = params.get("max_depth", 6)
        min_samples_leaf = params.get("min_samples_leaf", 20)

        self.model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "GradientBoostingModel":
        self.feature_names_ = X.columns.tolist()
        self.model.fit(X.values, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names_].values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names_].values)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        n_feats = len(self.feature_names_)
        return {feat: float(1.0 / n_feats) for feat in self.feature_names_}
