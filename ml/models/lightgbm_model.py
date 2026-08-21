# High-performance LightGBM gradient boosted tree model.
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from .base import BaseModelWrapper

class LightGBMModel(BaseModelWrapper):
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__("LightGBM", hyperparameters)
        params = self.hyperparameters.copy()
        n_estimators = params.get("n_estimators", 150)
        learning_rate = params.get("learning_rate", 0.05)
        num_leaves = params.get("num_leaves", 31)
        max_depth = params.get("max_depth", -1)
        min_child_samples = params.get("min_child_samples", 20)
        subsample = params.get("subsample", 0.8)
        colsample_bytree = params.get("colsample_bytree", 0.8)

        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "LightGBMModel":
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
        importances = self.model.feature_importances_
        total = np.sum(importances) or 1.0
        norm_importances = importances / total
        return {feat: float(score) for feat, score in zip(self.feature_names_, norm_importances)}
