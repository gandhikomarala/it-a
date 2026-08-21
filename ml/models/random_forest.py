# Random Forest classifier with ensemble bagging and depth control.
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModelWrapper

class RandomForestModel(BaseModelWrapper):
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__("RandomForest", hyperparameters)
        params = self.hyperparameters.copy()
        n_estimators = params.get("n_estimators", 150)
        max_depth = params.get("max_depth", 10)
        min_samples_split = params.get("min_samples_split", 5)
        class_weight = params.get("class_weight", "balanced")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "RandomForestModel":
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
        return {feat: float(score) for feat, score in zip(self.feature_names_, importances)}
