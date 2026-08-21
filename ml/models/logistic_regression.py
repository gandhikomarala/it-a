# Regularized Logistic Regression baseline model.
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .base import BaseModelWrapper

class LogisticRegressionModel(BaseModelWrapper):
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__("LogisticRegression", hyperparameters)
        params = self.hyperparameters.copy()
        C_val = params.get("C", 1.0)
        max_iter = params.get("max_iter", 1000)
        solver = params.get("solver", "lbfgs")
        class_weight = params.get("class_weight", "balanced")
        
        self.model = LogisticRegression(
            C=C_val,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "LogisticRegressionModel":
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
        coefs = np.abs(self.model.coef_[0])
        total = np.sum(coefs) or 1.0
        norm_coefs = coefs / total
        return {feat: float(score) for feat, score in zip(self.feature_names_, norm_coefs)}
