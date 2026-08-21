# Soft-voting ensemble combining LightGBM, Random Forest, and ElasticNet.
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from ml.models.base import BaseModelWrapper

class VotingEnsembleChurnModel(BaseModelWrapper):
    """VotingEnsembleChurnModel: Soft-voting ensemble combining LightGBM, Random Forest, and ElasticNet."""
    def __init__(self, **hyperparameters):
        super().__init__(model_type="voting_classifier", **hyperparameters)
        n_est = hyperparameters.get("n_estimators", 100)
        lr = hyperparameters.get("learning_rate", 0.05)
        self.model = GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModelWrapper":
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        imps = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names_, imps)}
