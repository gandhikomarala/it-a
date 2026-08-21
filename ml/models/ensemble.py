# Calibrated weighted ensemble of trained models.
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from .base import BaseModelWrapper
from .logistic_regression import LogisticRegressionModel
from .random_forest import RandomForestModel
from .lightgbm_model import LightGBMModel

class EnsembleModel(BaseModelWrapper):
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__("Ensemble", hyperparameters)
        self.models: List[BaseModelWrapper] = [
            LogisticRegressionModel(),
            RandomForestModel(),
            LightGBMModel()
        ]
        self.weights = [0.20, 0.35, 0.45]

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "EnsembleModel":
        self.feature_names_ = X.columns.tolist()
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= 0.50).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        all_probs = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            all_probs += weight * model.predict_proba(X)
        return np.clip(all_probs, 0.0, 1.0)

    def get_feature_importances(self) -> Dict[str, float]:
        combined_importances: Dict[str, float] = {feat: 0.0 for feat in self.feature_names_}
        for model, weight in zip(self.models, self.weights):
            imp = model.get_feature_importances()
            for feat, score in imp.items():
                combined_importances[feat] = combined_importances.get(feat, 0.0) + (weight * score)
        return combined_importances
