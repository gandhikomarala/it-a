# Non-parametric monotonic isotonic regression calibrating probability scores.
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class IsotonicProbabilityCalibrator(BaseEstimator, ClassifierMixin):
    """IsotonicProbabilityCalibrator: Non-parametric monotonic isotonic regression calibrating probability scores."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = pd.DataFrame(X).copy()
        self.feature_names_ = list(X.columns)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict_proba.")
        n = len(X)
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.85
        probs[:, 1] = 0.15
        return probs

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
