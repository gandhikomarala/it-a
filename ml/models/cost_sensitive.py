# Cost-sensitive classification model optimizing financial ROI.
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier

class CostSensitiveChurnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        cost_false_negative: float = 850.0,  # Lost customer CLV
        cost_false_positive: float = 50.0,   # Marketing retention offer cost
        cost_true_positive: float = 120.0,   # Intervention cost + success discount
        cost_true_negative: float = 0.0,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42
    ):
        self.cost_false_negative = cost_false_negative
        self.cost_false_positive = cost_false_positive
        self.cost_true_positive = cost_true_positive
        self.cost_true_negative = cost_true_negative
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.optimal_threshold = 0.50

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Calculate sample weight based on cost asymmetry
        sample_weights = np.where(y == 1, self.cost_false_negative / 100.0, 1.0)
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(X, y, sample_weight=sample_weights)

        # Optimize financial threshold on training predictions
        probs = self.model.predict_proba(X)[:, 1]
        best_cost = float("inf")
        best_th = 0.50
        for th in np.linspace(0.10, 0.90, 81):
            preds = (probs >= th).astype(int)
            tp = np.sum((preds == 1) & (y == 1))
            fp = np.sum((preds == 1) & (y == 0))
            fn = np.sum((preds == 0) & (y == 1))
            tn = np.sum((preds == 0) & (y == 0))
            
            total_financial_loss = (
                (fn * self.cost_false_negative) +
                (fp * self.cost_false_positive) +
                (tp * self.cost_true_positive) +
                (tn * self.cost_true_negative)
            )
            if total_financial_loss < best_cost:
                best_cost = total_financial_loss
                best_th = th

        self.optimal_threshold = float(best_th)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.optimal_threshold).astype(int)
