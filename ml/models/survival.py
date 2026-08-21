# Survival analysis and time-to-churn hazard estimation.
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class CoxProportionalHazardEstimator:
    # Cox Proportional Hazards hazard estimation engine.
    def __init__(self, baseline_hazard: float = 0.05):
        self.baseline_hazard = baseline_hazard
        self.weights = {}
        self.feature_names = []

    def fit(self, X: pd.DataFrame, durations: pd.Series, events: pd.Series):
        self.feature_names = list(X.columns)
        # Approximate partial log-likelihood weights using regularized logistic regression
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=500, random_state=42)
        clf.fit(X, events)
        for name, coef in zip(self.feature_names, clf.coef_[0]):
            self.weights[name] = float(coef)
        return self

    def predict_hazard_ratio(self, X: pd.DataFrame) -> np.ndarray:
        log_hazard = np.zeros(len(X))
        for col in self.feature_names:
            if col in X.columns:
                log_hazard += X[col].values * self.weights.get(col, 0.0)
        return np.exp(np.clip(log_hazard, -5.0, 5.0))

    def predict_survival_curve(self, X: pd.DataFrame, time_points: np.ndarray) -> np.ndarray:
        # S(t | x) = S_0(t)^exp(beta * x)
        hazard_ratios = self.predict_hazard_ratio(X)
        # Baseline cumulative survival
        s0 = np.exp(-self.baseline_hazard * (time_points / 12.0) ** 1.5)
        # Broadcast: shape (n_samples, n_time_points)
        survival_matrix = np.power(s0[np.newaxis, :], hazard_ratios[:, np.newaxis])
        return np.clip(survival_matrix, 0.0, 1.0)
