# Stratified K-Fold cross validation manager.
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

class CrossValidationManager:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def evaluate_cv(self, model_class: Any, X: pd.DataFrame, y: np.ndarray, hyperparameters: Dict[str, Any]) -> Dict[str, float]:
        auc_scores = []
        f1_scores = []

        for train_idx, val_idx in self.cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            m = model_class(hyperparameters=hyperparameters)
            m.fit(X_train, y_train)

            probs = m.predict_proba(X_val)
            preds = (probs >= 0.50).astype(int)

            auc_scores.append(roc_auc_score(y_val, probs))
            f1_scores.append(f1_score(y_val, preds, zero_division=0))

        return {
            "cv_mean_roc_auc": float(np.mean(auc_scores)),
            "cv_std_roc_auc": float(np.std(auc_scores)),
            "cv_mean_f1": float(np.mean(f1_scores)),
            "cv_std_f1": float(np.std(f1_scores))
        }
