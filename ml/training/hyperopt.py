# Hyperparameter tuning engine supporting FAST, STANDARD, and FULL modes.
from typing import Dict, Any

class HyperparameterTuner:
    @staticmethod
    def get_search_grid(algorithm: str, mode: str = "STANDARD") -> Dict[str, Any]:
        if algorithm == "LightGBM":
            if mode == "FAST":
                return {"n_estimators": 100, "learning_rate": 0.08, "num_leaves": 31}
            elif mode == "STANDARD":
                return {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6}
            else:
                return {"n_estimators": 250, "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8, "subsample": 0.8}
        elif algorithm == "RandomForest":
            if mode == "FAST":
                return {"n_estimators": 100, "max_depth": 8}
            elif mode == "STANDARD":
                return {"n_estimators": 150, "max_depth": 10, "min_samples_split": 5}
            else:
                return {"n_estimators": 250, "max_depth": 15, "min_samples_split": 4}
        elif algorithm == "GradientBoosting":
            return {"learning_rate": 0.08, "max_iter": 120, "max_depth": 6}
        elif algorithm == "LogisticRegression":
            return {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
        return {}
