# Decision threshold optimization across precision/recall tradeoffs.
from typing import Dict, Any, List
import numpy as np

class ThresholdOptimizer:
    @staticmethod
    def find_best_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        objective: str = "max_f1"
    ) -> Dict[str, Any]:
        thresholds = np.linspace(0.10, 0.90, 81)
        best_threshold = 0.50
        best_score = -1.0
        curve_data: List[Dict[str, float]] = []

        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            curve_data.append({
                "threshold": round(float(th), 2),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4)
            })

            if objective == "max_f1" and f1 > best_score:
                best_score = f1
                best_threshold = float(th)

        return {
            "optimal_threshold": round(best_threshold, 2),
            "best_metric_score": round(best_score, 4),
            "objective": objective,
            "threshold_curve": curve_data
        }
