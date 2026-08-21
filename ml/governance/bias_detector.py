# Algorithmic bias and equal opportunity auditor.
from typing import Dict, Any, List
import numpy as np
import pandas as pd

class ModelBiasAuditor:
    @staticmethod
    def evaluate_fairness(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        sensitive_attribute: pd.Series,
        attribute_name: str = "demographic_group"
    ) -> Dict[str, Any]:
        groups = sensitive_attribute.unique()
        group_results = {}

        for g in groups:
            mask = (sensitive_attribute == g).values
            n = np.sum(mask)
            if n == 0:
                continue

            yt_g = y_true[mask]
            yp_g = y_pred[mask]
            
            tpr = np.sum((yp_g == 1) & (yt_g == 1)) / (np.sum(yt_g == 1) + 1e-5)
            fpr = np.sum((yp_g == 1) & (yt_g == 0)) / (np.sum(yt_g == 0) + 1e-5)
            selection_rate = np.mean(yp_g)

            group_results[str(g)] = {
                "sample_size": int(n),
                "selection_rate": round(float(selection_rate), 4),
                "true_positive_rate": round(float(tpr), 4),
                "false_positive_rate": round(float(fpr), 4),
                "mean_churn_probability": round(float(np.mean(y_prob[mask])), 4)
            }

        rates = [v["selection_rate"] for v in group_results.values() if v["selection_rate"] > 0]
        disparate_impact = min(rates) / max(rates) if rates else 1.0

        return {
            "attribute_name": attribute_name,
            "disparate_impact_ratio": round(float(disparate_impact), 4),
            "passes_four_fifths_rule": bool(disparate_impact >= 0.80),
            "group_metrics": group_results
        }
