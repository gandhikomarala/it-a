# Demographic parity, disparate impact, and equal opportunity bias auditor.
from typing import Dict, List, Any
import numpy as np
import pandas as pd

class ModelFairnessAuditor:
    # Audits ML model predictions for algorithmic bias across protected attributes.
    @staticmethod
    def audit_subgroup_fairness(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: pd.Series,
        attribute_name: str = "gender"
    ) -> Dict[str, Any]:
        groups = protected_attribute.unique()
        subgroup_metrics = {}

        for group in groups:
            mask = (protected_attribute == group).values
            n_group = np.sum(mask)
            if n_group == 0:
                continue

            y_t = y_true[mask]
            y_p = y_pred[mask]

            pos_rate = float(np.mean(y_p))
            tpr = float(np.sum((y_p == 1) & (y_t == 1)) / (np.sum(y_t == 1) + 1e-5))
            fpr = float(np.sum((y_p == 1) & (y_t == 0)) / (np.sum(y_t == 0) + 1e-5))

            subgroup_metrics[str(group)] = {
                "sample_size": int(n_group),
                "positive_prediction_rate": round(pos_rate, 4),
                "true_positive_rate": round(tpr, 4),
                "false_positive_rate": round(fpr, 4)
            }

        # Calculate Disparate Impact Ratio (DIR)
        pos_rates = [v["positive_prediction_rate"] for v in subgroup_metrics.values() if v["positive_prediction_rate"] > 0]
        disparate_impact_ratio = min(pos_rates) / max(pos_rates) if pos_rates else 1.0

        # Disparate impact is compliant under the 80% four-fifths rule
        is_compliant = disparate_impact_ratio >= 0.80

        return {
            "attribute_name": attribute_name,
            "disparate_impact_ratio": round(float(disparate_impact_ratio), 4),
            "is_four_fifths_compliant": bool(is_compliant),
            "subgroups": subgroup_metrics
        }
