# Prediction output drift tracking.
from typing import Dict, Any
import numpy as np

class PredictionDriftMonitor:
    @staticmethod
    def audit_probabilities(baseline_probs: np.ndarray, current_probs: np.ndarray) -> Dict[str, Any]:
        from packages.utilities.math_stats import compute_psi
        psi = compute_psi(baseline_probs, current_probs)
        return {
            "prediction_psi": round(psi, 4),
            "baseline_mean": round(float(np.mean(baseline_probs)), 4),
            "current_mean": round(float(np.mean(current_probs)), 4),
            "is_drifted": psi >= 0.20
        }
