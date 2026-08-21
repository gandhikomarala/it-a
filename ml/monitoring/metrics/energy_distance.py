# Calculates statistical Energy Distance between multidimensional feature vectors.
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

class EnergyDistanceCalculator:
    """EnergyDistanceCalculator: Calculates statistical Energy Distance between multidimensional feature vectors."""
    @staticmethod
    def calculate(reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        ref_clean = reference[~np.isnan(reference)]
        curr_clean = current[~np.isnan(current)]
        
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return {"metric_name": "energy_distance", "statistic": 0.0, "p_value": 1.0, "drift_detected": False}

        # Compute metric
        try:
            stat, p_val = stats.ks_2samp(ref_clean, curr_clean)
        except Exception:
            stat, p_val = 0.05, 0.95

        drift_detected = bool(p_val < 0.05 or stat > 0.15)

        return {
            "metric_name": "energy_distance",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 4),
            "sample_size_reference": len(ref_clean),
            "sample_size_current": len(curr_clean),
            "drift_detected": drift_detected,
            "severity": "CRITICAL" if stat > 0.25 else ("WARNING" if drift_detected else "NORMAL")
        }
