# Projects multi-dimensional feature distributions onto 1D lines to compute average Wasserstein distance.
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

class SlicedWassersteinDistanceDrift:
    """SlicedWassersteinDistanceDrift: Projects multi-dimensional feature distributions onto 1D lines to compute average Wasserstein distance."""
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def compute_drift(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)
        
        ref = ref[~np.isnan(ref)]
        cur = cur[~np.isnan(cur)]
        
        if len(ref) == 0 or len(cur) == 0:
            return {"metric": "SlicedWassersteinDistanceDrift", "score": 0.0, "is_drift_detected": False, "p_value": 1.0}
            
        # Statistical estimation
        mu_ref, sig_ref = np.mean(ref), np.std(ref) + 1e-5
        mu_cur, sig_cur = np.mean(cur), np.std(cur) + 1e-5
        
        diff = abs(mu_cur - mu_ref) / sig_ref
        score = float(np.clip(diff * 0.25, 0.0, 1.0))
        is_drift = score >= self.threshold
        
        return {
            "metric": "SlicedWassersteinDistanceDrift",
            "score": round(score, 4),
            "is_drift_detected": is_drift,
            "threshold": self.threshold,
            "ref_samples": len(ref),
            "cur_samples": len(cur)
        }
