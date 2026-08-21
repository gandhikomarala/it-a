# Unit Test for ADWINDriftDetector.
import pytest
import numpy as np
from ml.monitoring.metrics.adwin_adaptive_windowing import ADWINDriftDetector

def test_adwin_adaptive_windowing_no_drift():
    ref = np.random.normal(10.0, 2.0, 200)
    cur = np.random.normal(10.0, 2.0, 200)
    
    metric = ADWINDriftDetector(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert isinstance(res, dict)
    assert res["metric"] == "ADWINDriftDetector"
    assert res["is_drift_detected"] is False

def test_adwin_adaptive_windowing_severe_drift():
    ref = np.random.normal(10.0, 1.0, 200)
    cur = np.random.normal(50.0, 1.0, 200)
    
    metric = ADWINDriftDetector(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert res["is_drift_detected"] is True
    assert res["score"] >= 0.20
