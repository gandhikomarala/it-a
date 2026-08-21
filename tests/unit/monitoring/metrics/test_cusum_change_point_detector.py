# Unit Test for CUSUMChangePointDetector.
import pytest
import numpy as np
from ml.monitoring.metrics.cusum_change_point_detector import CUSUMChangePointDetector

def test_cusum_change_point_detector_no_drift():
    ref = np.random.normal(10.0, 2.0, 200)
    cur = np.random.normal(10.0, 2.0, 200)
    
    metric = CUSUMChangePointDetector(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert isinstance(res, dict)
    assert res["metric"] == "CUSUMChangePointDetector"
    assert res["is_drift_detected"] is False

def test_cusum_change_point_detector_severe_drift():
    ref = np.random.normal(10.0, 1.0, 200)
    cur = np.random.normal(50.0, 1.0, 200)
    
    metric = CUSUMChangePointDetector(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert res["is_drift_detected"] is True
    assert res["score"] >= 0.20
