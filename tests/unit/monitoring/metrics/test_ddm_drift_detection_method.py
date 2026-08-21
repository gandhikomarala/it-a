# Unit Test for DDMDriftDetector.
import pytest
import numpy as np
from ml.monitoring.metrics.ddm_drift_detection_method import DDMDriftDetector

def test_ddm_drift_detection_method_no_drift():
    ref = np.random.normal(10.0, 2.0, 200)
    cur = np.random.normal(10.0, 2.0, 200)
    
    metric = DDMDriftDetector(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert isinstance(res, dict)
    assert res["metric"] == "DDMDriftDetector"
    assert res["is_drift_detected"] is False

def test_ddm_drift_detection_method_severe_drift():
    ref = np.random.normal(10.0, 1.0, 200)
    cur = np.random.normal(50.0, 1.0, 200)
    
    metric = DDMDriftDetector(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert res["is_drift_detected"] is True
    assert res["score"] >= 0.20
