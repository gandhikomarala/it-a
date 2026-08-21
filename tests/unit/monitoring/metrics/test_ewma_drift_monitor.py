# Unit Test for EWMADriftMonitor.
import pytest
import numpy as np
from ml.monitoring.metrics.ewma_drift_monitor import EWMADriftMonitor

def test_ewma_drift_monitor_no_drift():
    ref = np.random.normal(10.0, 2.0, 200)
    cur = np.random.normal(10.0, 2.0, 200)
    
    metric = EWMADriftMonitor(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert isinstance(res, dict)
    assert res["metric"] == "EWMADriftMonitor"
    assert res["is_drift_detected"] is False

def test_ewma_drift_monitor_severe_drift():
    ref = np.random.normal(10.0, 1.0, 200)
    cur = np.random.normal(50.0, 1.0, 200)
    
    metric = EWMADriftMonitor(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert res["is_drift_detected"] is True
    assert res["score"] >= 0.20
