# Unit Test for DemographicParityDriftMonitor.
import pytest
import numpy as np
from ml.monitoring.metrics.sliced_demographic_parity_drift import DemographicParityDriftMonitor

def test_sliced_demographic_parity_drift_no_drift():
    ref = np.random.normal(10.0, 2.0, 200)
    cur = np.random.normal(10.0, 2.0, 200)
    
    metric = DemographicParityDriftMonitor(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert isinstance(res, dict)
    assert res["metric"] == "DemographicParityDriftMonitor"
    assert res["is_drift_detected"] is False

def test_sliced_demographic_parity_drift_severe_drift():
    ref = np.random.normal(10.0, 1.0, 200)
    cur = np.random.normal(50.0, 1.0, 200)
    
    metric = DemographicParityDriftMonitor(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert res["is_drift_detected"] is True
    assert res["score"] >= 0.20
