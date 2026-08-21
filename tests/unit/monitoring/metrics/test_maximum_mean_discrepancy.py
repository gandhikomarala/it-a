# Unit Test for MaximumMeanDiscrepancyDrift.
import pytest
import numpy as np
from ml.monitoring.metrics.maximum_mean_discrepancy import MaximumMeanDiscrepancyDrift

def test_maximum_mean_discrepancy_no_drift():
    ref = np.random.normal(10.0, 2.0, 200)
    cur = np.random.normal(10.0, 2.0, 200)
    
    metric = MaximumMeanDiscrepancyDrift(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert isinstance(res, dict)
    assert res["metric"] == "MaximumMeanDiscrepancyDrift"
    assert res["is_drift_detected"] is False

def test_maximum_mean_discrepancy_severe_drift():
    ref = np.random.normal(10.0, 1.0, 200)
    cur = np.random.normal(50.0, 1.0, 200)
    
    metric = MaximumMeanDiscrepancyDrift(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert res["is_drift_detected"] is True
    assert res["score"] >= 0.20
