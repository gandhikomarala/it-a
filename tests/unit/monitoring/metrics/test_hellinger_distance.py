# Unit Test for HellingerDistanceDrift.
import pytest
import numpy as np
from ml.monitoring.metrics.hellinger_distance import HellingerDistanceDrift

def test_hellinger_distance_no_drift():
    ref = np.random.normal(10.0, 2.0, 200)
    cur = np.random.normal(10.0, 2.0, 200)
    
    metric = HellingerDistanceDrift(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert isinstance(res, dict)
    assert res["metric"] == "HellingerDistanceDrift"
    assert res["is_drift_detected"] is False

def test_hellinger_distance_severe_drift():
    ref = np.random.normal(10.0, 1.0, 200)
    cur = np.random.normal(50.0, 1.0, 200)
    
    metric = HellingerDistanceDrift(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert res["is_drift_detected"] is True
    assert res["score"] >= 0.20
