# Unit Test for TotalVariationDistanceDrift.
import pytest
import numpy as np
from ml.monitoring.metrics.total_variation_distance import TotalVariationDistanceDrift

def test_total_variation_distance_no_drift():
    ref = np.random.normal(10.0, 2.0, 200)
    cur = np.random.normal(10.0, 2.0, 200)
    
    metric = TotalVariationDistanceDrift(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert isinstance(res, dict)
    assert res["metric"] == "TotalVariationDistanceDrift"
    assert res["is_drift_detected"] is False

def test_total_variation_distance_severe_drift():
    ref = np.random.normal(10.0, 1.0, 200)
    cur = np.random.normal(50.0, 1.0, 200)
    
    metric = TotalVariationDistanceDrift(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert res["is_drift_detected"] is True
    assert res["score"] >= 0.20
