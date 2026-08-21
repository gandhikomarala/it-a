# Unit Test for SymmetrizedKLDivergenceDrift.
import pytest
import numpy as np
from ml.monitoring.metrics.symmetrized_kl_divergence import SymmetrizedKLDivergenceDrift

def test_symmetrized_kl_divergence_no_drift():
    ref = np.random.normal(10.0, 2.0, 200)
    cur = np.random.normal(10.0, 2.0, 200)
    
    metric = SymmetrizedKLDivergenceDrift(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert isinstance(res, dict)
    assert res["metric"] == "SymmetrizedKLDivergenceDrift"
    assert res["is_drift_detected"] is False

def test_symmetrized_kl_divergence_severe_drift():
    ref = np.random.normal(10.0, 1.0, 200)
    cur = np.random.normal(50.0, 1.0, 200)
    
    metric = SymmetrizedKLDivergenceDrift(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert res["is_drift_detected"] is True
    assert res["score"] >= 0.20
