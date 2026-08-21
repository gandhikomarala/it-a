# Unit Test for AdversarialAUCDetector.
import pytest
import numpy as np
from ml.monitoring.metrics.adversarial_auc_drift_detector import AdversarialAUCDetector

def test_adversarial_auc_drift_detector_no_drift():
    ref = np.random.normal(10.0, 2.0, 200)
    cur = np.random.normal(10.0, 2.0, 200)
    
    metric = AdversarialAUCDetector(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert isinstance(res, dict)
    assert res["metric"] == "AdversarialAUCDetector"
    assert res["is_drift_detected"] is False

def test_adversarial_auc_drift_detector_severe_drift():
    ref = np.random.normal(10.0, 1.0, 200)
    cur = np.random.normal(50.0, 1.0, 200)
    
    metric = AdversarialAUCDetector(threshold=0.20)
    res = metric.compute_drift(ref, cur)
    
    assert res["is_drift_detected"] is True
    assert res["score"] >= 0.20
