# Unit test for BrierScoreCalibrationMonitor.
import pytest
import numpy as np
from ml.monitoring.metrics.brier_score_monitor import BrierScoreCalibrationMonitor

def test_brier_score_calculation():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 100)
    curr = np.random.normal(0.2, 1.1, 100)
    
    res = BrierScoreCalibrationMonitor.calculate(ref, curr)
    assert isinstance(res, dict)
    assert "metric_name" in res
    assert "statistic" in res
    assert "drift_detected" in res
    assert res["severity"] in ["NORMAL", "WARNING", "CRITICAL"]
