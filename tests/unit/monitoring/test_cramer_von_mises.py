# Unit test for CramerVonMisesDriftCalculator.
import pytest
import numpy as np
from ml.monitoring.metrics.cramer_von_mises import CramerVonMisesDriftCalculator

def test_cramer_von_mises_calculation():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 100)
    curr = np.random.normal(0.2, 1.1, 100)
    
    res = CramerVonMisesDriftCalculator.calculate(ref, curr)
    assert isinstance(res, dict)
    assert "metric_name" in res
    assert "statistic" in res
    assert "drift_detected" in res
    assert res["severity"] in ["NORMAL", "WARNING", "CRITICAL"]
