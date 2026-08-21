# Unit test for SlicedDemographicDriftCalculator.
import pytest
import numpy as np
from ml.monitoring.metrics.sliced_demographic_drift import SlicedDemographicDriftCalculator

def test_sliced_drift_calculation():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 100)
    curr = np.random.normal(0.2, 1.1, 100)
    
    res = SlicedDemographicDriftCalculator.calculate(ref, curr)
    assert isinstance(res, dict)
    assert "metric_name" in res
    assert "statistic" in res
    assert "drift_detected" in res
    assert res["severity"] in ["NORMAL", "WARNING", "CRITICAL"]
