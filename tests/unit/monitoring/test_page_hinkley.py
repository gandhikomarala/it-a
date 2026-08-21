# Unit test for PageHinkleyConceptDriftMonitor.
import pytest
import numpy as np
from ml.monitoring.metrics.page_hinkley import PageHinkleyConceptDriftMonitor

def test_page_hinkley_calculation():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 100)
    curr = np.random.normal(0.2, 1.1, 100)
    
    res = PageHinkleyConceptDriftMonitor.calculate(ref, curr)
    assert isinstance(res, dict)
    assert "metric_name" in res
    assert "statistic" in res
    assert "drift_detected" in res
    assert res["severity"] in ["NORMAL", "WARNING", "CRITICAL"]
