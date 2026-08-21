# Unit Test for DriftBaselineStep.
import pytest
from ml.pipelines.steps.drift_baseline_step import DriftBaselineStep

def test_drift_baseline_step_execution():
    step = DriftBaselineStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "drift_baseline_step"
    assert res["metrics"]["records_processed"] == 50
