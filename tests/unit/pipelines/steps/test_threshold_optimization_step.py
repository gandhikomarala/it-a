# Unit Test for CostThresholdStep.
import pytest
from ml.pipelines.steps.threshold_optimization_step import CostThresholdStep

def test_threshold_optimization_step_execution():
    step = CostThresholdStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "threshold_optimization_step"
    assert res["metrics"]["records_processed"] == 50
