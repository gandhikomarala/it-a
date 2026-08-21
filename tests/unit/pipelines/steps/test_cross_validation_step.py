# Unit Test for StratifiedCVStep.
import pytest
from ml.pipelines.steps.cross_validation_step import StratifiedCVStep

def test_cross_validation_step_execution():
    step = StratifiedCVStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "cross_validation_step"
    assert res["metrics"]["records_processed"] == 50
