# Unit Test for TemporalSplitStep.
import pytest
from ml.pipelines.steps.train_test_split_step import TemporalSplitStep

def test_train_test_split_step_execution():
    step = TemporalSplitStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "train_test_split_step"
    assert res["metrics"]["records_processed"] == 50
