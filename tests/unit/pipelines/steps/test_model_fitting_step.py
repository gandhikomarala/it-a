# Unit Test for ModelFittingStep.
import pytest
from ml.pipelines.steps.model_fitting_step import ModelFittingStep

def test_model_fitting_step_execution():
    step = ModelFittingStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "model_fitting_step"
    assert res["metrics"]["records_processed"] == 50
