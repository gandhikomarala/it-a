# Unit Test for ModelCardStep.
import pytest
from ml.pipelines.steps.model_card_generation_step import ModelCardStep

def test_model_card_generation_step_execution():
    step = ModelCardStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "model_card_generation_step"
    assert res["metrics"]["records_processed"] == 50
