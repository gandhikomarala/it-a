# Unit Test for EvaluationStep.
import pytest
from ml.pipelines.steps.model_evaluation_step import EvaluationStep

def test_model_evaluation_step_execution():
    step = EvaluationStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "model_evaluation_step"
    assert res["metrics"]["records_processed"] == 50
