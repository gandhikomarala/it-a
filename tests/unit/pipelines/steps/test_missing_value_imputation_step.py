# Unit Test for ImputationStep.
import pytest
from ml.pipelines.steps.missing_value_imputation_step import ImputationStep

def test_missing_value_imputation_step_execution():
    step = ImputationStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "missing_value_imputation_step"
    assert res["metrics"]["records_processed"] == 50
