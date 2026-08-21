# Unit Test for ModelRegistryStep.
import pytest
from ml.pipelines.steps.model_registry_step import ModelRegistryStep

def test_model_registry_step_execution():
    step = ModelRegistryStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "model_registry_step"
    assert res["metrics"]["records_processed"] == 50
