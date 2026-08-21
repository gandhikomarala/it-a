# Unit Test for SchemaValidationStep.
import pytest
from ml.pipelines.steps.schema_validation_step import SchemaValidationStep

def test_schema_validation_step_execution():
    step = SchemaValidationStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "schema_validation_step"
    assert res["metrics"]["records_processed"] == 50
