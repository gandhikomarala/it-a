# Unit Test for BiasAuditStep.
import pytest
from ml.pipelines.steps.bias_auditing_step import BiasAuditStep

def test_bias_auditing_step_execution():
    step = BiasAuditStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "bias_auditing_step"
    assert res["metrics"]["records_processed"] == 50
