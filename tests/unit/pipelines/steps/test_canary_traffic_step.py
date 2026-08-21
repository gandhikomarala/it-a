# Unit Test for CanaryPromotionStep.
import pytest
from ml.pipelines.steps.canary_traffic_step import CanaryPromotionStep

def test_canary_traffic_step_execution():
    step = CanaryPromotionStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "canary_traffic_step"
    assert res["metrics"]["records_processed"] == 50
