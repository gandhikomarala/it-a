# Unit Test for OutlierClippingStep.
import pytest
from ml.pipelines.steps.outlier_clipping_step import OutlierClippingStep

def test_outlier_clipping_step_execution():
    step = OutlierClippingStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "outlier_clipping_step"
    assert res["metrics"]["records_processed"] == 50
