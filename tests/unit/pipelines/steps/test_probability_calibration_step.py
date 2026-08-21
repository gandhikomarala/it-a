# Unit Test for CalibrationStep.
import pytest
from ml.pipelines.steps.probability_calibration_step import CalibrationStep

def test_probability_calibration_step_execution():
    step = CalibrationStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "probability_calibration_step"
    assert res["metrics"]["records_processed"] == 50
