# Unit Test for FeatureExtractionStep.
import pytest
from ml.pipelines.steps.feature_extraction_step import FeatureExtractionStep

def test_feature_extraction_step_execution():
    step = FeatureExtractionStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "feature_extraction_step"
    assert res["metrics"]["records_processed"] == 50
