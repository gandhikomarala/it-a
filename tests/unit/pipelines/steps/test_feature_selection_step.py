# Unit Test for FeatureSelectionStep.
import pytest
from ml.pipelines.steps.feature_selection_step import FeatureSelectionStep

def test_feature_selection_step_execution():
    step = FeatureSelectionStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "feature_selection_step"
    assert res["metrics"]["records_processed"] == 50
