# Unit Test for SHAPExplainerStep.
import pytest
from ml.pipelines.steps.shap_explainer_step import SHAPExplainerStep

def test_shap_explainer_step_execution():
    step = SHAPExplainerStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "shap_explainer_step"
    assert res["metrics"]["records_processed"] == 50
