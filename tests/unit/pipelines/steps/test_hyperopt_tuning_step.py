# Unit Test for BayesianHyperoptStep.
import pytest
from ml.pipelines.steps.hyperopt_tuning_step import BayesianHyperoptStep

def test_hyperopt_tuning_step_execution():
    step = BayesianHyperoptStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "hyperopt_tuning_step"
    assert res["metrics"]["records_processed"] == 50
