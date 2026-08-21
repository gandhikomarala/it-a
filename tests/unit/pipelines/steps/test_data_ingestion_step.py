# Unit Test for DataIngestionStep.
import pytest
from ml.pipelines.steps.data_ingestion_step import DataIngestionStep

def test_data_ingestion_step_execution():
    step = DataIngestionStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "data_ingestion_step"
    assert res["metrics"]["records_processed"] == 50
