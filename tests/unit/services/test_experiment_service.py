# Unit Test for ExperimentService.
import pytest
from backend.services.experiment_service import ExperimentService

@pytest.mark.asyncio
async def test_experiment_service_status():
    service = ExperimentService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "ExperimentService"

@pytest.mark.asyncio
async def test_experiment_service_operation():
    service = ExperimentService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "experiment_service_task"
