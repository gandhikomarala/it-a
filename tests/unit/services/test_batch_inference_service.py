# Unit Test for BatchInferenceService.
import pytest
from backend.services.batch_inference_service import BatchInferenceService

@pytest.mark.asyncio
async def test_batch_inference_service_status():
    service = BatchInferenceService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "BatchInferenceService"

@pytest.mark.asyncio
async def test_batch_inference_service_operation():
    service = BatchInferenceService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "batch_inference_service_task"
