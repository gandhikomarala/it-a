# Unit Test for APIKeyService.
import pytest
from backend.services.api_key_service import APIKeyService

@pytest.mark.asyncio
async def test_api_key_service_status():
    service = APIKeyService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "APIKeyService"

@pytest.mark.asyncio
async def test_api_key_service_operation():
    service = APIKeyService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "api_key_service_task"
