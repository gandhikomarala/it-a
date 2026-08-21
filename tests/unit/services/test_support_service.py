# Unit Test for SupportService.
import pytest
from backend.services.support_service import SupportService

@pytest.mark.asyncio
async def test_support_service_status():
    service = SupportService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "SupportService"

@pytest.mark.asyncio
async def test_support_service_operation():
    service = SupportService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "support_service_task"
