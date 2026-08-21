# Unit Test for CustomerEventService.
import pytest
from backend.services.event_service import CustomerEventService

@pytest.mark.asyncio
async def test_event_service_status():
    service = CustomerEventService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "CustomerEventService"

@pytest.mark.asyncio
async def test_event_service_operation():
    service = CustomerEventService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "event_service_task"
