# Unit Test for NotificationService.
import pytest
from backend.services.notification_service import NotificationService

@pytest.mark.asyncio
async def test_notification_service_status():
    service = NotificationService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "NotificationService"

@pytest.mark.asyncio
async def test_notification_service_operation():
    service = NotificationService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "notification_service_task"
