# Unit Test for WebhookDeliveryService.
import pytest
from backend.services.webhook_delivery_service import WebhookDeliveryService

@pytest.mark.asyncio
async def test_webhook_delivery_service_status():
    service = WebhookDeliveryService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "WebhookDeliveryService"

@pytest.mark.asyncio
async def test_webhook_delivery_service_operation():
    service = WebhookDeliveryService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "webhook_delivery_service_task"
