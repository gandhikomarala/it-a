# Unit Test for SubscriptionService.
import pytest
from backend.services.subscription_service import SubscriptionService

@pytest.mark.asyncio
async def test_subscription_service_status():
    service = SubscriptionService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "SubscriptionService"

@pytest.mark.asyncio
async def test_subscription_service_operation():
    service = SubscriptionService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "subscription_service_task"
