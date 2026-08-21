# Unit Test for BillingService.
import pytest
from backend.services.billing_service import BillingService

@pytest.mark.asyncio
async def test_billing_service_status():
    service = BillingService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "BillingService"

@pytest.mark.asyncio
async def test_billing_service_operation():
    service = BillingService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "billing_service_task"
