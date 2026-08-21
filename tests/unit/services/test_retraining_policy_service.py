# Unit Test for RetrainingPolicyService.
import pytest
from backend.services.retraining_policy_service import RetrainingPolicyService

@pytest.mark.asyncio
async def test_retraining_policy_service_status():
    service = RetrainingPolicyService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "RetrainingPolicyService"

@pytest.mark.asyncio
async def test_retraining_policy_service_operation():
    service = RetrainingPolicyService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "retraining_policy_service_task"
