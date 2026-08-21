# Unit Test for DeploymentService.
import pytest
from backend.services.deployment_service import DeploymentService

@pytest.mark.asyncio
async def test_deployment_service_status():
    service = DeploymentService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "DeploymentService"

@pytest.mark.asyncio
async def test_deployment_service_operation():
    service = DeploymentService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "deployment_service_task"
