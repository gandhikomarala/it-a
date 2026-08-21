# Unit Test for OrganizationService.
import pytest
from backend.services.organization_service import OrganizationService

@pytest.mark.asyncio
async def test_organization_service_status():
    service = OrganizationService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "OrganizationService"

@pytest.mark.asyncio
async def test_organization_service_operation():
    service = OrganizationService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "organization_service_task"
