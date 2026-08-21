# Unit Test for CRMSyncService.
import pytest
from backend.services.crm_sync_service import CRMSyncService

@pytest.mark.asyncio
async def test_crm_sync_service_status():
    service = CRMSyncService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "CRMSyncService"

@pytest.mark.asyncio
async def test_crm_sync_service_operation():
    service = CRMSyncService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "crm_sync_service_task"
