# Unit Test for FairnessAuditService.
import pytest
from backend.services.fairness_audit_service import FairnessAuditService

@pytest.mark.asyncio
async def test_fairness_audit_service_status():
    service = FairnessAuditService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "FairnessAuditService"

@pytest.mark.asyncio
async def test_fairness_audit_service_operation():
    service = FairnessAuditService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "fairness_audit_service_task"
