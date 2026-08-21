# Unit Test for AuditService.
import pytest
from backend.services.audit_service import AuditService

@pytest.mark.asyncio
async def test_audit_service_status():
    service = AuditService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "AuditService"

@pytest.mark.asyncio
async def test_audit_service_operation():
    service = AuditService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "audit_service_task"
