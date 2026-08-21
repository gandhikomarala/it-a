# Unit Test for ExecutiveReportService.
import pytest
from backend.services.executive_report_service import ExecutiveReportService

@pytest.mark.asyncio
async def test_executive_report_service_status():
    service = ExecutiveReportService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "ExecutiveReportService"

@pytest.mark.asyncio
async def test_executive_report_service_operation():
    service = ExecutiveReportService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "executive_report_service_task"
