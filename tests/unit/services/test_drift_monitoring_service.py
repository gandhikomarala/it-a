# Unit Test for DriftMonitoringService.
import pytest
from backend.services.drift_monitoring_service import DriftMonitoringService

@pytest.mark.asyncio
async def test_drift_monitoring_service_status():
    service = DriftMonitoringService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "DriftMonitoringService"

@pytest.mark.asyncio
async def test_drift_monitoring_service_operation():
    service = DriftMonitoringService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "drift_monitoring_service_task"
