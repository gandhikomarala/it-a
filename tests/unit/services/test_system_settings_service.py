# Unit Test for SystemSettingsService.
import pytest
from backend.services.system_settings_service import SystemSettingsService

@pytest.mark.asyncio
async def test_system_settings_service_status():
    service = SystemSettingsService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "SystemSettingsService"

@pytest.mark.asyncio
async def test_system_settings_service_operation():
    service = SystemSettingsService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "system_settings_service_task"
