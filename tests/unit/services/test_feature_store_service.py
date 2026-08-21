# Unit Test for FeatureStoreService.
import pytest
from backend.services.feature_store_service import FeatureStoreService

@pytest.mark.asyncio
async def test_feature_store_service_status():
    service = FeatureStoreService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "FeatureStoreService"

@pytest.mark.asyncio
async def test_feature_store_service_operation():
    service = FeatureStoreService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "feature_store_service_task"
