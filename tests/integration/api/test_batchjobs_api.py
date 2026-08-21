# API Integration Test for Batch Inference Job Scheduler Endpoint.
import pytest
from httpx import AsyncClient
from backend.main import app

@pytest.mark.asyncio
async def test_batchjobs_api_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
