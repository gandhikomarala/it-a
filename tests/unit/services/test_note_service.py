# Unit Test for CustomerNoteService.
import pytest
from backend.services.note_service import CustomerNoteService

@pytest.mark.asyncio
async def test_note_service_status():
    service = CustomerNoteService(db=None)
    status = await service.get_service_status()
    assert status["status"] == "OPERATIONAL"
    assert status["service_name"] == "CustomerNoteService"

@pytest.mark.asyncio
async def test_note_service_operation():
    service = CustomerNoteService(db=None)
    res = await service.execute_operation({"test_key": "test_value"})
    assert res["status"] == "SUCCESS"
    assert res["operation"] == "note_service_task"
