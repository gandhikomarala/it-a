# DriftRepository entity repository.
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.monitoring import DriftReport
from .base import BaseRepository

class DriftRepository(BaseRepository[DriftReport]):
    def __init__(self, session: AsyncSession):
        super().__init__(DriftReport, session)
