# MonitoringMetricRepository entity repository.
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.monitoring import MonitoringMetric
from .base import BaseRepository

class MonitoringMetricRepository(BaseRepository[MonitoringMetric]):
    def __init__(self, session: AsyncSession):
        super().__init__(MonitoringMetric, session)
