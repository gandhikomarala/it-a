# ExperimentRunRepository entity repository.
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.experiment import ExperimentRun
from .base import BaseRepository

class ExperimentRunRepository(BaseRepository[ExperimentRun]):
    def __init__(self, session: AsyncSession):
        super().__init__(ExperimentRun, session)
