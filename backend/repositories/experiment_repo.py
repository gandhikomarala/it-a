# ExperimentRepository entity repository.
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.experiment import Experiment
from .base import BaseRepository

class ExperimentRepository(BaseRepository[Experiment]):
    def __init__(self, session: AsyncSession):
        super().__init__(Experiment, session)
