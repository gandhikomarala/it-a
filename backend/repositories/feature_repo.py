# FeatureRepository entity repository.
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.feature import FeatureDefinition
from .base import BaseRepository

class FeatureRepository(BaseRepository[FeatureDefinition]):
    def __init__(self, session: AsyncSession):
        super().__init__(FeatureDefinition, session)
