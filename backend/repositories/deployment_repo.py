# DeploymentRepository entity repository.
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.model import ModelDeployment
from .base import BaseRepository

class DeploymentRepository(BaseRepository[ModelDeployment]):
    def __init__(self, session: AsyncSession):
        super().__init__(ModelDeployment, session)
