# Audit log repository.
from typing import List, Optional, Tuple
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.audit import AuditLog
from .base import BaseRepository

class AuditRepository(BaseRepository[AuditLog]):
    def __init__(self, session: AsyncSession):
        super().__init__(AuditLog, session)

    async def log_action(self, **kwargs) -> AuditLog:
        return await self.create(**kwargs)
