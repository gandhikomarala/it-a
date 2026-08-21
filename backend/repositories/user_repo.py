# User and authentication repository.
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.user import User, APIKey, RefreshToken
from .base import BaseRepository

class UserRepository(BaseRepository[User]):
    def __init__(self, session: AsyncSession):
        super().__init__(User, session)

    async def get_by_email(self, email: str) -> Optional[User]:
        stmt = select(User).where(User.email == email.lower().strip(), User.is_deleted.is_(False))
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_api_key(self, prefix: str) -> Optional[APIKey]:
        stmt = select(APIKey).where(APIKey.key_prefix == prefix, APIKey.is_active.is_(True))
        res = await self.session.execute(stmt)
        return res.scalars().first()
