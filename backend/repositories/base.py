# Generic Async CRUD Repository with pagination, sorting, and soft deletion.
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar
from sqlalchemy import func, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.base import Base

ModelType = TypeVar("ModelType", bound=Base)

class BaseRepository(Generic[ModelType]):
    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session

    async def get_by_id(self, id: str) -> Optional[ModelType]:
        stmt = select(self.model).where(self.model.id == id, self.model.is_deleted.is_(False))
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = "created_at",
        ascending: bool = False
    ) -> Tuple[List[ModelType], int]:
        count_stmt = select(func.count(self.model.id)).where(self.model.is_deleted.is_(False))
        total_res = await self.session.execute(count_stmt)
        total = total_res.scalar() or 0

        stmt = select(self.model).where(self.model.is_deleted.is_(False))
        if hasattr(self.model, order_by):
            col = getattr(self.model, order_by)
            stmt = stmt.order_by(col.asc() if ascending else col.desc())
        
        stmt = stmt.offset(skip).limit(limit)
        res = await self.session.execute(stmt)
        items = res.scalars().all()
        return list(items), total

    async def create(self, **kwargs) -> ModelType:
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def update(self, id: str, **kwargs) -> Optional[ModelType]:
        instance = await self.get_by_id(id)
        if not instance:
            return None
        for key, value in kwargs.items():
            if hasattr(instance, key) and value is not None:
                setattr(instance, key, value)
        await self.session.flush()
        return instance

    async def soft_delete(self, id: str) -> bool:
        from datetime import datetime, timezone
        instance = await self.get_by_id(id)
        if not instance:
            return False
        instance.is_deleted = True
        instance.deleted_at = datetime.now(timezone.utc)
        await self.session.flush()
        return True
