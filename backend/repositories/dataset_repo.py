# Dataset repository.
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.dataset import Dataset, DatasetVersion, DataQualityReport
from .base import BaseRepository

class DatasetRepository(BaseRepository[Dataset]):
    def __init__(self, session: AsyncSession):
        super().__init__(Dataset, session)

    async def get_by_name(self, name: str) -> Optional[Dataset]:
        stmt = select(Dataset).where(Dataset.name == name, Dataset.is_deleted.is_(False))
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_version(self, dataset_id: str, version_num: int) -> Optional[DatasetVersion]:
        stmt = select(DatasetVersion).where(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.version_number == version_num
        )
        res = await self.session.execute(stmt)
        return res.scalars().first()
