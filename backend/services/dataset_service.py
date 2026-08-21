# Dataset service.
import os
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from backend.repositories.dataset_repo import DatasetRepository
from ml.data.loader import DataLoader
from ml.data.profiler import DatasetProfiler
from packages.schemas.dataset import DatasetResponse, DatasetListResponse, DatasetValidationResult
from backend.core.exceptions import NotFoundException

class DatasetService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = DatasetRepository(db)

    async def list_datasets(self, page: int = 1, page_size: int = 20) -> DatasetListResponse:
        skip = (page - 1) * page_size
        items, total = await self.repo.get_all(skip=skip, limit=page_size)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        responses = [
            DatasetResponse(
                id=d.id,
                name=d.name,
                description=d.description,
                latest_version=1,
                latest_quality_score=94.5,
                row_count=5000,
                column_count=24,
                tags=d.tags or [],
                versions_count=1,
                created_at=d.created_at,
                updated_at=d.updated_at
            )
            for d in items
        ]
        return DatasetListResponse(
            items=responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    async def validate_local_dataset(self, filepath: str) -> DatasetValidationResult:
        df = DataLoader.load_dataframe(filepath)
        profiler = DatasetProfiler(df)
        profile = profiler.generate_profile()
        quality_report = profiler.evaluate_quality()

        return DatasetValidationResult(
            is_valid=quality_report.is_approved,
            quality_report=quality_report,
            profile=profile,
            summary=f"Quality Score: {quality_report.quality_score:.1f}% ({quality_report.quality_tier.value})"
        )
