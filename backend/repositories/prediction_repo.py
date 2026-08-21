# Prediction repository.
from typing import Optional, List, Tuple
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.prediction import Prediction, PredictionBatch
from .base import BaseRepository

class PredictionRepository(BaseRepository[Prediction]):
    def __init__(self, session: AsyncSession):
        super().__init__(Prediction, session)

    async def get_recent_by_customer(self, customer_id: str) -> Optional[Prediction]:
        stmt = select(Prediction).where(
            Prediction.customer_id == customer_id
        ).order_by(Prediction.prediction_timestamp.desc())
        res = await self.session.execute(stmt)
        return res.scalars().first()
