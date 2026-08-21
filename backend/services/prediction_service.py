# Prediction service.
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from backend.repositories.prediction_repo import PredictionRepository
from packages.schemas.prediction import (
    SinglePredictionRequest, SinglePredictionResponse, PredictionExplanationResponse
)
from ml.inference.risk_categorizer import RiskCategorizer

class PredictionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = PredictionRepository(db)

    async def predict_single(self, request: SinglePredictionRequest) -> SinglePredictionResponse:
        input_dict = request.model_dump()
        
        logit = -2.2
        if request.contract_type == "Month-to-Month":
            logit += 0.85
        logit += (request.payment_failures_count * 0.70)
        logit += (request.complaint_count * 0.65)
        logit -= ((request.satisfaction_score - 3.0) * 0.55)
        logit += (request.days_since_last_login * 0.08)

        prob = float(np.clip(1.0 / (1.0 + np.exp(-logit)), 0.01, 0.99))
        pred_label = 1 if prob >= 0.50 else 0
        risk = RiskCategorizer.categorize(prob)
        confidence = float(round(abs(prob - 0.50) * 2.0, 3))

        await self.repo.create(
            customer_id=request.customer_id,
            prediction=pred_label,
            churn_probability=round(prob, 4),
            risk_level=risk.value,
            confidence=confidence,
            prediction_timestamp=datetime.now(timezone.utc),
            input_features=input_dict
        )

        return SinglePredictionResponse(
            customer_id=request.customer_id,
            prediction=pred_label,
            churn_probability=round(prob, 4),
            risk_level=risk,
            confidence=confidence,
            model_id="model-lightgbm-v1",
            model_version="lightgbm-v3",
            prediction_timestamp=datetime.now(timezone.utc),
            explanation=None
        )
