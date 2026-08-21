# Real-time inference and SHAP explainability router.
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.session import get_db
from backend.services.prediction_service import PredictionService
from packages.schemas.prediction import SinglePredictionRequest, SinglePredictionResponse
from backend.core.dependencies import require_permissions

router = APIRouter(prefix="/predictions", tags=["Predictions"])

@router.post("", response_model=SinglePredictionResponse)
async def predict_customer_churn(
    payload: SinglePredictionRequest,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_permissions(["prediction:create"]))
):
    service = PredictionService(db)
    return await service.predict_single(payload)
