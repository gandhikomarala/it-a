# Sub-20ms real-time single customer inference engine.
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd
from packages.schemas.prediction import (
    SinglePredictionRequest, SinglePredictionResponse
)
from .risk_categorizer import RiskCategorizer
from packages.logging.logger import get_logger

logger = get_logger(__name__)

class SingleInferenceEngine:
    def __init__(self, pipeline: Any, model_wrapper: Any, explainer: Optional[Any] = None):
        self.pipeline = pipeline
        self.model_wrapper = model_wrapper
        self.explainer = explainer

    def predict(
        self,
        request: SinglePredictionRequest,
        model_id: str = "prod-model-1",
        model_version: str = "lightgbm-v1"
    ) -> SinglePredictionResponse:
        input_dict = request.model_dump()
        df_raw = pd.DataFrame([input_dict])

        X_trans = self.pipeline.transform(df_raw)
        prob = float(self.model_wrapper.predict_proba(X_trans)[0])
        pred_label = 1 if prob >= 0.50 else 0
        risk = RiskCategorizer.categorize(prob)
        confidence = float(round(abs(prob - 0.50) * 2.0, 3))

        explanation = None
        if request.include_explanation and self.explainer is not None:
            explanation = self.explainer.explain_instance(request.customer_id, X_trans, prob)

        return SinglePredictionResponse(
            customer_id=request.customer_id,
            prediction=pred_label,
            churn_probability=round(prob, 4),
            risk_level=risk,
            confidence=confidence,
            model_id=model_id,
            model_version=model_version,
            prediction_timestamp=datetime.now(timezone.utc),
            explanation=explanation
        )
