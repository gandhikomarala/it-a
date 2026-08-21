"""Real-time and batch prediction, SHAP explanation schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from packages.shared.enums import RiskLevel, BatchStatus

class SHAPFactorContribution(BaseModel):
    feature_name: str
    display_name: str
    feature_value: Any
    shap_value: float
    contribution_percentage: float
    impact_direction: str = Field(..., description="POSITIVE (Increases churn) | NEGATIVE (Decreases churn)")

class PredictionExplanationResponse(BaseModel):
    customer_id: str
    base_value: float
    prediction_probability: float
    top_positive_factors: List[SHAPFactorContribution]
    top_negative_factors: List[SHAPFactorContribution]
    all_contributions: List[SHAPFactorContribution]
    summary_text: str

class SinglePredictionRequest(BaseModel):
    customer_id: str
    age: int
    gender: str
    region: str
    income: float
    subscription_type: str
    contract_type: str
    payment_method: str
    monthly_charge: float
    tenure_months: int
    total_spend: float
    daily_usage_hours: float
    monthly_usage_hours: float
    login_count_monthly: int
    payment_failures_count: int = 0
    late_payments_count: int = 0
    complaint_count: int = 0
    satisfaction_score: float = 3.5
    days_since_last_login: int = 1
    include_explanation: bool = True

class SinglePredictionResponse(BaseModel):
    customer_id: str
    prediction: int = Field(..., description="0 = Retain, 1 = Churn")
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_id: str
    model_version: str
    prediction_timestamp: datetime
    explanation: Optional[PredictionExplanationResponse] = None

class BatchPredictionRequest(BaseModel):
    dataset_version_id: Optional[str] = None
    file_path: Optional[str] = None
    model_version_id: Optional[str] = None
    output_format: str = Field("CSV", pattern="^(CSV|PARQUET|JSON)$")
    include_explanations: bool = False

class BatchJobStatusResponse(BaseModel):
    batch_id: str
    status: BatchStatus
    total_records: int
    processed_records: int
    successful_records: int
    failed_records: int
    progress_percentage: float
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output_file_url: Optional[str] = None
    error_message: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    batch_id: str
    message: str
    status: BatchStatus
    total_records: int
    estimated_duration_seconds: float

class RiskThresholdConfig(BaseModel):
    low_cutoff: float = Field(0.30, ge=0.05, le=0.50)
    high_cutoff: float = Field(0.70, ge=0.50, le=0.95)
