"""ML experiment tracking, runs, and comparison schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from packages.shared.enums import TaskStatus

class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=150)
    description: Optional[str] = None
    dataset_version_id: str
    feature_set_id: Optional[str] = None
    tags: List[str] = []

class ExperimentMetricSchema(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    confusion_matrix: Dict[str, int]
    training_time_seconds: float
    inference_latency_ms: float

class ExperimentRunCreate(BaseModel):
    experiment_id: str
    run_name: str
    algorithm: str = Field(..., description="LogisticRegression | RandomForest | GradientBoosting | LightGBM | Ensemble")
    hyperparameters: Dict[str, Any]
    training_mode: str = Field("STANDARD", pattern="^(FAST|STANDARD|FULL)$")

class ExperimentRunResponse(BaseModel):
    id: str
    experiment_id: str
    run_name: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    metrics: Optional[ExperimentMetricSchema] = None
    status: TaskStatus
    error_message: Optional[str] = None
    artifact_uri: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    dataset_version_id: str
    runs_count: int
    best_roc_auc: Optional[float] = None
    best_algorithm: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ExperimentComparisonSchema(BaseModel):
    experiment_id: str
    runs: List[ExperimentRunResponse]
    metric_comparison: Dict[str, Dict[str, float]]
    radar_chart_data: List[Dict[str, Any]]
    recommendation: str
