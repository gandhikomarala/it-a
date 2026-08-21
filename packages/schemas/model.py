"""Model registry, versioning, deployment, and lifecycle schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from packages.shared.enums import ModelStage
from .experiment import ExperimentMetricSchema

class ModelRegistrationSchema(BaseModel):
    name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = None
    experiment_run_id: str
    version_tag: Optional[str] = None
    stage: ModelStage = ModelStage.DEVELOPMENT
    metadata: Dict[str, Any] = {}

class ModelArtifactSchema(BaseModel):
    artifact_id: str
    model_version_id: str
    artifact_type: str
    file_path: str
    file_size_bytes: int
    checksum_sha256: str
    format: str

class ModelVersionResponse(BaseModel):
    id: str
    model_id: str
    version_number: int
    version_tag: str
    algorithm: str
    stage: ModelStage
    metrics: ExperimentMetricSchema
    hyperparameters: Dict[str, Any]
    dataset_version_id: str
    feature_set_id: Optional[str] = None
    artifact_path: str
    checksum_sha256: str
    is_active_production: bool = False
    created_by: str
    created_at: datetime

    class Config:
        from_attributes = True

class ModelResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    active_production_version: Optional[int] = None
    active_production_version_id: Optional[str] = None
    production_roc_auc: Optional[float] = None
    versions_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ModelDeploymentCreate(BaseModel):
    model_version_id: str
    target_stage: ModelStage = ModelStage.PRODUCTION
    deployment_notes: Optional[str] = None

class ModelDeploymentResponse(BaseModel):
    id: str
    model_version_id: str
    stage: ModelStage
    deployed_by: str
    deployment_notes: Optional[str] = None
    deployed_at: datetime
    is_current: bool

class ModelPromotionRequest(BaseModel):
    model_version_id: str
    target_stage: ModelStage
    rationale: str

class ModelRollbackRequest(BaseModel):
    model_id: str
    target_version_number: int
    reason: str

class ModelMetricsComparison(BaseModel):
    current_production_version: int
    current_metrics: ExperimentMetricSchema
    candidate_version: int
    candidate_metrics: ExperimentMetricSchema
    relative_improvements: Dict[str, float]
    safeguard_passed: bool
    safeguard_details: Dict[str, Any]
