"""Automated retraining policies, triggers, and validation schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from packages.shared.enums import RetrainingTriggerType, TaskStatus

class RetrainingPolicySchema(BaseModel):
    id: Optional[str] = None
    name: str = "Standard Retraining Policy"
    is_auto_retrain_enabled: bool = True
    is_auto_deploy_enabled: bool = False  # Strict safeguard: require approval by default
    drift_threshold_psi: float = Field(0.25, ge=0.10, le=0.50)
    performance_drop_threshold: float = Field(0.05, ge=0.01, le=0.20)
    cron_schedule: Optional[str] = "0 0 * * 0"  # Every Sunday midnight
    min_new_samples_required: int = 1000

class RetrainingTriggerRequest(BaseModel):
    trigger_type: RetrainingTriggerType
    dataset_version_id: Optional[str] = None
    policy_id: Optional[str] = None
    rationale: str

class DeploymentSafeguardCheck(BaseModel):
    check_name: str
    passed: bool
    threshold: float
    actual_value: float
    details: str

class CandidateEvaluationResult(BaseModel):
    candidate_run_id: str
    candidate_roc_auc: float
    production_roc_auc: float
    delta_roc_auc: float
    safeguard_checks: List[DeploymentSafeguardCheck]
    all_checks_passed: bool
    recommended_action: str = Field(..., description="PROMOTE | MANUAL_REVIEW | REJECT")

class RetrainingJobResponse(BaseModel):
    job_id: str
    trigger_type: RetrainingTriggerType
    status: TaskStatus
    candidate_model_version_id: Optional[str] = None
    evaluation_result: Optional[CandidateEvaluationResult] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
