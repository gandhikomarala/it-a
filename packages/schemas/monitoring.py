"""Model and data drift monitoring schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from packages.shared.enums import DriftStatus

class FeatureDriftMetric(BaseModel):
    feature_name: str
    data_type: str
    psi_value: float
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    chi_square_statistic: Optional[float] = None
    chi_square_p_value: Optional[float] = None
    drift_status: DriftStatus
    baseline_mean: Optional[float] = None
    current_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    current_std: Optional[float] = None
    missing_rate_baseline: float
    missing_rate_current: float

class PredictionDistributionMetrics(BaseModel):
    drift_status: DriftStatus
    psi_value: float
    baseline_mean_prob: float
    current_mean_prob: float
    baseline_high_risk_pct: float
    current_high_risk_pct: float

class DriftReportSchema(BaseModel):
    id: str
    model_version_id: str
    overall_drift_status: DriftStatus
    max_psi: float
    features_drifted_count: int
    total_features_monitored: int
    feature_metrics: List[FeatureDriftMetric]
    prediction_distribution: PredictionDistributionMetrics
    sample_size: int
    created_at: datetime

class ModelPerformanceSnapshot(BaseModel):
    id: str
    model_version_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    evaluated_samples_count: int
    window_start: datetime
    window_end: datetime
    created_at: datetime

class MonitoringDashboardSummary(BaseModel):
    model_id: str
    model_name: str
    production_version: int
    drift_status: DriftStatus
    last_drift_check: datetime
    avg_latency_ms: float
    requests_last_24h: int
    prediction_errors_last_24h: int
    current_roc_auc: float
    drift_trend: List[Dict[str, Any]]
    performance_history: List[ModelPerformanceSnapshot]
