# Monitoring & Drift models.
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from backend.database.base import Base, BaseEntityMixin

class DriftReport(Base, BaseEntityMixin):
    __tablename__ = "drift_reports"
    model_version_id = Column(String(36), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    overall_drift_status = Column(String(30), nullable=False, index=True)
    max_psi = Column(Float, nullable=False)
    features_drifted_count = Column(Integer, default=0)
    total_features_monitored = Column(Integer, default=0)
    feature_metrics = Column(JSON, default=list)
    prediction_distribution = Column(JSON, default=dict)
    sample_size = Column(Integer, default=0)

class FeatureDriftMetricRecord(Base, BaseEntityMixin):
    __tablename__ = "feature_drift_metrics"
    drift_report_id = Column(String(36), ForeignKey("drift_reports.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_name = Column(String(100), nullable=False)
    psi_value = Column(Float, nullable=False)
    ks_statistic = Column(Float, nullable=True)
    ks_p_value = Column(Float, nullable=True)
    drift_status = Column(String(30), nullable=False)

class MonitoringMetric(Base, BaseEntityMixin):
    __tablename__ = "monitoring_metrics"
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    labels = Column(JSON, default=dict)
    recorded_at = Column(DateTime(timezone=True), nullable=False, index=True)

class ModelPerformanceSnapshot(Base, BaseEntityMixin):
    __tablename__ = "model_performance_snapshots"
    model_version_id = Column(String(36), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    evaluated_samples_count = Column(Integer, default=0)
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
