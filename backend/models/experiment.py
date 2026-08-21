# Experiment models.
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from backend.database.base import Base, BaseEntityMixin

class Experiment(Base, BaseEntityMixin):
    __tablename__ = "experiments"
    name = Column(String(150), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    dataset_version_id = Column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False)
    feature_set_id = Column(String(36), ForeignKey("feature_sets.id", ondelete="SET NULL"), nullable=True)
    runs = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan")

class ExperimentRun(Base, BaseEntityMixin):
    __tablename__ = "experiment_runs"
    experiment_id = Column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    run_name = Column(String(150), nullable=False)
    algorithm = Column(String(100), nullable=False, index=True)
    hyperparameters = Column(JSON, default=dict)
    training_mode = Column(String(30), default="STANDARD")
    status = Column(String(30), default="QUEUED", index=True)
    error_message = Column(Text, nullable=True)
    artifact_uri = Column(String(500), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    experiment = relationship("Experiment", back_populates="runs")
    metric = relationship("ExperimentMetric", back_populates="run", uselist=False, cascade="all, delete-orphan")

class ExperimentMetric(Base, BaseEntityMixin):
    __tablename__ = "experiment_metrics"
    run_id = Column(String(36), ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=False, unique=True)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)
    brier_score = Column(Float, nullable=False)
    confusion_matrix = Column(JSON, default=dict)
    training_time_seconds = Column(Float, default=0.0)
    inference_latency_ms = Column(Float, default=0.0)
    
    run = relationship("ExperimentRun", back_populates="metric")

class ExperimentParameter(Base, BaseEntityMixin):
    __tablename__ = "experiment_parameters"
    run_id = Column(String(36), ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    param_name = Column(String(100), nullable=False)
    param_value = Column(String(255), nullable=False)
