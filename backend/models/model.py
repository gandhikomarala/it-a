# MLModel and Deployments models.
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from backend.database.base import Base, BaseEntityMixin

class MLModel(Base, BaseEntityMixin):
    __tablename__ = "models"
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")

class ModelVersion(Base, BaseEntityMixin):
    __tablename__ = "model_versions"
    model_id = Column(String(36), ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    version_tag = Column(String(50), nullable=False, index=True)
    algorithm = Column(String(100), nullable=False)
    stage = Column(String(50), default="DEVELOPMENT", index=True)
    hyperparameters = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    dataset_version_id = Column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False)
    feature_set_id = Column(String(36), ForeignKey("feature_sets.id", ondelete="SET NULL"), nullable=True)
    artifact_path = Column(String(500), nullable=False)
    checksum_sha256 = Column(String(64), nullable=False)
    is_active_production = Column(Boolean, default=False, index=True)
    created_by = Column(String(100), default="system")
    
    model = relationship("MLModel", back_populates="versions")
    deployments = relationship("ModelDeployment", back_populates="version")

class ModelMetric(Base, BaseEntityMixin):
    __tablename__ = "model_metrics"
    model_version_id = Column(String(36), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)

class ModelArtifact(Base, BaseEntityMixin):
    __tablename__ = "model_artifacts"
    model_version_id = Column(String(36), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    artifact_type = Column(String(50), default="MODEL_BUNDLE")
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    checksum_sha256 = Column(String(64), nullable=False)
    format = Column(String(20), default="joblib")

class ModelDeployment(Base, BaseEntityMixin):
    __tablename__ = "model_deployments"
    model_version_id = Column(String(36), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    stage = Column(String(50), nullable=False, index=True)
    deployed_by = Column(String(100), nullable=False)
    deployment_notes = Column(Text, nullable=True)
    deployed_at = Column(DateTime(timezone=True), nullable=False)
    is_current = Column(Boolean, default=True, index=True)
    
    version = relationship("ModelVersion", back_populates="deployments")

class DeploymentHistory(Base, BaseEntityMixin):
    __tablename__ = "deployment_history"
    model_id = Column(String(36), ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True)
    from_version = Column(Integer, nullable=True)
    to_version = Column(Integer, nullable=False)
    action = Column(String(50), nullable=False)
    reason = Column(Text, nullable=True)
    actor = Column(String(100), nullable=False)
