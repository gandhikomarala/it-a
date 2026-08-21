# Dataset, DatasetVersion, DataSource, DataQualityReport models.
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from backend.database.base import Base, BaseEntityMixin

class DataSource(Base, BaseEntityMixin):
    __tablename__ = "data_sources"
    name = Column(String(100), nullable=False)
    source_type = Column(String(50), nullable=False)
    connection_config = Column(JSON, default=dict)

class Dataset(Base, BaseEntityMixin):
    __tablename__ = "datasets"
    name = Column(String(150), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    source_type = Column(String(50), default="FILE_UPLOAD")
    tags = Column(JSON, default=list)
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")

class DatasetVersion(Base, BaseEntityMixin):
    __tablename__ = "dataset_versions"
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    file_format = Column(String(20), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    checksum_sha256 = Column(String(64), nullable=False)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    quality_score = Column(Float, nullable=False, default=0.0)
    is_approved = Column(Boolean, default=False, index=True)
    profile_data = Column(JSON, default=dict)
    
    dataset = relationship("Dataset", back_populates="versions")
    quality_report = relationship("DataQualityReport", back_populates="version", uselist=False)

class DataQualityReport(Base, BaseEntityMixin):
    __tablename__ = "data_quality_reports"
    dataset_version_id = Column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, unique=True)
    quality_score = Column(Float, nullable=False)
    quality_tier = Column(String(30), nullable=False)
    completeness_score = Column(Float, nullable=False)
    validity_score = Column(Float, nullable=False)
    uniqueness_score = Column(Float, nullable=False)
    consistency_score = Column(Float, nullable=False)
    issues_detected = Column(JSON, default=list)
    recommendations = Column(JSON, default=list)
    is_approved = Column(Boolean, default=False)
    
    version = relationship("DatasetVersion", back_populates="quality_report")
