# Prediction models.
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from backend.database.base import Base, BaseEntityMixin

class Prediction(Base, BaseEntityMixin):
    __tablename__ = "predictions"
    customer_id = Column(String(64), nullable=False, index=True)
    prediction = Column(Integer, nullable=False)
    churn_probability = Column(Float, nullable=False, index=True)
    risk_level = Column(String(20), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    model_id = Column(String(36), ForeignKey("models.id", ondelete="SET NULL"), nullable=True)
    model_version_id = Column(String(36), ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True, index=True)
    prediction_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    input_features = Column(JSON, default=dict)
    
    explanation = relationship("PredictionExplanation", back_populates="prediction_record", uselist=False)

class PredictionExplanation(Base, BaseEntityMixin):
    __tablename__ = "prediction_explanations"
    prediction_id = Column(String(36), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, unique=True)
    base_value = Column(Float, nullable=False)
    top_positive_factors = Column(JSON, default=list)
    top_negative_factors = Column(JSON, default=list)
    all_contributions = Column(JSON, default=list)
    summary_text = Column(Text, nullable=False)
    
    prediction_record = relationship("Prediction", back_populates="explanation")

class PredictionBatch(Base, BaseEntityMixin):
    __tablename__ = "prediction_batches"
    model_version_id = Column(String(36), ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(30), default="QUEUED", index=True)
    total_records = Column(Integer, default=0)
    processed_records = Column(Integer, default=0)
    successful_records = Column(Integer, default=0)
    failed_records = Column(Integer, default=0)
    input_file_path = Column(String(500), nullable=True)
    output_file_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
