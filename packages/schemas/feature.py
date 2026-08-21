"""Feature store, extraction, and definition schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class FeatureDefinitionSchema(BaseModel):
    name: str
    data_type: str
    description: str
    transformation_logic: str
    category: str = Field(..., description="E.g., Usage, Billing, Support, Engagement, Demographics")
    is_target: bool = False
    importance_score: Optional[float] = None

class FeatureSetCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = None
    dataset_version_id: str
    feature_names: List[str]
    target_column: str = "churn"

class FeatureSetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    version: int
    dataset_version_id: str
    feature_count: int
    target_column: str
    features: List[FeatureDefinitionSchema]
    created_at: datetime

    class Config:
        from_attributes = True

class FeatureImportanceSchema(BaseModel):
    feature_name: str
    importance: float
    normalized_importance: float
    rank: int

class FeatureCorrelationMatrix(BaseModel):
    features: List[str]
    matrix: List[List[float]]

class FeatureStoreQuery(BaseModel):
    customer_ids: List[str]
    feature_names: Optional[List[str]] = None
    as_of_timestamp: Optional[datetime] = None
