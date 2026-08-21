"""Dataset ingestion, validation, and profiling schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from packages.shared.enums import QualityScoreTier

class ColumnDistributionSchema(BaseModel):
    column_name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    median: Optional[float] = None
    quantiles: Optional[Dict[str, float]] = None
    top_categories: Optional[Dict[str, int]] = None
    outlier_count: int = 0

class DatasetProfileSchema(BaseModel):
    row_count: int
    column_count: int
    memory_usage_bytes: int
    columns: List[ColumnDistributionSchema]
    duplicate_rows_count: int
    duplicate_rows_percentage: float

class DataQualityReportSchema(BaseModel):
    quality_score: float = Field(..., ge=0.0, le=100.0)
    quality_tier: QualityScoreTier
    completeness_score: float
    validity_score: float
    uniqueness_score: float
    consistency_score: float
    issues_detected: List[Dict[str, Any]]
    recommendations: List[str]
    is_approved: bool

class DatasetValidationResult(BaseModel):
    is_valid: bool
    quality_report: DataQualityReportSchema
    profile: DatasetProfileSchema
    summary: str

class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=150)
    description: Optional[str] = None
    data_source_type: str = Field("FILE_UPLOAD", pattern="^(FILE_UPLOAD|S3_IMPORT|DATABASE_CONNECTOR|API)$")
    tags: List[str] = []

class DatasetVersionResponse(BaseModel):
    id: str
    dataset_id: str
    version_number: int
    file_path: str
    file_format: str
    file_size_bytes: int
    checksum_sha256: str
    row_count: int
    column_count: int
    quality_score: float
    is_approved: bool
    profile: Optional[DatasetProfileSchema] = None
    quality_report: Optional[DataQualityReportSchema] = None
    created_at: datetime

    class Config:
        from_attributes = True

class DatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    latest_version: int
    latest_quality_score: Optional[float] = None
    row_count: int
    column_count: int
    tags: List[str] = []
    versions_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class DatasetFilter(BaseModel):
    search: Optional[str] = None
    min_quality_score: Optional[float] = None
    is_approved: Optional[bool] = None
    page: int = 1
    page_size: int = 20

class DatasetListResponse(BaseModel):
    items: List[DatasetResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
