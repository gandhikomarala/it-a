"""Reporting engine configuration and export schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ReportGenerationRequest(BaseModel):
    title: str
    report_type: str = Field(..., description="MODEL_PERFORMANCE | CHURN_EXECUTIVE | DATA_QUALITY | DRIFT_AUDIT | EXPERIMENT_COMPARISON")
    format: str = Field("PDF", pattern="^(PDF|HTML|CSV|JSON)$")
    parameters: Dict[str, Any] = {}

class ReportResponse(BaseModel):
    id: str
    title: str
    report_type: str
    format: str
    download_url: str
    file_size_bytes: int
    generated_by: str
    created_at: datetime

    class Config:
        from_attributes = True

class ReportFilter(BaseModel):
    report_type: Optional[str] = None
    page: int = 1
    page_size: int = 20
