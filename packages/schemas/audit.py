"""Audit logging and compliance tracking schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from packages.shared.enums import AuditAction

class AuditLogResponse(BaseModel):
    id: str
    actor_id: Optional[str]
    actor_email: Optional[str]
    action: AuditAction
    resource_type: str
    resource_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    status: str = "SUCCESS"  # SUCCESS | FAILURE
    details: Dict[str, Any] = {}
    created_at: datetime

    class Config:
        from_attributes = True

class AuditLogFilter(BaseModel):
    actor_email: Optional[str] = None
    action: Optional[AuditAction] = None
    resource_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    page: int = 1
    page_size: int = 25

class AuditLogListResponse(BaseModel):
    items: List[AuditLogResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
