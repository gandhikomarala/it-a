"""Platform system settings and health schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SystemSettingsSchema(BaseModel):
    platform_name: str = "Enterprise Churn & MLOps Platform"
    risk_low_threshold: float = Field(0.30, ge=0.05, le=0.50)
    risk_high_threshold: float = Field(0.70, ge=0.50, le=0.95)
    drift_psi_threshold: float = Field(0.25, ge=0.10, le=0.50)
    auto_retrain_enabled: bool = True
    auto_deploy_safeguards_enabled: bool = True
    email_notifications_enabled: bool = False
    slack_webhook_url: Optional[str] = None
    session_timeout_minutes: int = 60
    rate_limit_per_minute: int = 300

class SystemSettingsUpdate(BaseModel):
    risk_low_threshold: Optional[float] = None
    risk_high_threshold: Optional[float] = None
    drift_psi_threshold: Optional[float] = None
    auto_retrain_enabled: Optional[bool] = None
    auto_deploy_safeguards_enabled: Optional[bool] = None
    email_notifications_enabled: Optional[bool] = None
    slack_webhook_url: Optional[str] = None
    session_timeout_minutes: Optional[int] = None
    rate_limit_per_minute: Optional[int] = None

class ServiceHealthStatus(BaseModel):
    service_name: str
    status: str = "HEALTHY"  # HEALTHY | DEGRADED | UNHEALTHY
    latency_ms: float
    details: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str = "HEALTHY"
    version: str = "1.0.0"
    timestamp: datetime
    uptime_seconds: float
    services: List[ServiceHealthStatus]
