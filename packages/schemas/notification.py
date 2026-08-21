"""Notification message and subscription schemas."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from packages.shared.enums import NotificationType, NotificationChannel

class NotificationCreate(BaseModel):
    user_id: Optional[str] = None  # None = Broadcast to all authorized users
    title: str
    message: str
    notification_type: NotificationType
    channel: NotificationChannel = NotificationChannel.IN_APP
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

class NotificationResponse(BaseModel):
    id: str
    user_id: Optional[str]
    title: str
    message: str
    notification_type: NotificationType
    channel: NotificationChannel
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True

class NotificationPreferenceUpdate(BaseModel):
    email_enabled: bool = True
    slack_webhook_url: Optional[str] = None
    alert_on_drift: bool = True
    alert_on_training_failure: bool = True
    alert_on_deployment: bool = True
