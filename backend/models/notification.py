# Notification models.
from sqlalchemy import Boolean, Column, ForeignKey, String, Text, JSON
from backend.database.base import Base, BaseEntityMixin

class Notification(Base, BaseEntityMixin):
    __tablename__ = "notifications"
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), nullable=False, index=True)
    channel = Column(String(30), default="IN_APP")
    is_read = Column(Boolean, default=False, index=True)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(50), nullable=True)
    metadata_json = Column(JSON, default=dict)

class NotificationTemplate(Base, BaseEntityMixin):
    __tablename__ = "notification_templates"
    template_key = Column(String(100), nullable=False, unique=True)
    subject_template = Column(String(255), nullable=False)
    body_template = Column(Text, nullable=False)

class NotificationPreference(Base, BaseEntityMixin):
    __tablename__ = "notification_preferences"
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    email_enabled = Column(Boolean, default=True)
    slack_webhook_url = Column(String(500), nullable=True)
    alert_on_drift = Column(Boolean, default=True)
    alert_on_training_failure = Column(Boolean, default=True)
    alert_on_deployment = Column(Boolean, default=True)
