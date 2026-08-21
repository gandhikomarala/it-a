# Audit log model.
from sqlalchemy import Column, String, Text, JSON
from backend.database.base import Base, BaseEntityMixin

class AuditLog(Base, BaseEntityMixin):
    __tablename__ = "audit_logs"
    actor_id = Column(String(36), nullable=True, index=True)
    actor_email = Column(String(255), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False, index=True)
    resource_id = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(300), nullable=True)
    status = Column(String(20), default="SUCCESS", index=True)
    details = Column(JSON, default=dict)
