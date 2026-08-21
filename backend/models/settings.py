# System setting model.
from sqlalchemy import Column, String, Text, JSON
from backend.database.base import Base, BaseEntityMixin

class SystemSetting(Base, BaseEntityMixin):
    __tablename__ = "system_settings"
    key = Column(String(100), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=False)
    value_type = Column(String(30), default="string")
    description = Column(Text, nullable=True)
