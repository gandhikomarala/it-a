# SQLAlchemy Declarative Base with UUID, timestamps, and soft-delete mixins.
import uuid
from datetime import datetime, timezone
from typing import Any, Dict
from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.orm import declarative_base, declared_attr

Base = declarative_base()

class BaseEntityMixin:
    @declared_attr
    def id(cls):
        return Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)

    @declared_attr
    def created_at(cls):
        return Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    @declared_attr
    def updated_at(cls):
        return Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            onupdate=lambda: datetime.now(timezone.utc),
            nullable=False
        )

    @declared_attr
    def is_deleted(cls):
        return Column(Boolean, default=False, nullable=False, index=True)

    @declared_attr
    def deleted_at(cls):
        return Column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for c in self.__table__.columns:
            val = getattr(self, c.name)
            if isinstance(val, datetime):
                val = val.isoformat()
            result[c.name] = val
        return result
