# User, Role, Permission, Organization, APIKey, RefreshToken models.
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from backend.database.base import Base, BaseEntityMixin

class Organization(Base, BaseEntityMixin):
    __tablename__ = "organizations"
    name = Column(String(150), nullable=False, unique=True)
    slug = Column(String(150), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

class Permission(Base, BaseEntityMixin):
    __tablename__ = "permissions"
    code = Column(String(100), nullable=False, unique=True, index=True)
    name = Column(String(150), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=False, index=True)

class Role(Base, BaseEntityMixin):
    __tablename__ = "roles"
    name = Column(String(50), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    permissions = relationship("Permission", secondary="role_permissions", lazy="joined")

class RolePermission(Base):
    __tablename__ = "role_permissions"
    role_id = Column(String(36), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)
    permission_id = Column(String(36), ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True)

class User(Base, BaseEntityMixin):
    __tablename__ = "users"
    email = Column(String(255), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role_name = Column(String(50), nullable=False, default="ANALYST", index=True)
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)

class UserRoleMapping(Base):
    __tablename__ = "user_roles"
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role_id = Column(String(36), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)

class APIKey(Base, BaseEntityMixin):
    __tablename__ = "api_keys"
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    key_prefix = Column(String(10), nullable=False, index=True)
    hashed_key = Column(String(255), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, index=True)

class RefreshToken(Base, BaseEntityMixin):
    __tablename__ = "refresh_tokens"
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_jti = Column(String(64), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_revoked = Column(Boolean, default=False, index=True)
