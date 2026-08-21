"""Authentication, Authorization, User, and Role schemas."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field
from packages.shared.enums import UserRole

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenPayload(BaseModel):
    sub: str
    email: str
    role: str
    permissions: List[str] = []
    exp: int
    iat: int
    jti: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    organization_name: Optional[str] = "Default Organization"
    role: UserRole = UserRole.ANALYST

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str
    last_name: str
    role: UserRole
    is_active: bool = True
    organization_id: Optional[str] = None

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8)

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    first_name: str
    last_name: str
    role: UserRole
    permissions: List[str]
    is_active: bool
    is_verified: bool
    last_login_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)

class PermissionResponse(BaseModel):
    id: str
    code: str
    name: str
    description: Optional[str] = None
    category: str

class RoleResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    permissions: List[PermissionResponse]

class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=100)
    expires_in_days: Optional[int] = 90

class APIKeyResponse(BaseModel):
    id: str
    name: str
    prefix: str
    key: Optional[str] = None  # Returned once upon creation
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool
