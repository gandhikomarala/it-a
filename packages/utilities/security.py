"""Security, password hashing, and JWT token management."""
import base64
import hashlib
import hmac
import os
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from packages.configuration.settings import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class PasswordHasher:
    """Secure password hashing using industry standard bcrypt."""

    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

class JWTManager:
    """JWT creation, signing, decoding, and validation."""

    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        settings = get_settings()
        to_encode = data.copy()
        now = datetime.now(timezone.utc)
        
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            
        to_encode.update({
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "jti": secrets.token_hex(16)
        })
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    @staticmethod
    def create_refresh_token(user_id: str, email: str) -> str:
        settings = get_settings()
        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode = {
            "sub": user_id,
            "email": email,
            "type": "refresh",
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "jti": secrets.token_hex(16)
        }
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        settings = get_settings()
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            return payload
        except JWTError as e:
            raise ValueError(f"Invalid or expired token: {str(e)}")

def generate_api_key() -> str:
    """Generate secure 40-character API key."""
    prefix = "chk_"
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}{random_part}"

def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographic alphanumeric token."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def mask_sensitive_data(val: str, show_last: int = 4) -> str:
    """Mask PII data, showing only trailing characters."""
    if not val or len(val) <= show_last:
        return "****"
    return f"{'*' * (len(val) - show_last)}{val[-show_last:]}"
