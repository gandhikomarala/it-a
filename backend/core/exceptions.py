# Centralized application exception hierarchy.
from typing import Any, Dict, Optional
from fastapi import HTTPException, status

class AppException(HTTPException):
    def __init__(self, status_code: int, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status_code,
            detail={"code": code, "message": message, "details": details or {}}
        )

class NotFoundException(AppException):
    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(status.HTTP_404_NOT_FOUND, "ERR_NOT_FOUND", message, details)

class UnauthorizedException(AppException):
    def __init__(self, message: str = "Authentication required", details: Optional[Dict[str, Any]] = None):
        super().__init__(status.HTTP_401_UNAUTHORIZED, "ERR_UNAUTHORIZED", message, details)

class ForbiddenException(AppException):
    def __init__(self, message: str = "Permission denied", details: Optional[Dict[str, Any]] = None):
        super().__init__(status.HTTP_403_FORBIDDEN, "ERR_FORBIDDEN", message, details)

class ValidationException(AppException):
    def __init__(self, message: str = "Validation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(status.HTTP_422_UNPROCESSABLE_ENTITY, "ERR_VALIDATION", message, details)

class ConflictException(AppException):
    def __init__(self, message: str = "Resource conflict", details: Optional[Dict[str, Any]] = None):
        super().__init__(status.HTTP_409_CONFLICT, "ERR_CONFLICT", message, details)
