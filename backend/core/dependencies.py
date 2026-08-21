# FastAPI Request Dependencies.
from typing import List, Callable, Optional
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.session import get_db
from backend.repositories.user_repo import UserRepository
from packages.utilities.security import JWTManager
from packages.shared.constants import ROLE_PERMISSIONS_MAP
from packages.schemas.auth import UserResponse
from packages.logging.logger import set_request_context
from .exceptions import UnauthorizedException, ForbiddenException

security_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    if not credentials or not credentials.credentials:
        raise UnauthorizedException("Missing authentication token")

    token = credentials.credentials
    try:
        payload = JWTManager.decode_token(token)
    except Exception as e:
        raise UnauthorizedException(f"Invalid access token: {str(e)}")

    user_id = payload.get("sub")
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    if not user or not user.is_active:
        raise UnauthorizedException("User account is inactive or deleted")

    permissions = ROLE_PERMISSIONS_MAP.get(user.role_name, [])
    set_request_context(request_id="api-req", user_id=user.id)

    return UserResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        role=user.role_name,
        permissions=permissions,
        is_active=user.is_active,
        is_verified=user.is_verified,
        last_login_at=user.last_login_at,
        created_at=user.created_at,
        updated_at=user.updated_at
    )

async def get_current_active_user(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    if not current_user.is_active:
        raise UnauthorizedException("User account is inactive")
    return current_user

def require_permissions(required_perms: List[str]) -> Callable:
    async def permission_checker(current_user: UserResponse = Depends(get_current_user)):
        user_perms = set(current_user.permissions)
        for perm in required_perms:
            if perm not in user_perms:
                raise ForbiddenException(f"Required permission '{perm}' is missing.")
        return current_user
    return permission_checker
