# Authentication service.
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from backend.repositories.user_repo import UserRepository
from backend.repositories.audit_repo import AuditRepository
from packages.schemas.auth import UserLogin, UserRegister, Token, UserResponse
from packages.utilities.security import PasswordHasher, JWTManager
from packages.shared.constants import ROLE_PERMISSIONS_MAP
from backend.core.exceptions import UnauthorizedException, ConflictException

class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_repo = UserRepository(db)
        self.audit_repo = AuditRepository(db)

    async def register(self, payload: UserRegister) -> UserResponse:
        existing = await self.user_repo.get_by_email(payload.email)
        if existing:
            raise ConflictException(f"User with email '{payload.email}' already exists.")

        hashed_pw = PasswordHasher.hash_password(payload.password)
        user = await self.user_repo.create(
            email=payload.email.lower().strip(),
            hashed_password=hashed_pw,
            first_name=payload.first_name,
            last_name=payload.last_name,
            role_name=payload.role.value,
            is_active=True,
            is_verified=True
        )

        await self.audit_repo.log_action(
            actor_id=user.id,
            actor_email=user.email,
            action="CREATE_USER",
            resource_type="USER",
            resource_id=user.id,
            status="SUCCESS"
        )

        permissions = ROLE_PERMISSIONS_MAP.get(user.role_name, [])
        return UserResponse(
            id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            role=user.role_name,
            permissions=permissions,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            updated_at=user.updated_at
        )

    async def login(self, payload: UserLogin) -> Token:
        user = await self.user_repo.get_by_email(payload.email)
        if not user or not PasswordHasher.verify_password(payload.password, user.hashed_password):
            raise UnauthorizedException("Invalid email or password")

        if not user.is_active:
            raise UnauthorizedException("Account is deactivated")

        user.last_login_at = datetime.now(timezone.utc)
        user.failed_login_attempts = 0
        await self.db.flush()

        permissions = ROLE_PERMISSIONS_MAP.get(user.role_name, [])
        token_data = {
            "sub": user.id,
            "email": user.email,
            "role": user.role_name,
            "permissions": permissions
        }

        access_token = JWTManager.create_access_token(token_data)
        refresh_token = JWTManager.create_refresh_token(user.id, user.email)

        await self.audit_repo.log_action(
            actor_id=user.id,
            actor_email=user.email,
            action="LOGIN",
            resource_type="AUTH",
            resource_id=user.id,
            status="SUCCESS"
        )

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=3600
        )
