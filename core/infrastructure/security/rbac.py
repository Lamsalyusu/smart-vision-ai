from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt_auth import TokenPayload, jwt_service

security_bearer = HTTPBearer(auto_error=False)
bearer_credentials = Security(security_bearer)


class AppRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"


class AppPermission(str, Enum):
    DETECT = "detect"
    VIEW_TELEMETRY = "view_telemetry"
    MANAGE_ZONES = "manage_zones"
    EXPORT_DATA = "export_data"
    ADMIN_USERS = "admin_users"


ROLE_PERMISSIONS_MAP: dict[AppRole, set[AppPermission]] = {
    AppRole.ADMIN: {
        AppPermission.DETECT,
        AppPermission.VIEW_TELEMETRY,
        AppPermission.MANAGE_ZONES,
        AppPermission.EXPORT_DATA,
        AppPermission.ADMIN_USERS,
    },
    AppRole.ANALYST: {
        AppPermission.DETECT,
        AppPermission.VIEW_TELEMETRY,
        AppPermission.MANAGE_ZONES,
        AppPermission.EXPORT_DATA,
    },
    AppRole.OPERATOR: {
        AppPermission.DETECT,
        AppPermission.VIEW_TELEMETRY,
        AppPermission.MANAGE_ZONES,
    },
    AppRole.VIEWER: {
        AppPermission.VIEW_TELEMETRY,
    },
}


def require_permission(required_perm: AppPermission) -> Callable[..., TokenPayload]:
    """FastAPI Dependency enforcing RBAC permission verification."""
    def dependency(credentials: HTTPAuthorizationCredentials | None = bearer_credentials) -> TokenPayload:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            payload = jwt_service.verify_token(credentials.credentials)
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(err),
                headers={"WWW-Authenticate": "Bearer"},
            ) from err

        role = AppRole(payload.role) if payload.role in [r.value for r in AppRole] else AppRole.VIEWER
        allowed_perms = ROLE_PERMISSIONS_MAP.get(role, set())
        if required_perm not in allowed_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role.value}' lacks required permission '{required_perm.value}'",
            )

        return payload

    return dependency
