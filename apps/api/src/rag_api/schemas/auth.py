"""认证请求/响应实体与映射。

契约形状来自 `rag_contracts.api.auth`；本模块只做 re-export 与
domain → 响应实体的映射。
"""

from __future__ import annotations

import uuid

from rag_api.services.auth import AuthenticatedUser
from rag_contracts.api.auth import (
    CurrentUserResponse,
    DevLoginRequest,
    LoginResponse,
    RefreshRequest,
    RefreshResponse,
)
from rag_db.models.membership import TenantMembership
from rag_db.models.user import User


def to_login_response(
    resolved: AuthenticatedUser,
    *,
    access_token: str,
    refresh_token: str,
    expires_at: int,
) -> LoginResponse:
    """登录结果 → 响应实体。"""
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        user_id=resolved.user_id,
        tenant_id=resolved.tenant_id,
        email=resolved.email,
        display_name=resolved.display_name,
        is_new_user=resolved.created,
    )


def to_refresh_response(
    *, access_token: str, refresh_token: str, expires_at: int
) -> RefreshResponse:
    """刷新结果 → 响应实体。"""
    return RefreshResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
    )


def to_current_user_response(
    user: User, *, tenant_id: uuid.UUID, membership: TenantMembership | None
) -> CurrentUserResponse:
    """当前用户 → 响应实体。"""
    return CurrentUserResponse(
        user_id=user.id,
        tenant_id=tenant_id,
        email=user.email,
        display_name=user.display_name,
        status=str(user.status),
        platform_role=str(user.platform_role),
        tenant_role=str(membership.role) if membership is not None else None,
    )


__all__ = [
    "CurrentUserResponse",
    "DevLoginRequest",
    "LoginResponse",
    "RefreshRequest",
    "RefreshResponse",
    "to_current_user_response",
    "to_login_response",
    "to_refresh_response",
]
