"""认证路由：dev 登录 + 当前用户信息。

## 为什么用工厂函数而不是模块级 router

dev 登录端点是否注册取决于配置。如果 `router` 是模块级单例，
某个 app 注册过 dev 端点后，**同一进程里创建的其它 app 也会看到它** ——
症状是"关闭 dev 后端点仍然存在"。

这在测试里会直接暴露（关闭 dev 的测试拿到 404 失败），
但在生产里更隐蔽：一次热重载、一个多 app 进程，就可能让
不应存在的登录入口存活下来。

## dev 登录在注册阶段判断

`build_auth_router(dev_enabled=...)` 而不是"注册了但运行时拒绝"：
端点不存在时连存在性都无法探测；运行时拒绝则会让攻击者确认路径存在，
且一旦配置判断写错就真的能登录。

## `/me` 为什么查库

token 里只有 user_id 与 tenant_id。展示用的邮箱、角色必须从库里查
——用户可能改名、角色可能被调整，token 里是签发时的快照。

这也顺带验证了"token 有效但用户已被禁用"：查库时发现
`status != active` 就拒绝。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import Principal, get_current_principal, get_db
from rag_api.identity.factory import build_identity_provider
from rag_api.identity.provider import ExternalIdentity, IdentityProviderError
from rag_api.identity.tokens import TokenService
from rag_api.openapi import AUTHENTICATED_ENDPOINT, PUBLIC_ENDPOINT
from rag_contracts.api.auth import (
    CurrentUserResponse,
    DevLoginRequest,
    LoginResponse,
    RefreshRequest,
    RefreshResponse,
)
from rag_contracts.enums import TenantRole
from rag_core.db import Database
from rag_core.errors import AuthenticationError
from rag_core.logging import get_logger
from rag_db.repositories import (
    MembershipRepository,
    TenantRepository,
    UserRepository,
)

log = get_logger(__name__)


def build_auth_router(*, dev_enabled: bool) -> APIRouter:
    """构造认证路由。dev 端点只在 `dev_enabled` 时挂载。"""
    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post(
        "/refresh",
        response_model=RefreshResponse,
        summary="刷新 access token",
        description=(
            "用 refresh token 换一对新 token（access + refresh 都换）。\n\n"
            "旧的 refresh token 立刻作废。同一个 token 被用第二次说明有第三方"
            "持有它（凭据泄漏），系统会作废整族——攻击者与受害者都必须重新登录。\n\n"
            "客户端必须保存返回的新 refresh token，且并发刷新时只发一次请求"
            "（否则后几次会撞复用检测）。"
        ),
        responses=PUBLIC_ENDPOINT,
    )
    async def refresh_token_endpoint(  # pyright: ignore[reportUnusedFunction]
        payload: RefreshRequest,
        request: Request,
    ) -> RefreshResponse:
        from rag_api.identity.refresh import RefreshTokenStore

        token_service: TokenService = request.app.state.token_service
        redis = request.app.state.redis

        # rotate 内部做：校验 → 标记旧 token 已用 → 签发新 token。
        # 检测到复用时抛 RefreshTokenReusedError（已作废整族）。
        user_id, tenant_id, new_refresh, _family = await RefreshTokenStore(
            redis.client
        ).rotate(refresh_token=payload.refresh_token)

        access_token, expires_at = token_service.issue(
            user_id=user_id, tenant_id=tenant_id
        )
        log.info(
            "token_refreshed",
            user_id=str(user_id),
            tenant_id=str(tenant_id),
        )
        return RefreshResponse(
            access_token=access_token,
            refresh_token=new_refresh,
            expires_at=expires_at,
        )

    if dev_enabled:
        _register_dev_routes(router)
    return router


def build_me_router() -> APIRouter:
    """构造 `/me` 路由。"""
    router = APIRouter(tags=["me"])

    @router.get(
        "/me",
        response_model=CurrentUserResponse,
        summary="当前用户信息",
        description="返回 token 对应的用户与当前租户角色。",
        # 关键：把 401/403/422/500 声明进 OpenAPI，
        # 前端生成 TS 类型时才能看到错误响应的形状
        responses=AUTHENTICATED_ENDPOINT,
    )
    async def get_me(
        principal: Principal = Depends(get_current_principal),
        session: AsyncSession = Depends(get_db),
    ) -> CurrentUserResponse:
        user = await UserRepository(session).get_active_by_id(principal.user_id)
        if user is None:
            # token 有效但用户已禁用/删除：必须拒绝，否则被禁用的用户还能操作
            raise AuthenticationError("账号不可用，请重新登录")

        membership = await MembershipRepository(session).get_active(
            tenant_id=principal.tenant_id, user_id=principal.user_id
        )
        return CurrentUserResponse(
            user_id=user.id,
            tenant_id=principal.tenant_id,
            email=user.email,
            display_name=user.display_name,
            status=str(user.status),
            platform_role=str(user.platform_role),
            tenant_role=str(membership.role) if membership is not None else None,
        )

    return router


async def _issue_refresh_token(
    request: Request, *, user_id: uuid.UUID, tenant_id: uuid.UUID
) -> str:
    """签发 refresh token。

    **只在登录时签一次**（首次，family_id=None 表示新族）。
    后续续期走 `/auth/refresh` 的 rotate（保持同一 family）。
    """
    from rag_api.identity.refresh import RefreshTokenStore

    token, _family = await RefreshTokenStore(request.app.state.redis.client).issue(
        user_id=user_id, tenant_id=tenant_id
    )
    return token


def _register_dev_routes(router: APIRouter) -> None:
    """挂载 dev 专用端点。

    调用方（`build_auth_router`）已确认 `dev_enabled`，
    且 `Settings` 的生产守卫会阻止生产环境开启它。
    """

    @router.post(
        "/dev-login",
        response_model=LoginResponse,
        summary="开发登录（仅非生产）",
        description=(
            "用 `dev:<subject>` 凭证登录，自动创建本地用户。"
            "**仅在 RAG_DEV_AUTH_ENABLED=true 时存在**，生产环境调用会 404。\n\n"
            "注意：新用户若未加入任何租户会返回 403 —— 这是有意的设计，"
            "自动建租户会让任何人登录即拥有租户。本地开发请用 `/dev-bootstrap`。"
        ),
        responses=PUBLIC_ENDPOINT,
    )
    async def dev_login(  # pyright: ignore[reportUnusedFunction]
        payload: DevLoginRequest,
        request: Request,
    ) -> LoginResponse:
        identity = await _verify_dev_credential(request, payload.credential)
        token_service: TokenService = request.app.state.token_service
        database: Database = request.app.state.database

        from rag_api.services.auth import AuthService

        # 首登需要建用户 / 读成员关系，此时尚无租户上下文；
        # 用管理连接（BYPASSRLS）——开通用户与租户本质上是跨租户的管理动作
        async with database.transaction_bypass_rls() as session:
            resolved = await AuthService(session).resolve_login(identity)

        token, expires_at = token_service.issue(
            user_id=resolved.user_id, tenant_id=resolved.tenant_id
        )
        # 签发 refresh token（A7 的轮换机制在此之前没有暴露端点，
        # 导致 access token 15 分钟过期后只能重新登录）
        refresh = await _issue_refresh_token(
            request, user_id=resolved.user_id, tenant_id=resolved.tenant_id
        )
        log.info(
            "dev_login_succeeded",
            user_id=str(resolved.user_id),
            tenant_id=str(resolved.tenant_id),
            is_new_user=resolved.created,
        )
        return LoginResponse(
            access_token=token,
            refresh_token=refresh,
            expires_at=expires_at,
            user_id=resolved.user_id,
            tenant_id=resolved.tenant_id,
            email=resolved.email,
            display_name=resolved.display_name,
            is_new_user=resolved.created,
        )

    @router.post(
        "/dev-bootstrap",
        response_model=LoginResponse,
        summary="开发引导：建租户并登录（仅非生产）",
        description=(
            "开发便利端点：若用户尚未加入任何租户，自动创建 "
            "`dev-<subject>` 租户并设为 owner，然后返回 token。\n\n"
            "**生产环境不存在此端点**。真实流程里开通租户必须由管理员操作——"
            "自动建租户会让任何人登录即拥有租户，配额完全失控。"
        ),
        responses=PUBLIC_ENDPOINT,
    )
    async def dev_bootstrap(  # pyright: ignore[reportUnusedFunction]
        payload: DevLoginRequest,
        request: Request,
    ) -> LoginResponse:
        identity = await _verify_dev_credential(request, payload.credential)
        token_service: TokenService = request.app.state.token_service
        database: Database = request.app.state.database

        # 建租户/成员是跨租户管理动作，用管理连接
        async with database.transaction_bypass_rls() as session:
            users = UserRepository(session)
            link = await users.find_by_identity(issuer=identity.issuer, subject=identity.subject)
            if link is not None:
                user = await users.get_by_id(link.user_id)
                if user is None:
                    raise AuthenticationError("账号数据异常，请联系管理员")
                is_new = False
            else:
                user = await users.create_user(
                    email=identity.email or f"{identity.subject}@{identity.issuer}",
                    display_name=identity.display_name or identity.subject,
                )
                await users.create_link(
                    user_id=user.id,
                    issuer=identity.issuer,
                    subject=identity.subject,
                    email=identity.email,
                )
                is_new = True

            memberships = await MembershipRepository(session).list_for_user(user.id)
            active = [m for m in memberships if m.status == "active"]
            if active:
                tenant_id = active[0].tenant_id
            else:
                slug = f"dev-{identity.subject}"[:64]
                tenants = TenantRepository(session)
                tenant = await tenants.get_by_slug(slug)
                if tenant is None:
                    tenant = await tenants.create(slug=slug, name=f"Dev — {identity.subject}")
                await MembershipRepository(session).create(
                    tenant_id=tenant.id,
                    user_id=user.id,
                    role=TenantRole.OWNER.value,
                )
                tenant_id = tenant.id

            user_id = user.id
            email = user.email
            display_name = user.display_name

        token, expires_at = token_service.issue(user_id=user_id, tenant_id=tenant_id)
        log.info(
            "dev_bootstrap_succeeded",
            user_id=str(user_id),
            tenant_id=str(tenant_id),
            is_new_user=is_new,
        )
        refresh = await _issue_refresh_token(
            request, user_id=user_id, tenant_id=tenant_id
        )
        return LoginResponse(
            access_token=token,
            refresh_token=refresh,
            expires_at=expires_at,
            user_id=user_id,
            tenant_id=tenant_id,
            email=email,
            display_name=display_name,
            is_new_user=is_new,
        )


async def _verify_dev_credential(request: Request, credential: str) -> ExternalIdentity:
    """校验 dev 凭证。失败时不暴露具体原因（防探测）。"""
    provider = build_identity_provider(request.app.state.settings)
    try:
        return await provider.verify(credential)
    except IdentityProviderError as exc:
        log.info("dev_credential_rejected", reason=exc.reason)
        raise AuthenticationError("凭证无效") from exc


def issue_token_for(
    token_service: TokenService, *, user_id: uuid.UUID, tenant_id: uuid.UUID
) -> tuple[str, int]:
    """签发 token 的薄封装，便于测试与其它路由复用。"""
    return token_service.issue(user_id=user_id, tenant_id=tenant_id)
