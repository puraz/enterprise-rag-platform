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

## 业务逻辑的位置

用户映射、开发引导建租户、`/me` 的查询与映射都在 `AuthService`；
本模块只做 HTTP 语义与 token 装配（Redis / TokenService 的接线）。
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import get_current_principal, get_db, get_session
from rag_api.deps.auth import Principal
from rag_api.identity.factory import build_identity_provider
from rag_api.identity.provider import ExternalIdentity, IdentityProviderError
from rag_api.identity.tokens import TokenService
from rag_api.openapi import AUTHENTICATED_ENDPOINT, PUBLIC_ENDPOINT
from rag_api.schemas.auth import (
    CurrentUserResponse,
    DevLoginRequest,
    LoginResponse,
    LogoutRequest,
    LogoutResponse,
    OIDCAuthorizeResponse,
    OIDCExchangeRequest,
    RefreshRequest,
    RefreshResponse,
    SwitchTenantRequest,
    TenantListResponse,
    to_current_user_response,
    to_login_response,
    to_refresh_response,
)
from rag_api.services.auth import AuthService
from rag_core.db import Database
from rag_core.errors import AuthenticationError, IdentityProviderUnavailableError
from rag_core.logging import get_logger

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
        user_id, tenant_id, new_refresh, _family = await RefreshTokenStore(redis.client).rotate(
            refresh_token=payload.refresh_token
        )

        access_token, expires_at = token_service.issue(user_id=user_id, tenant_id=tenant_id)
        log.info(
            "token_refreshed",
            user_id=str(user_id),
            tenant_id=str(tenant_id),
        )
        return to_refresh_response(
            access_token=access_token,
            refresh_token=new_refresh,
            expires_at=expires_at,
        )

    if dev_enabled:
        _register_dev_routes(router)
    _register_oidc_routes(router)
    _register_logout_route(router)
    return router


def _register_oidc_routes(router: APIRouter) -> None:
    """挂载 OIDC 授权码 + PKCE 的两个端点。

    ## 为什么是两个端点而不是一个"登录并回调"

    标准 OIDC 流程必须有浏览器往返（用户在 IdP 登录）：

        ① GET  /auth/oidc/authorize  → {authorize_url, state}
        ② 浏览器跳到 IdP，用户登录
        ③ IdP 302 → 前端回调页，URL 里带 code & state
        ④ POST /auth/oidc/exchange   → {access_token, refresh_token, ...}

    如果把 ①④ 合成一个端点，服务端必须在 ②③ 期间持有会话状态，
    这会让无状态 API 变成有状态（多副本部署时粘性会话要求，
    重启即丢）。拆开后，状态只在 Redis（state 记录）与浏览器
    （code_verifier）里，API 保持无状态。

    ## 为什么 ④ 是 POST 而不是 GET 回调

    `code` 在 URL 里会进浏览器历史、Referer、网关访问日志。
    所以 ③ 的回调页由前端接管（不请求后端），前端立刻用 POST
    把 code 换成 token，缩短 code 的暴露窗口。
    """

    from rag_api.identity.oidc_state import OIDCStateStore

    @router.get(
        "/oidc/authorize",
        response_model=OIDCAuthorizeResponse,
        summary="发起 OIDC 登录（返回跳转地址）",
        description=(
            "生成 PKCE challenge 与一次性 state，返回 IdP 授权地址。\n\n"
            "**前端必须保存 state**，回调时原样传回 `POST /auth/oidc/exchange`。\n\n"
            "`code_verifier` **不在此响应里**：它不是这个接口生成的，"
            "必须由前端自己生成并保存（否则 PKCE 失去意义——服务端持有一份"
            "就等于任何人拿到 state 都能完成授权）。\n\n"
            "未配置 OIDC 时返回 503（`AUTH.IDP_UNAVAILABLE`），前端应提示"
            "「企业账号登录不可用，请联系管理员」。"
        ),
        responses=PUBLIC_ENDPOINT,
    )
    async def oidc_authorize(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        redirect_uri: str,
        code_challenge: str,
        code_challenge_method: str = "S256",
    ) -> OIDCAuthorizeResponse:
        """返回授权地址。

        ## 为什么 code_challenge 由前端传

        见上：verifier 必须在浏览器里。前端先调
        `crypto.subtle.digest('SHA-256', verifier)` 得到 challenge，
        再把它交给本接口去拼 URL。

        ## `code_challenge_method` 只接受 S256

        `plain` 让 challenge 与 verifier 相同，等于把 verifier 暴露在
        authorize URL（进浏览器历史与代理日志）。虽然 RFC 允许，但
        没有任何理由用它。
        """
        from rag_core.errors import ValidationFailedError

        if code_challenge_method != "S256":
            raise ValidationFailedError(
                "只支持 S256 PKCE",
                details={"supported": "S256"},
            )
        if not code_challenge or len(code_challenge) < 43:
            raise ValidationFailedError("code_challenge 非法（S256 至少 43 字符）")

        settings = request.app.state.settings
        if settings.identity.provider == "dev":
            # dev 环境没有 IdP：显式报错而不是让前端跳到一个空地址
            raise IdentityProviderUnavailableError(
                "当前环境未配置企业身份提供者，请使用本地开发登录"
            )

        provider = _oidc_provider(request)
        state_store = OIDCStateStore(request.app.state.redis.client)
        record = await state_store.issue(
            redirect_uri=redirect_uri,
            client_ip=_client_ip(request),
        )

        authorize_url = await provider.build_authorize_url(
            redirect_uri=redirect_uri,
            state=record.state,
            code_challenge=code_challenge,
        )
        return OIDCAuthorizeResponse(authorize_url=authorize_url, state=record.state)

    @router.post(
        "/oidc/exchange",
        response_model=LoginResponse,
        summary="OIDC 回调：授权码换本系统 token",
        description=(
            "校验 state（一次性消费）→ 用 code + verifier 换 IdP token → "
            "验签 → 映射本地用户 → 签发本系统 token。\n\n"
            "**state 用过即失效**：同一个 state 第二次提交会被拒绝（重放防护）。\n\n"
            "首次登录会自动创建本地用户；若该用户不属于任何租户，"
            "返回 403 并提示联系管理员开通（**不自动建租户**）。"
        ),
        responses=PUBLIC_ENDPOINT,
    )
    async def oidc_exchange(  # pyright: ignore[reportUnusedFunction]
        payload: OIDCExchangeRequest,
        request: Request,
    ) -> LoginResponse:
        settings = request.app.state.settings
        if settings.identity.provider == "dev":
            raise IdentityProviderUnavailableError("当前环境未配置企业身份提供者")

        provider = _oidc_provider(request)
        state_store = OIDCStateStore(request.app.state.redis.client)

        # 先消费 state：即使后面 exchange 失败也不能重试同一个 state
        # （失败的 exchange 可能已经把授权码用掉了）
        record = await state_store.consume(state=payload.state)
        if record.redirect_uri != payload.redirect_uri:
            # 回调 URI 不一致是重定向劫持的典型特征
            log.warning("oidc_redirect_uri_mismatch", expected=record.redirect_uri)
            raise AuthenticationError("回调地址不一致，请重新登录")

        tokens = await provider.exchange_code_for_tokens(
            code=payload.code,
            code_verifier=payload.code_verifier,
            redirect_uri=payload.redirect_uri,
        )
        credential = str(tokens.get("id_token") or tokens.get("access_token") or "")
        if not credential:
            raise AuthenticationError("身份提供者未返回有效凭证")

        identity = await _verify_external_credential(provider, credential)

        token_service: TokenService = request.app.state.token_service
        database: Database = request.app.state.database

        # 首登建用户/读成员关系时还没有租户上下文 —— 用管理连接
        async with database.transaction_bypass_rls() as session:
            resolved = await AuthService(session).resolve_login(identity)

        access_token, expires_at = token_service.issue(
            user_id=resolved.user_id, tenant_id=resolved.tenant_id
        )
        refresh = await _issue_refresh_token(
            request, user_id=resolved.user_id, tenant_id=resolved.tenant_id
        )
        log.info(
            "oidc_login_succeeded",
            user_id=str(resolved.user_id),
            tenant_id=str(resolved.tenant_id),
            is_new_user=resolved.created,
        )
        return to_login_response(
            resolved,
            access_token=access_token,
            refresh_token=refresh,
            expires_at=expires_at,
        )


def _register_logout_route(router: APIRouter) -> None:
    """挂载登出端点。

        ## 为什么需要它（此前是缺口）

    `RefreshTokenStore.revoke_token()` 早就实现了，但**没有任何端点调用它** ——
        用户点"退出登录"只能清本地存储，refresh token 在服务端仍然有效
        （30 天）。被盗的 token 在这段时间内可以持续换取 access token。

        ## 为什么只作废当前 refresh token，不作废整族

        多设备登录是正常场景。在手机登出不该把电脑也踢下线。
        "整族作废"只用于**检测到复用**（那是攻击信号），不是登出。
    """

    from rag_api.identity.refresh import RefreshTokenStore

    @router.post(
        "/logout",
        response_model=LogoutResponse,
        summary="登出（作废当前 refresh token）",
        description=(
            "作废当前设备的 refresh token。其它设备不受影响。\n\n"
            "**access token 依然会用自己的有效期自然过期**：它无状态，"
            "服务端不维护黑名单。需要立即失效全部 token 请由管理员"
            "触发强制下线（`tokens_valid_after`）。"
        ),
        responses=AUTHENTICATED_ENDPOINT,
    )
    async def logout(  # pyright: ignore[reportUnusedFunction]
        payload: LogoutRequest,
        request: Request,
        principal: Principal = Depends(get_current_principal),
    ) -> LogoutResponse:
        if payload.refresh_token is None:
            # 没有 refresh token 可作废：用户可能只清了本地存储。
            # 不报错（登出必须是幂等的），如实告诉前端 revoked=false。
            log.info("logout_without_refresh_token", user_id=str(principal.user_id))
            return LogoutResponse(revoked=False)

        store = RefreshTokenStore(request.app.state.redis.client)
        await store.revoke_token(refresh_token=payload.refresh_token)
        log.info("logout_revoked_refresh_token", user_id=str(principal.user_id))
        return LogoutResponse(revoked=True)


def _oidc_provider(request: Request) -> Any:
    """取（或构造）OIDC provider 单例。

    与 LLM provider 一样的惰性单例策略：构造需要 discovery 配置，
    配置错时应该在**用到时**报 503，而不是让整个 API 起不来。
    """
    existing = getattr(request.app.state, "_oidc_provider", None)
    if existing is not None:
        return existing

    from rag_api.identity.oidc import OIDCIdentityProvider

    settings = request.app.state.settings
    provider = OIDCIdentityProvider(settings.oidc)
    request.app.state._oidc_provider = provider
    return provider


async def _verify_external_credential(provider: Any, credential: str) -> ExternalIdentity:
    """校验 IdP 凭证。失败时统一转成 401（不暴露具体原因）。"""
    try:
        return await provider.verify(credential)  # type: ignore[no-any-return]
    except IdentityProviderError as exc:
        log.info("oidc_credential_rejected", reason=exc.reason)
        raise AuthenticationError("企业账号凭证校验失败，请重新登录") from exc


def _client_ip(request: Request) -> str | None:
    """取客户端 IP（仅用于审计，可空）。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


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
        user, membership = await AuthService(session).current_user(
            user_id=principal.user_id, tenant_id=principal.tenant_id
        )
        return to_current_user_response(user, tenant_id=principal.tenant_id, membership=membership)

    @router.get(
        "/me/tenants",
        response_model=TenantListResponse,
        summary="我所属的租户列表",
        description=(
            "返回当前用户所有 active 租户成员关系，并标出 token 里生效的那个。\n\n"
            "**为什么需要它**：access token 只能带一个 `tenant_id`。没有这个接口，"
            "多租户用户（外部顾问、跨部门协作）登录后就永远停在第一个租户，"
            "无法知道还能切到哪。"
        ),
        responses=AUTHENTICATED_ENDPOINT,
    )
    async def list_my_tenants(
        principal: Principal = Depends(get_current_principal),
        session: AsyncSession = Depends(get_session),
    ) -> TenantListResponse:
        items = await AuthService(session).list_tenants(user_id=principal.user_id)
        return TenantListResponse(
            items=items,
            active_tenant_id=principal.tenant_id,
        )

    @router.post(
        "/me/tenants/switch",
        response_model=LoginResponse,
        summary="切换当前租户（换发 token）",
        description=(
            "校验目标租户的 active 成员关系后，签发一个 `tid` 指向该租户的新 token。\n\n"
            "**refresh token 也会换新**：旧的绑定在旧租户上，继续用会让"
            "`/me/tenants` 的 active 标记与新 token 不一致。\n\n"
            "只有**自己**的成员关系会被检查——请求里的 `tenant_id` 只用于"
            "定位候选，不构成授权依据。"
        ),
        responses=AUTHENTICATED_ENDPOINT,
    )
    async def switch_tenant(
        payload: SwitchTenantRequest,
        request: Request,
        principal: Principal = Depends(get_current_principal),
        session: AsyncSession = Depends(get_session),
    ) -> LoginResponse:
        resolved = await AuthService(session).switch_tenant(
            user_id=principal.user_id, target_tenant_id=payload.tenant_id
        )
        token_service: TokenService = request.app.state.token_service
        access_token, expires_at = token_service.issue(
            user_id=resolved.user_id, tenant_id=resolved.tenant_id
        )
        refresh = await _issue_refresh_token(
            request, user_id=resolved.user_id, tenant_id=resolved.tenant_id
        )
        log.info(
            "tenant_switched",
            user_id=str(resolved.user_id),
            from_tenant=str(principal.tenant_id),
            to_tenant=str(resolved.tenant_id),
        )
        return to_login_response(
            resolved,
            access_token=access_token,
            refresh_token=refresh,
            expires_at=expires_at,
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
        return to_login_response(
            resolved,
            access_token=token,
            refresh_token=refresh,
            expires_at=expires_at,
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
            resolved = await AuthService(session).bootstrap_tenant(identity)

        token, expires_at = token_service.issue(
            user_id=resolved.user_id, tenant_id=resolved.tenant_id
        )
        log.info(
            "dev_bootstrap_succeeded",
            user_id=str(resolved.user_id),
            tenant_id=str(resolved.tenant_id),
            is_new_user=resolved.created,
        )
        refresh = await _issue_refresh_token(
            request, user_id=resolved.user_id, tenant_id=resolved.tenant_id
        )
        return to_login_response(
            resolved,
            access_token=token,
            refresh_token=refresh,
            expires_at=expires_at,
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


__all__ = ["build_auth_router", "build_me_router", "issue_token_for"]
