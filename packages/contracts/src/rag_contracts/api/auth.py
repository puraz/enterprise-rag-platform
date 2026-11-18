"""认证相关请求/响应契约。

**必须放 contracts 包**（而不是 apps/api/schemas）：前端需要这些类型的
TS 定义，而 TS 由 `packages/contracts` 导出。放在 api 里前端就拿不到。
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class DevLoginRequest(BaseModel):
    """开发登录请求（仅非生产可用）。"""

    model_config = ConfigDict(extra="forbid")

    credential: str = Field(
        description="dev 凭证，格式 `dev:<subject>` 或 `dev:<issuer>:<subject>`",
        min_length=5,
        max_length=200,
        examples=["dev:alice"],
    )


class LoginResponse(BaseModel):
    """登录成功的响应。"""

    model_config = ConfigDict(frozen=True)

    access_token: str = Field(description="本系统签发的 access token")
    refresh_token: str = Field(
        description=(
            "刷新令牌（只在这里返回一次）。"
            "客户端应保存并在 access token 过期时用它换新的一对 token。"
            "**轮换语义**：每次刷新都会作废旧 refresh token 并签发新的，"
            "所以客户端必须覆盖保存，不能在并发请求里重复使用同一个。"
        )
    )
    token_type: str = Field(default="Bearer", description="固定为 Bearer")
    expires_at: int = Field(description="token 过期时间（Unix 秒），前端据此提前刷新")
    user_id: uuid.UUID = Field(description="本地用户 ID")
    tenant_id: uuid.UUID = Field(description="当前激活的租户 ID")
    email: str
    display_name: str
    is_new_user: bool = Field(description="是否本次登录新建的用户（首登）")


class CurrentUserResponse(BaseModel):
    """`GET /me` 的响应。"""

    model_config = ConfigDict(frozen=True)

    user_id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    display_name: str
    status: str
    platform_role: str
    tenant_role: str | None = Field(
        default=None, description="当前租户内的角色（owner/admin/…），非成员时为 null"
    )


class HealthResponse(BaseModel):
    """健康检查响应。"""

    model_config = ConfigDict(frozen=True)

    status: str = Field(description="ok / degraded / error")
    version: str
    environment: str
    checks: dict[str, str] = Field(
        default_factory=dict, description="各依赖的检查结果（ok / 错误摘要）"
    )


class OIDCAuthorizeResponse(BaseModel):
    """OIDC 登录跳转信息。

    ## 为什么不在服务端 302 直接跳

    前端 SPA 需要先把 `state` 与 `code_verifier` 存到 localStorage/sessionStorage
    （缓存住才能在校验回调时用）。服务端直接 302 的话，前端拿不到这两个值，
    回调时就无法完成 PKCE 校验。

    所以拆成两步：本接口返回 URL 与 state，前端自己跳。
    `code_verifier` **绝不出现在响应里**（它必须只留在浏览器）。
    """

    model_config = ConfigDict(frozen=True)

    authorize_url: str = Field(description="IdP 授权地址（含 state 与 PKCE challenge）")
    state: str = Field(description="CSRF 校验值，前端需原样带回回调接口")


class OIDCExchangeRequest(BaseModel):
    """授权码换取本系统 token。

    ## 为什么 code_verifier 由前端传

    PKCE 的前提是"只有发起授权的客户端持有 verifier"。放在服务端状态里
    会被多副本/重启破坏，也会让服务端变成可被重放的中心。
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=2048, description="IdP 回调带回的授权码")
    state: str = Field(min_length=1, max_length=256, description="授权时下发的 state")
    code_verifier: str = Field(
        min_length=43, max_length=128, description="前端保存的 PKCE code_verifier"
    )
    redirect_uri: str = Field(min_length=1, max_length=500, description="必须与授权时一致")


class LogoutRequest(BaseModel):
    """登出请求。

    `refresh_token` 可选：只传 access token 时按"不报文"处理（用户可能
    已经丢了本地存储）。带上时同时作废该 refresh token。
    """

    model_config = ConfigDict(extra="forbid")

    refresh_token: str | None = Field(
        default=None,
        description="要作废的 refresh token。只作废当前设备，不影响其它设备",
    )


class LogoutResponse(BaseModel):
    """登出响应。

    ## 为什么不是 204

    登出是"有结果的动作"：前端要知道 refresh token 是否真的被作废
    （比如它已过期时，服务端只能返回 revoked=false）。用单对象表达，
    与"无内容"的三类形态不冲突。
    """

    model_config = ConfigDict(frozen=True)

    revoked: bool = Field(description="是否成功作废了 refresh token（未提供时为 false）")


class TenantMembershipItem(BaseModel):
    """用户可访问的一个租户。

    ## 为什么要有这个接口

    用户可能属于多个租户（外部顾问、跨部门协作）。但 access token 里
    只能带**一个** `tenant_id`，没有列表接口前端就不知道"我还能切到哪"，
    表现为"登录后永远停在第一个租户"。
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: uuid.UUID
    tenant_slug: str
    tenant_name: str
    role: str = Field(description="该租户内的角色：owner/admin/knowledge_admin/member/viewer")
    is_active: bool = Field(description="该租户当前是否可用（暂停/删除的为 false）")


class TenantListResponse(BaseModel):
    """`GET /me/tenants` 响应。"""

    model_config = ConfigDict(frozen=True)

    items: list[TenantMembershipItem]
    active_tenant_id: uuid.UUID = Field(description="当前 token 生效的租户")


class SwitchTenantRequest(BaseModel):
    """切换租户。

    **tenant_id 只用于校验成员关系**：新 token 的 tenant 来自服务端
    校验通过的成员关系，绝不直接采信请求参数。
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: uuid.UUID = Field(description="目标租户（必须是当前用户的 active 成员关系）")


class RefreshRequest(BaseModel):
    """刷新 token 请求。

    ## 为什么 refresh token 放在 body 而不是 Authorization 头

    语义上它是"凭据"而不是"身份声明"：用 Authorization 头会让中间层
    （网关、日志）把它当 access token 记录到访问日志里。
    放 body 则只出现在请求体（通常不记日志）。
    """

    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(
        min_length=20, max_length=200, description="登录时返回的 refresh token"
    )


class RefreshResponse(BaseModel):
    """刷新结果：一对新 token。

    **旧的 refresh token 已作废**（轮换）。客户端必须用新的替换旧的，
    否则下次刷新会因"复用检测"导致整族作废、强制重新登录。
    """

    model_config = ConfigDict(frozen=True)

    access_token: str
    refresh_token: str
    token_type: str = Field(default="Bearer")
    expires_at: int = Field(description="access token 过期时间（Unix 秒）")


__all__ = [
    "CurrentUserResponse",
    "DevLoginRequest",
    "HealthResponse",
    "LoginResponse",
    "LogoutRequest",
    "LogoutResponse",
    "OIDCAuthorizeResponse",
    "OIDCExchangeRequest",
    "RefreshRequest",
    "RefreshResponse",
    "SwitchTenantRequest",
    "TenantListResponse",
    "TenantMembershipItem",
]
