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
