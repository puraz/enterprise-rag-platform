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
