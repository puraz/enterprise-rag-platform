"""身份与认证相关配置。

拆成子配置的原因与其他域一致：认证参数会随 A7 增多
（JWKS 地址、客户端 ID、PKCE 设置、token 有效期等），
塞进根 `Settings` 会变成又一个巨型类。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IdentitySettings(BaseModel):
    """身份提供者与令牌配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    provider: str = "dev"
    """身份提供者名。对应 `rag_api.identity.factory._BUILDERS` 的键。

    - `dev`：本地开发（仅非生产可用）
    - A7 加 `keycloak` / `wecom` / `feishu` / `dingtalk` / `authing`
    """

    # ---- 本系统签发的 token ----
    # A4 阶段只做"可验证的签名 token"，A7 补齐 refresh 旋转与复用检测。
    jwt_algorithm: str = "HS256"
    """签名算法。

    HS256（对称）便于本地开发；生产应换 RS256（非对称），
    这样其它服务能只凭公钥验证而不需要共享密钥。
    A7 接 Keycloak 时同步切换。
    """

    jwt_secret: str = Field(
        default="dev-only-insecure-secret-change-me",
        min_length=16,
        description="HS256 签名密钥。生产必须由环境变量提供",
    )
    access_token_ttl_s: int = Field(default=1800, ge=60, le=86400)
    """access token 有效期（秒）。默认 30 分钟。"""

    issuer: str = "enterprise-rag-platform"
    """本系统作为 token 签发方的 iss 声明。"""
