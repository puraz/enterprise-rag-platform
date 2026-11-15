"""身份与认证相关配置。

拆成子配置的原因与其他域一致：认证参数会随 A7 增多
（JWKS 地址、客户端 ID、PKCE 设置、token 有效期等），
塞进根 `Settings` 会变成又一个巨型类。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr


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


class OIDCSettings(BaseModel):
    """OIDC 身份提供者配置。

    适用于 Keycloak / Auth0 / Okta / Azure AD 等标准 OIDC 实现。
    换 IdP 只改这里的配置，代码不动。
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    issuer: str = "http://127.0.0.1:8080/realms/rag"
    """Issuer URL。**必须与 IdP 宣称的 `iss` 完全一致**（含尾部斜杠有无）。

    不一致会导致所有 token 校验失败（`InvalidIssuerError`），
    且错误信息不会告诉你"少了个斜杠"。Keycloak 的格式是
    `http://host:port/realms/<realm>`。
    """

    client_id: str = "rag-web"
    client_secret: SecretStr | None = None
    """confidential client 才需要。SPA 用 PKCE，不需要 secret。"""

    allowed_algorithms: tuple[str, ...] = ("RS256", "ES256")
    """允许的签名算法白名单。

    **绝不能包含 HS256 与 none**：
    - HS256 是对称算法，需要 IdP 与 RP 共享密钥——OIDC 里不存在这种共享；
    - none 表示不签名，任何攻击者都能伪造 token。

    Keycloak 默认用 RS256；Auth0 用 RS256；部分用 ES256。
    """

    jwks_cache_ttl_s: int = Field(default=3600, ge=60)
    """JWKS 缓存时间。

    太短会让每次请求都拉 JWKS（IdP 压力）；太长会让密钥轮换
    后的新 token 验证失败（但有 kid 未命中时强制刷新的兜底）。
    """

    clock_skew_s: int = Field(default=60, ge=0)
    """时钟偏差容忍（秒）。多副本间时钟可能有秒级差异。"""

    http_timeout_s: float = Field(default=10.0, gt=0)
