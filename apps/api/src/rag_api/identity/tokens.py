"""本系统令牌签发与校验。

## A4 的范围 vs A7 的范围

A4 做**最小可用**：签发 access token、校验签名与过期。
A7 补齐：refresh token、旋转、复用检测、强制下线（`tokens_valid_after`）。

**为什么 A4 不做 refresh 就够用**：本地开发与自动化测试只需
"能登录、能带着 token 调受保护接口"。先把链路跑通，
A7 再补完整的令牌生命周期，避免 A4 就陷入 OIDC/PKCE 的细节。

## 为什么不用 `python-jose`

`python-jose` 长期缺少维护，且有已知的算法混淆 CVE（`alg=none` 攻击）。
`PyJWT` 是更活跃的选择，且强制要求显式指定 `algorithms` 参数——
**不做算法白名单的 JWT 校验是严重漏洞**：攻击者可以把 header 改成
`{"alg": "none"}` 或换成对称算法来绕过签名。

A4 用 HS256（对称），A7 接 Keycloak 时切 RS256 + JWKS。

## Claims 设计

    {
      "iss": "enterprise-rag-platform",
      "sub": "<user_id>",          # 本地用户 ID
      "tid": "<tenant_id>",        # 当前激活的租户
      "iat": <签发时间>,            # 用于 tokens_valid_after 比对（强制下线）
      "exp": <过期时间>,
      "jti": "<token 唯一 ID>"      # A7 用于撤销列表
    }

`sub` 必须放**本地 user_id** 而不是外部 subject：
- 外部 subject 在多个 IdP 下可能重复；
- 每次请求都查 identity_links 才能知道是谁，多一次查询；
- 而且本地 ID 才是我们所有表的外键。

`tid` 放在 token 里而不是让客户端传：`docs/PROMPT.md` 要求
「tenant_id 从 token 解析，绝不取请求参数」。放 token 里是唯一的正确做法。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from rag_core.errors import AuthenticationError, TokenExpiredError, TokenInvalidError
from rag_core.logging import get_logger
from rag_core.settings import IdentitySettings

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """校验通过后的 claims。只保留需要的字段，避免上层误用其它值。"""

    user_id: uuid.UUID
    tenant_id: uuid.UUID
    issued_at: int
    expires_at: int
    token_id: str


class TokenService:
    """签发与校验 access token。

    刻意做成类而不是模块函数：密钥与算法来自配置，
    测试里可以构造不同配置的实例，不必改全局状态。
    """

    def __init__(self, settings: IdentitySettings) -> None:
        self._settings = settings

    def issue(self, *, user_id: uuid.UUID, tenant_id: uuid.UUID) -> tuple[str, int]:
        """签发 access token，返回 `(token, expires_at_epoch)`。

        返回过期时间给客户端：前端据此决定何时提前刷新，
        而不是等到 401 才知道过期了（那会导致用户操作中断）。
        """
        import jwt

        now = int(time.time())
        expires_at = now + self._settings.access_token_ttl_s
        claims = {
            "iss": self._settings.issuer,
            "sub": str(user_id),
            "tid": str(tenant_id),
            "iat": now,
            "exp": expires_at,
            "jti": uuid.uuid4().hex,
        }
        token = jwt.encode(
            claims,
            self._settings.jwt_secret,
            algorithm=self._settings.jwt_algorithm,
        )
        return token, expires_at

    def verify(self, token: str) -> TokenClaims:
        """校验 token，返回 claims。

        Raises:
            TokenExpiredError: 已过期（前端应尝试 refresh，A7 实现）
            TokenInvalidError: 签名/格式/issuer 不合法

        **必须显式传 `algorithms=[...]`**：PyJWT 要求这样，
        否则会接受 header 里声明的任意算法（`alg=none` 攻击）。
        """
        import jwt

        try:
            payload = jwt.decode(
                token,
                self._settings.jwt_secret,
                # 算法白名单：只接受配置里指定的那一个。
                # 这里如果写成 `algorithms=jwt.algorithms.get_default_algorithms()`
                # 就等于允许所有算法，包括 none。
                algorithms=[self._settings.jwt_algorithm],
                issuer=self._settings.issuer,
                # 留一点时钟偏差容忍：多副本间时钟可能有秒级差异
                leeway=30,
                options={"require": ["exp", "iat", "iss", "sub", "tid"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise TokenExpiredError() from exc
        except jwt.InvalidTokenError as exc:
            # 不把底层原因（"Signature verification failed" 等）传给客户端：
            # 那等于告诉攻击者哪一步错了。
            log.info("token_verify_failed", reason=type(exc).__name__)
            raise TokenInvalidError() from exc

        try:
            user_id = uuid.UUID(str(payload["sub"]))
            tenant_id = uuid.UUID(str(payload["tid"]))
        except (ValueError, KeyError) as exc:
            raise TokenInvalidError() from exc

        return TokenClaims(
            user_id=user_id,
            tenant_id=tenant_id,
            issued_at=int(payload["iat"]),
            expires_at=int(payload["exp"]),
            token_id=str(payload.get("jti", "")),
        )


def bearer_token(authorization: str | None) -> str:
    """从 `Authorization` 头提取 Bearer token。

    严格校验前缀大小写：HTTP 规范里 scheme 不区分大小写，
    但混用会导致调试困难，所以接受 `Bearer` / `bearer` 两种常见写法。
    """
    if not authorization:
        raise AuthenticationError("缺少 Authorization 头")

    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise AuthenticationError("Authorization 头必须是 Bearer 方案")
    if not value.strip():
        raise AuthenticationError("Bearer token 为空")
    return value.strip()
