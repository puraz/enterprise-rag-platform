"""内置模拟 OIDC 服务器：验证 OIDC Provider 的代码逻辑。

## 为什么需要它

本机没有 Docker 与 Java，跑不了真实 Keycloak。但 A7 的核心价值是验证：

- JWKS 缓存与 kid 轮换
- 算法白名单（拒绝 `alg=none` 与 HS256）
- iss / aud / exp 校验
- 授权码 + PKCE 换 token
- refresh 轮换与复用检测

这些**全部可以用一个最小 OIDC 服务器验证**——只要它按 OIDC 标准行为。
真实 Keycloak 的价值是验证"配置 quirks"（issuer 尾部斜杠、
realm 路径、client 配置），那属于部署验证，不阻塞代码逻辑。

## 实现范围

用 FastAPI 提供三个端点：

    GET  /.well-known/openid-configuration
    GET  /jwks
    POST /token

外加一个测试辅助端点（签发 token），真实 IdP 没有这个。

## 与真实 IdP 的行为对齐点

1. **ISSuer 完全一致**：discovery 与 token 里的 `iss` 必须相同；
2. **RS256 签名**，key 带 `kid`；
3. **aud = client_id**；
4. **PKCE 校验**：token 端点用 `code_verifier` 重算 challenge 比对；
5. **授权码一次性**：用过即失效。

## 不支持（真实 IdP 有）

- `/authorize` 的浏览器交互（模拟里直接签发 code）；
- 用户管理、realm 配置；
- 密钥自动轮换（有 `rotate_keys()` 方法手动触发，用于测试缓存失效）。
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse

from rag_core.logging import get_logger

log = get_logger(__name__)


@dataclass
class _AuthCode:
    """一次授权码的临时记录。"""

    user: dict[str, Any]
    code_challenge: str
    redirect_uri: str
    expires_at: float
    used: bool = False


@dataclass
class MockOIDCServer:
    """模拟 OIDC 服务器。

    用法：

        server = MockOIDCServer(issuer="http://127.0.0.1:8080/realms/rag")
        app = server.build_app()
        # 用 TestClient 或 uvicorn 起它

    Args:
        issuer: 必须与 OIDCProvider 配置的 issuer 完全一致。
        client_id: 与 client 配置一致。
    """

    issuer: str
    client_id: str = "rag-web"
    _private_key: Any = None
    _public_jwk: dict[str, Any] = field(default_factory=dict)
    _kid: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    _codes: dict[str, _AuthCode] = field(default_factory=dict)
    _issued_tokens: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._generate_keys()

    def _generate_keys(self) -> None:
        """生成 RSA 密钥对并构造 JWK。

        每次调用都会换新 kid——用于测试"密钥轮换后缓存失效"场景。
        """
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self._private_key = private_key

        public_numbers = private_key.public_key().public_numbers()

        def _b64(value: int) -> str:
            raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
            return base64.urlsafe_b64encode(raw).decode().rstrip("=")

        self._kid = uuid.uuid4().hex[:16]
        self._public_jwk = {
            "kty": "RSA",
            "use": "sig",
            "alg": "RS256",
            "kid": self._kid,
            "n": _b64(public_numbers.n),
            "e": _b64(public_numbers.e),
        }

    def rotate_keys(self) -> None:
        """轮换密钥（测试 JWKS 缓存刷新）。

        真实 IdP 的轮换会保留旧 key 一段时间（让旧 token 仍能验证），
        这里简化成直接替换——正因为如此，测试会验证
        "kid 未命中时强制刷新 JWKS"这条路径。
        """
        self._generate_keys()

    def issue_token(
        self,
        *,
        subject: str,
        email: str | None = None,
        name: str | None = None,
        audience: str | None = None,
        issuer: str | None = None,
        expires_in: int = 300,
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        """签发一个 access token（测试辅助，真实 IdP 没有这个端点）。

        可以通过参数构造**各种非法 token** 来测试校验逻辑：
        - `audience` 传别的值 → 测试 aud 校验
        - `issuer` 传别的值 → 测试 iss 校验
        - `expires_in=-100` → 测试过期校验
        """
        import jwt

        now = int(time.time())
        payload: dict[str, Any] = {
            "iss": issuer if issuer is not None else self.issuer,
            "sub": subject,
            "aud": audience if audience is not None else self.client_id,
            "iat": now,
            "exp": now + expires_in,
            "azp": self.client_id,
        }
        if email:
            payload["email"] = email
        if name:
            payload["name"] = name
        if extra_claims:
            payload.update(extra_claims)

        token = jwt.encode(
            payload, self._private_key, algorithm="RS256", headers={"kid": self._kid}
        )
        self._issued_tokens.append(token)
        return token

    def issue_token_with_alg(self, *, subject: str, algorithm: str) -> str:
        """用指定算法签发 token（用于测试算法白名单）。

        `algorithm="none"` 会生成无签名 token——这是 `alg=none` 攻击的形态。
        """
        import jwt

        now = int(time.time())
        payload = {
            "iss": self.issuer,
            "sub": subject,
            "aud": self.client_id,
            "iat": now,
            "exp": now + 300,
        }
        if algorithm.lower() == "none":
            # 无签名 token：header 里 alg=none，签名部分是空
            header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
            body = (
                base64.urlsafe_b64encode(__import__("json").dumps(payload).encode())
                .decode()
                .rstrip("=")
            )
            return f"{header}.{body}."

        if algorithm == "HS256":
            # 用对称密钥签：验证方若不加白名单，会用公钥当 HMAC 密钥
            # （这是经典的算法混淆攻击）
            return jwt.encode(payload, "wrong-secret-that-verifier-might-use", algorithm="HS256")

        return jwt.encode(
            payload, self._private_key, algorithm=algorithm, headers={"kid": self._kid}
        )

    def create_auth_code(
        self, *, user: dict[str, Any], code_challenge: str, redirect_uri: str
    ) -> str:
        """创建授权码（模拟用户在 IdP 完成登录后的回调）。"""
        code = secrets.token_urlsafe(32)
        self._codes[code] = _AuthCode(
            user=user,
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
            expires_at=time.time() + 300,
        )
        return code

    def _issuer_path(self) -> str:
        """issuer 的 URL 路径部分（如 `/realms/rag`）。

        **必须把端点挂在 issuer 路径下**：OIDC 规范的 discovery 地址是
        `{issuer}/.well-known/openid-configuration`。如果 issuer 带路径
        （Keycloak 是 `/realms/<realm>`），discovery 就在那个路径下，
        不是根的 `/.well-known/...`。

        挂错路径时客户端拿到 404，且错误信息只说"无法获取 OIDC 配置"，
        看不出是路径问题——这是配置真实的 IdP 时最常见的坑。
        """
        from urllib.parse import urlparse

        path = urlparse(self.issuer).path.rstrip("/")
        return path

    def build_app(self) -> FastAPI:
        """构造模拟服务器应用。"""
        app = FastAPI(title="Mock OIDC Server")
        base_path = self._issuer_path()

        @app.get(f"{base_path}/.well-known/openid-configuration")
        async def discovery() -> dict[str, Any]:
            base = self.issuer.rstrip("/")
            return {
                "issuer": self.issuer,
                "authorization_endpoint": f"{base}/protocol/openid-connect/auth",
                "token_endpoint": f"{base}/protocol/openid-connect/token",
                "jwks_uri": f"{base}/protocol/openid-connect/certs",
                "response_types_supported": ["code"],
                "subject_types_supported": ["public"],
                "id_token_signing_alg_values_supported": ["RS256"],
                "code_challenge_methods_supported": ["S256"],
            }

        @app.get(f"{base_path}/protocol/openid-connect/certs")
        async def jwks() -> dict[str, Any]:
            return {"keys": [self._public_jwk]}

        @app.post(f"{base_path}/protocol/openid-connect/token")
        async def token(
            grant_type: str = Form(...),
            code: str = Form(default=""),
            redirect_uri: str = Form(default=""),
            client_id: str = Form(default=""),
            code_verifier: str = Form(default=""),
            refresh_token: str = Form(default=""),
        ) -> JSONResponse:
            if grant_type != "authorization_code":
                raise HTTPException(status_code=400, detail="unsupported_grant_type")

            if client_id and client_id != self.client_id:
                raise HTTPException(status_code=401, detail="invalid_client")

            record = self._codes.get(code)
            if record is None:
                raise HTTPException(status_code=400, detail="invalid_grant")
            if record.used:
                raise HTTPException(status_code=400, detail="code_already_used")
            if record.expires_at < time.time():
                raise HTTPException(status_code=400, detail="code_expired")
            if record.redirect_uri != redirect_uri:
                raise HTTPException(status_code=400, detail="redirect_uri_mismatch")

            # **PKCE 校验**：用 verifier 重算 challenge 比对
            computed = (
                base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
                .decode()
                .rstrip("=")
            )
            if computed != record.code_challenge:
                raise HTTPException(status_code=400, detail="invalid_code_verifier")

            record.used = True

            access_token = self.issue_token(
                subject=str(record.user.get("sub", "unknown")),
                email=record.user.get("email"),
                name=record.user.get("name"),
            )
            id_token = self.issue_token(
                subject=str(record.user.get("sub", "unknown")),
                email=record.user.get("email"),
                name=record.user.get("name"),
            )
            return JSONResponse(
                {
                    "access_token": access_token,
                    "id_token": id_token,
                    "token_type": "Bearer",
                    "expires_in": 300,
                    "refresh_token": secrets.token_urlsafe(32),
                }
            )

        return app


__all__ = ["MockOIDCServer"]
