"""通用 OIDC 身份提供者（Keycloak / Auth0 / Okta / 企业 SSO 通用）。

## 为什么基于 OIDC 标准而不是 Keycloak 专用 API

`docs/PROMPT.md` 要求"多 IdP 可配（Keycloak / 企业微信 / 飞书 / 钉钉 / Authing）"。
OIDC 是这其中的共同语言——Keycloak、Auth0、Okta、Azure AD、
Google Workspace 都是标准 OIDC。基于标准实现，换 IdP 只改配置
（issuer / client_id / client_secret），代码不动。

企业微信/飞书/钉钉的 OAuth2 不完全符合 OIDC（缺 id_token 或 claims 不规范），
那些需要单独的适配器，但它们的 `IdentityProvider` 接口相同（A4 已定义）。

## 不手写 JWT 验签（关键安全点）

`docs/PROMPT.md` 明确要求"不手写 JWT 校验：缓存 JWKS，验签 + 校验
iss/aud/exp/nonce"。这不是保守，是必须：

| 自己写容易犯的错 | 后果 |
|---|---|
| 不校验 `alg`（信任 token 头里的算法） | `alg=none` 攻击：攻击者伪造任意 token |
| 不校验 `iss` | 用别的 IdP 签的 token 冒充本系统的 |
| 不校验 `aud` | 用签给别的客户端的 token 冒充本客户端 |
| 不校验 `nonce` | 重放攻击（PKCE 流程里防的是授权码注入） |
| 用 `secret` 验 RS256 | 类型错误，或降级成不验签 |
| JWKS 不过期 | 密钥轮换后全部用户登录失败（或更糟：旧密钥泄漏后仍生效） |

所以用 PyJWT + 显式算法白名单 + 完整 claims 校验。

## JWKS 缓存的两个必须

1. **按 kid 缓存 + 支持轮换**：IdP 轮换密钥时会同时公布新旧 key。
   缓存里没有对应 kid 时要**重新拉取一次**（而不是直接失败）——
   Keycloak 轮换密钥后旧 token 仍然有效，但新 token 用的是新 kid。
2. **有 TTL**：不能永久缓存。默认 1 小时；也可以在 kid 未命中时强制刷新。

## PKCE 流程（Authorization Code + PKCE）

本系统是 SPA + 后端 API，所以用 **public client + PKCE**：

    /authorize?response_type=code&client_id=...&redirect_uri=...
              &scope=openid profile email
              &code_challenge=<S256(code_verifier)>
              &code_challenge_method=S256
              &state=<random>
    → 回调 ?code=...&state=...
    → POST /token (grant_type=authorization_code, code, code_verifier, client_id)

**code_verifier 只在前端保存**（sessionStorage），不经过后端。
后端拿到 code + verifier 去换 token。

为什么必须 PKCE：public client 没有 client_secret（SPA 里存不住），
没有 PKCE 时授权码一旦泄漏（Referer 头、浏览器历史）就能被换走。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import httpx

from rag_api.identity.provider import (
    ExternalIdentity,
    IdentityProvider,
    IdentityProviderError,
)
from rag_core.logging import get_logger
from rag_core.settings.identity import OIDCSettings

log = get_logger(__name__)


@dataclass
class _JWKSCache:
    """JWKS 缓存。"""

    keys: dict[str, Any] = field(default_factory=dict)
    """kid → 公钥对象（PyJWT 能识别的格式）。"""

    fetched_at: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def is_fresh(self, ttl_s: float) -> bool:
        return bool(self.keys) and (time.monotonic() - self.fetched_at) < ttl_s


class OIDCIdentityProvider(IdentityProvider):
    """标准 OIDC 提供者。

    支持：
    - Discovery 自动发现端点（`/.well-known/openid-configuration`）；
    - JWKS 缓存 + kid 轮换；
    - RS256/ES256 验签（**显式算法白名单**）；
    - iss / aud / exp / iat 完整校验；
    - Authorization Code + PKCE 换 token。
    """

    name = "oidc"

    def __init__(
        self,
        settings: OIDCSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._jwks = _JWKSCache()
        self._discovery: dict[str, Any] | None = None
        self._client = client or httpx.AsyncClient(
            timeout=settings.http_timeout_s,
            # 认证相关请求不应跟随重定向：重定向到别的域可能是钓鱼
            follow_redirects=False,
        )
        self._owns_client = client is None

    async def _get_discovery(self) -> dict[str, Any]:
        """获取 OIDC discovery 文档（缓存）。

        `issuer/.well-known/openid-configuration` 给出 authorize/token/jwks
        的真实地址。不自己拼路径的原因：不同 IdP 的路径不一样
        （Keycloak 带 `/realms/<realm>`，Auth0 带 `/`，Azure 带 `/v2.0`）。
        """
        if self._discovery is not None:
            return self._discovery

        url = f"{self._settings.issuer.rstrip('/')}/.well-known/openid-configuration"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            discovery: dict[str, Any] = response.json()
        except httpx.HTTPError as exc:
            log.error("oidc_discovery_failed", url=url, error=type(exc).__name__)
            raise IdentityProviderError("无法获取 OIDC 配置", reason="discovery_failed") from exc

        # **校验 issuer 一致**：防 discovery 文档被篡改指向别的 IdP
        advertised = str(discovery.get("issuer", ""))
        if advertised.rstrip("/") != self._settings.issuer.rstrip("/"):
            log.error(
                "oidc_issuer_mismatch",
                expected=self._settings.issuer,
                advertised=advertised,
            )
            raise IdentityProviderError("OIDC issuer 不匹配", reason="issuer_mismatch")

        self._discovery = discovery
        return discovery

    async def _load_jwks(self, *, force: bool = False) -> dict[str, Any]:
        """加载并缓存 JWKS。

        `force=True` 用于 kid 未命中时强制刷新（IdP 刚轮换密钥）。
        """
        ttl = self._settings.jwks_cache_ttl_s
        if not force and self._jwks.is_fresh(ttl):
            return self._jwks.keys

        async with self._jwks.lock:
            # 双重检查：等锁期间可能已被其它协程刷新
            if not force and self._jwks.is_fresh(ttl):
                return self._jwks.keys

            discovery = await self._get_discovery()
            jwks_uri = str(discovery.get("jwks_uri", ""))
            if not jwks_uri:
                raise IdentityProviderError("OIDC 配置缺少 jwks_uri", reason="bad_discovery")

            try:
                response = await self._client.get(jwks_uri)
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
            except httpx.HTTPError as exc:
                log.error("oidc_jwks_fetch_failed", error=type(exc).__name__)
                raise IdentityProviderError("无法获取 JWKS", reason="jwks_failed") from exc

            keys: dict[str, Any] = {}
            for jwk in payload.get("keys") or []:
                kid = str(jwk.get("kid") or "")
                if not kid:
                    continue
                try:
                    # 用 PyJWT 的 jwk 解析（自动处理 RSA/EC）
                    from jwt import PyJWK

                    keys[kid] = PyJWK(jwk).key
                except Exception as exc:
                    # 单个 key 解析失败不影响其它 key（IdP 可能公布多种算法）
                    log.warning(
                        "oidc_jwk_parse_failed",
                        kid=kid,
                        kty=jwk.get("kty"),
                        error=type(exc).__name__,
                    )
                    continue

            if not keys:
                raise IdentityProviderError("JWKS 里没有可用密钥", reason="jwks_empty")

            self._jwks.keys = keys
            self._jwks.fetched_at = time.monotonic()
            log.info("oidc_jwks_loaded", key_count=len(keys))
            return keys

    async def verify(self, credential: str) -> ExternalIdentity:
        """校验 access token（或 id_token），返回外部身份。

        **算法白名单是硬要求**：不加白名单时 PyJWT 会接受 token 头里
        声明的任意算法，包括 `none`——攻击者可以伪造任意用户的 token。
        """
        import jwt

        credential = credential.strip()
        if not credential:
            raise IdentityProviderError("凭证为空", reason="empty")

        try:
            header = jwt.get_unverified_header(credential)
        except jwt.InvalidTokenError as exc:
            raise IdentityProviderError("凭证格式非法", reason="bad_format") from exc

        kid = str(header.get("kid") or "")
        alg = str(header.get("alg") or "")

        # **拒绝 token 自己声明的算法不在白名单内**：先查白名单再取 key，
        # 顺序反了就等于让攻击者决定用什么算法验签
        if alg not in self._settings.allowed_algorithms:
            log.warning("oidc_alg_rejected", alg=alg)
            raise IdentityProviderError("不支持的签名算法", reason="bad_alg")

        keys = await self._load_jwks()
        key = keys.get(kid)
        if key is None:
            # kid 未命中：IdP 可能刚轮换密钥，强制刷新一次
            log.info("oidc_kid_missing_refresh", kid=kid)
            keys = await self._load_jwks(force=True)
            key = keys.get(kid)
        if key is None:
            raise IdentityProviderError("找不到对应公钥", reason="unknown_kid")

        try:
            payload = jwt.decode(
                credential,
                key,
                algorithms=list(self._settings.allowed_algorithms),
                issuer=self._settings.issuer,
                audience=self._settings.client_id,
                leeway=self._settings.clock_skew_s,
                options={
                    "require": ["exp", "iat", "iss", "sub"],
                    # **必须校验 aud**：不校验的话，签给别的客户端的 token
                    # 也能用来访问本系统
                    "verify_aud": True,
                },
            )
        except jwt.ExpiredSignatureError as exc:
            raise IdentityProviderError("凭证已过期", reason="expired") from exc
        except jwt.InvalidAudienceError as exc:
            raise IdentityProviderError("凭证的 audience 不匹配", reason="bad_aud") from exc
        except jwt.InvalidIssuerError as exc:
            raise IdentityProviderError("凭证的 issuer 不匹配", reason="bad_iss") from exc
        except jwt.InvalidTokenError as exc:
            log.info("oidc_token_invalid", reason=type(exc).__name__)
            raise IdentityProviderError("凭证无效", reason="invalid") from exc

        subject = str(payload.get("sub") or "")
        if not subject:
            raise IdentityProviderError("凭证缺少 sub", reason="no_sub")

        return ExternalIdentity(
            issuer=self._settings.issuer,
            subject=subject,
            email=_str_or_none(payload.get("email")),
            display_name=_str_or_none(payload.get("name") or payload.get("preferred_username")),
            tenant_hint=_str_or_none(payload.get("tenant") or payload.get("organization")),
            roles=tuple(str(r) for r in (payload.get("roles") or [])),
            attributes={
                key_name: str(value)
                for key_name, value in payload.items()
                # 只保留标量字段，且**排除 token 本身与敏感声明**
                if key_name
                not in {
                    "nonce",
                    "at_hash",
                    "c_hash",
                    "azp",
                    "jti",
                    "session_state",
                }
                and isinstance(value, (str, int, float, bool))
            },
        )

    # ---- PKCE 流程 ----

    @staticmethod
    def generate_pkce_pair() -> tuple[str, str]:
        """生成 (code_verifier, code_challenge)。

        `code_verifier` 由前端保存，`code_challenge = BASE64URL(SHA256(verifier))`。
        用 S256 而不是 plain：plain 时 verifier 直接出现在 authorize URL 里，
        等于没有保护。
        """
        verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
        digest = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        return verifier, challenge

    async def build_authorize_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        code_challenge: str,
        scope: str = "openid profile email",
        extra_params: dict[str, str] | None = None,
    ) -> str:
        """构造授权跳转 URL。"""
        discovery = await self._get_discovery()
        authorize_endpoint = str(discovery.get("authorization_endpoint", ""))
        if not authorize_endpoint:
            raise IdentityProviderError("OIDC 配置缺少 authorization_endpoint")

        params = {
            "response_type": "code",
            "client_id": self._settings.client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        if extra_params:
            params.update(extra_params)
        return f"{authorize_endpoint}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, *, code: str, code_verifier: str, redirect_uri: str
    ) -> dict[str, Any]:
        """用授权码换 token。

        PKCE 的 `code_verifier` 必须与 authorize 时的 `code_challenge` 匹配，
        否则 IdP 拒绝——这是防授权码拦截的核心。
        """
        discovery = await self._get_discovery()
        token_endpoint = str(discovery.get("token_endpoint", ""))
        if not token_endpoint:
            raise IdentityProviderError("OIDC 配置缺少 token_endpoint")

        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self._settings.client_id,
            "code_verifier": code_verifier,
        }
        # confidential client 才需要 secret（SPA 用 PKCE，不需要）
        if self._settings.client_secret is not None:
            form["client_secret"] = self._settings.client_secret.get_secret_value()

        try:
            response = await self._client.post(token_endpoint, data=form)
        except httpx.HTTPError as exc:
            raise IdentityProviderError("换取 token 失败", reason="token_request_failed") from exc

        if response.status_code != 200:
            # 不把 IdP 的错误体直接透给客户端（可能含内部信息）
            log.warning(
                "oidc_token_exchange_rejected",
                status=response.status_code,
                body=response.text[:200],
            )
            raise IdentityProviderError("授权码无效或已过期", reason="token_exchange_rejected")

        tokens: dict[str, Any] = response.json()
        if "access_token" not in tokens and "id_token" not in tokens:
            raise IdentityProviderError("IdP 未返回 token", reason="no_token")
        return tokens

    async def health(self) -> bool:
        """探活：能否取到 discovery。"""
        try:
            await self._get_discovery()
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _str_or_none(value: Any) -> str | None:
    """安全转字符串（None 保持 None）。"""
    return str(value) if value is not None else None
