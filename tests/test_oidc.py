"""OIDC 身份提供者测试（用模拟 OIDC 服务器，无需真实 Keycloak）。

这批测试守的是**JWT 校验的安全边界**——每一条都对应一种真实的
token 伪造/重放攻击：

- `alg=none`：伪造任意用户
- 算法混淆（用公钥当 HMAC 密钥）
- 错误的 issuer / audience
- 过期 token
- kid 轮换后的缓存失效
- PKCE 的 code_verifier 不匹配
- 授权码重放
"""

from __future__ import annotations

import httpx
import pytest

from rag_api.identity.oidc import OIDCIdentityProvider
from rag_api.identity.provider import IdentityProviderError
from rag_api.testing.mock_oidc import MockOIDCServer
from rag_core.settings.identity import OIDCSettings

ISSUER = "http://oidc.test/realms/rag"
CLIENT_ID = "rag-web"


@pytest.fixture
def oidc_server() -> MockOIDCServer:
    return MockOIDCServer(issuer=ISSUER, client_id=CLIENT_ID)


@pytest.fixture
def provider(oidc_server: MockOIDCServer) -> OIDCIdentityProvider:
    """构造一个把 HTTP 请求路由到模拟服务器的 provider。

    用 `httpx.ASGITransport`（**异步原生**）而不是 `TestClient`
    + `MockTransport`：

    - `TestClient` 是同步的（内部跑自己的事件循环），在异步测试里
      调用会与外层循环冲突，报 "Event loop is closed" 或挂死；
    - `ASGITransport` 直接把 ASGI app 当传输层，无端口、无循环冲突。
    """
    app = oidc_server.build_app()
    settings = OIDCSettings(
        issuer=ISSUER,
        client_id=CLIENT_ID,
        allowed_algorithms=("RS256",),
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ISSUER)
    return OIDCIdentityProvider(settings, client=client)


# ==================================================== 正常路径


async def test_verify_valid_token(
    provider: OIDCIdentityProvider, oidc_server: MockOIDCServer
) -> None:
    """合法 token 能验出身份。"""
    token = oidc_server.issue_token(subject="user-123", email="a@example.com", name="Alice")
    identity = await provider.verify(token)

    assert identity.issuer == ISSUER
    assert identity.subject == "user-123"
    assert identity.email == "a@example.com"
    assert identity.display_name == "Alice"


async def test_identity_issuer_is_the_configured_one(
    provider: OIDCIdentityProvider, oidc_server: MockOIDCServer
) -> None:
    """`issuer` 必须是配置里的值（用于构造 (issuer, subject) 唯一键）。

    如果实现用了 token 里的 iss（而不是配置的），攻击者可以用
    "issuer 不同但格式合法"的 token 伪造身份。
    """
    token = oidc_server.issue_token(subject="u1")
    identity = await provider.verify(token)
    assert identity.issuer == ISSUER


# ==================================================== 算法攻击（核心安全）


async def test_rejects_alg_none_token(oidc_server: MockOIDCServer) -> None:
    """**`alg=none` 攻击**：无签名 token 必须拒绝。

    这是 JWT 最经典的漏洞：不校验算法的实现会接受
    `{"alg":"none"}` 的 token，等于任何人都能伪造任意用户。

    PyJWT 有算法白名单后天然拒绝，但这条测试保证我们的
    `allowed_algorithms` 配置真的生效。
    """
    app = oidc_server.build_app()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ISSUER)
    provider = OIDCIdentityProvider(
        OIDCSettings(issuer=ISSUER, client_id=CLIENT_ID, allowed_algorithms=("RS256",)),
        client=client,
    )

    forged = oidc_server.issue_token_with_alg(subject="admin", algorithm="none")
    with pytest.raises(IdentityProviderError, match="不支持的签名算法"):
        await provider.verify(forged)


async def test_rejects_hs256_when_not_allowed(oidc_server: MockOIDCServer) -> None:
    """**算法混淆攻击**：HS256 token 在只允许 RS256 时必须拒绝。

    攻击原理：如果实现用公钥当 HMAC 密钥验 HS256 签名，
    攻击者可以用公开的 JWKS 里的公钥自己签一个 token。
    """
    app = oidc_server.build_app()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ISSUER)
    provider = OIDCIdentityProvider(
        OIDCSettings(issuer=ISSUER, client_id=CLIENT_ID, allowed_algorithms=("RS256",)),
        client=client,
    )

    forged = oidc_server.issue_token_with_alg(subject="admin", algorithm="HS256")
    with pytest.raises(IdentityProviderError, match="不支持的签名算法"):
        await provider.verify(forged)


# ==================================================== claims 校验


async def test_rejects_wrong_issuer(
    provider: OIDCIdentityProvider, oidc_server: MockOIDCServer
) -> None:
    """**issuer 不匹配必须拒绝**。

    不校验 iss 的话，攻击者可以用"另一个 IdP 签的 token"
    访问本系统——只要那个 IdP 也签发给同一个 sub。
    """
    token = oidc_server.issue_token(subject="u1", issuer="http://evil.example/realms/x")
    with pytest.raises(IdentityProviderError, match="issuer"):
        await provider.verify(token)


async def test_rejects_wrong_audience(
    provider: OIDCIdentityProvider, oidc_server: MockOIDCServer
) -> None:
    """**aud 不匹配必须拒绝**。

    不校验 aud 的话，签给别的客户端（比如同租户的另一个应用）
    的 token 能被用来访问本系统。
    """
    token = oidc_server.issue_token(subject="u1", audience="some-other-client")
    with pytest.raises(IdentityProviderError, match="audience"):
        await provider.verify(token)


async def test_rejects_expired_token(
    provider: OIDCIdentityProvider, oidc_server: MockOIDCServer
) -> None:
    """过期 token 必须拒绝。"""
    token = oidc_server.issue_token(subject="u1", expires_in=-100)
    with pytest.raises(IdentityProviderError, match="过期"):
        await provider.verify(token)


async def test_rejects_garbage_token(provider: OIDCIdentityProvider) -> None:
    """乱填的字符串必须拒绝，不能崩。"""
    with pytest.raises(IdentityProviderError):
        await provider.verify("not-a-jwt-at-all")


async def test_rejects_empty_credential(provider: OIDCIdentityProvider) -> None:
    """空凭证拒绝。"""
    with pytest.raises(IdentityProviderError, match="为空"):
        await provider.verify("   ")


# ==================================================== JWKS 缓存与轮换


async def test_jwks_is_cached(provider: OIDCIdentityProvider, oidc_server: MockOIDCServer) -> None:
    """JWKS 应被缓存，不每次请求都拉。

    不缓存的话每个 API 请求都多一次 IdP 往返（几十毫秒），
    且 IdP 会被打爆。
    """
    token = oidc_server.issue_token(subject="u1")

    # 第一次：拉取 JWKS
    await provider.verify(token)
    fetched_at = provider._jwks.fetched_at

    # 第二次：应复用缓存
    await provider.verify(token)
    assert provider._jwks.fetched_at == fetched_at, "JWKS 被重复拉取"


async def test_jwks_refreshes_on_unknown_kid(
    provider: OIDCIdentityProvider, oidc_server: MockOIDCServer
) -> None:
    """**kid 未命中时强制刷新 JWKS**。

    IdP 轮换密钥后，新 token 用新 kid。如果缓存里没有就失败，
    会导致所有用户突然登录失败——而根因（密钥轮换）完全看不出来。
    """
    # 先用旧密钥验一次，建立缓存
    old_token = oidc_server.issue_token(subject="u1")
    await provider.verify(old_token)

    # IdP 轮换密钥
    oidc_server.rotate_keys()

    # 新 token 用新 kid
    new_token = oidc_server.issue_token(subject="u2")
    identity = await provider.verify(new_token)

    assert identity.subject == "u2", "密钥轮换后应能验证新 token"


# ==================================================== PKCE 流程


def test_pkce_pair_is_valid() -> None:
    """PKCE 对的 S256 计算正确。"""
    import base64
    import hashlib

    verifier, challenge = OIDCIdentityProvider.generate_pkce_pair()

    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    assert challenge == expected
    # verifier 长度应在 43-128 之间（RFC 7636）
    assert 43 <= len(verifier) <= 128


async def test_authorize_url_contains_pkce_params(
    provider: OIDCIdentityProvider,
) -> None:
    """授权 URL 必须包含 PKCE 参数与 state。"""
    _verifier, challenge = OIDCIdentityProvider.generate_pkce_pair()
    url = await provider.build_authorize_url(
        redirect_uri="http://localhost:3000/callback",
        state="random-state",
        code_challenge=challenge,
    )

    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    assert "state=random-state" in url
    assert f"client_id={CLIENT_ID}" in url
    assert "response_type=code" in url


async def test_code_exchange_succeeds_with_correct_verifier(
    provider: OIDCIdentityProvider, oidc_server: MockOIDCServer
) -> None:
    """正确的 code_verifier 能换到 token。"""
    verifier, challenge = OIDCIdentityProvider.generate_pkce_pair()
    code = oidc_server.create_auth_code(
        user={"sub": "u1", "email": "a@example.com"},
        code_challenge=challenge,
        redirect_uri="http://localhost:3000/callback",
    )

    tokens = await provider.exchange_code_for_tokens(
        code=code,
        code_verifier=verifier,
        redirect_uri="http://localhost:3000/callback",
    )

    assert "access_token" in tokens
    identity = await provider.verify(tokens["access_token"])
    assert identity.subject == "u1"


async def test_code_exchange_fails_with_wrong_verifier(
    provider: OIDCIdentityProvider, oidc_server: MockOIDCServer
) -> None:
    """**PKCE 核心保护**：verifier 不匹配时必须失败。

    这是防"授权码被拦截"的唯一机制：拦截者拿到 code 但没有 verifier，
    换不到 token。
    """
    _verifier, challenge = OIDCIdentityProvider.generate_pkce_pair()
    code = oidc_server.create_auth_code(
        user={"sub": "u1"},
        code_challenge=challenge,
        redirect_uri="http://localhost:3000/callback",
    )

    with pytest.raises(IdentityProviderError, match="授权码无效"):
        await provider.exchange_code_for_tokens(
            code=code,
            code_verifier="wrong-verifier-that-does-not-match",
            redirect_uri="http://localhost:3000/callback",
        )


async def test_code_cannot_be_reused(
    provider: OIDCIdentityProvider, oidc_server: MockOIDCServer
) -> None:
    """授权码是一次性的（重放攻击防护）。"""
    verifier, challenge = OIDCIdentityProvider.generate_pkce_pair()
    code = oidc_server.create_auth_code(
        user={"sub": "u1"},
        code_challenge=challenge,
        redirect_uri="http://localhost:3000/callback",
    )

    await provider.exchange_code_for_tokens(
        code=code, code_verifier=verifier, redirect_uri="http://localhost:3000/callback"
    )

    # 第二次用同一个 code 必须失败
    with pytest.raises(IdentityProviderError):
        await provider.exchange_code_for_tokens(
            code=code, code_verifier=verifier, redirect_uri="http://localhost:3000/callback"
        )


# ==================================================== 配置


def test_default_algorithms_exclude_unsafe() -> None:
    """**默认算法白名单不能含 HS256 或 none。**

    HS256 需要 IdP 与 RP 共享对称密钥——OIDC 里不存在这种共享；
    none 表示不签名。两者都会导致 token 可被伪造。
    """
    settings = OIDCSettings()

    assert "none" not in settings.allowed_algorithms
    assert "HS256" not in settings.allowed_algorithms
    assert "RS256" in settings.allowed_algorithms


def test_jwks_ttl_has_lower_bound() -> None:
    """JWKS 缓存 TTL 不能太短（否则每次请求都拉）。

    下限 60 秒：再短就没有缓存意义了。
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OIDCSettings(jwks_cache_ttl_s=1)
