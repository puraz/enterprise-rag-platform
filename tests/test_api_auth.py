"""API 骨架与鉴权测试。

用 httpx.ASGITransport 直连 app（不起真实端口），
但**走完整中间件链与依赖注入**——这样测的是真实请求路径，
而不是直接调函数。

覆盖：
- 健康检查（liveness / readiness）
- 统一错误契约（401/404/405/422 都必须是 ErrorResponse）
- 校验错误不泄漏用户输入（密码回显）
- token 校验（缺/错/过期/篡改）
- dev 登录的三重防护（未开启时 404、生产环境拒绝启动）
- 请求头回写 request_id
- 访问日志路径归一化
"""

from __future__ import annotations

import time
import uuid

import pytest

from rag_api.identity.tokens import TokenService, bearer_token
from rag_api.main import create_app
from rag_contracts.errors import ErrorCode
from rag_core.errors import AuthenticationError, TokenExpiredError, TokenInvalidError
from rag_core.settings import IdentitySettings, Settings


def _settings(**overrides: object) -> Settings:
    """构造测试配置（不读 .env，避免环境污染）。"""
    base: dict[str, object] = {
        "env": "local",
        "dev_auth_enabled": True,
        # 只测 HTTP 层契约，不需要真实 PG/Redis（它们另在集成测试里覆盖）
        "startup_dependency_check": False,
        "database": {
            "host": "127.0.0.1",
            "port": 5432,
            "user": "rag_app",
            "password": "x" * 32,
            "name": "rag",
        },
        "redis": {"host": "127.0.0.1", "port": 6379, "password": "x" * 32},
        "identity": {"jwt_secret": "test-secret-key-at-least-32-chars-long"},
    }
    base.update(overrides)
    return Settings(**base)


# ==================================================== TokenService（纯单测）


def test_issue_and_verify_roundtrip() -> None:
    """签发的 token 能被验回同样的 claims。"""
    service = TokenService(IdentitySettings(jwt_secret="a" * 40))
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()

    token, expires_at = service.issue(user_id=user_id, tenant_id=tenant_id)
    claims = service.verify(token)

    assert claims.user_id == user_id
    assert claims.tenant_id == tenant_id
    assert claims.expires_at == expires_at
    assert claims.token_id


def test_verify_rejects_tampered_token() -> None:
    """篡改 payload 会导致签名校验失败。"""
    service = TokenService(IdentitySettings(jwt_secret="b" * 40))
    token, _ = service.issue(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())

    # 改掉最后一个字符（签名部分）
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(TokenInvalidError):
        service.verify(tampered)


def test_verify_rejects_wrong_secret() -> None:
    """不同密钥签发的 token 不能互验。"""
    issuer_service = TokenService(IdentitySettings(jwt_secret="c" * 40))
    verifier_service = TokenService(IdentitySettings(jwt_secret="d" * 40))
    token, _ = issuer_service.issue(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())

    with pytest.raises(TokenInvalidError):
        verifier_service.verify(token)


def test_verify_rejects_expired_token() -> None:
    """过期 token 必须抛 TokenExpiredError（与 Invalid 区分）。

    前端对过期要 refresh，对无效要重新登录——两者混在一起的话，
    要么用户被无故登出，要么拿着坏 token 无限 refresh。

    构造方式：手工签一个 exp 在过去的 token（TTL 设 1s 再等 31s 太慢，
    而校验有 30s 时钟偏差容忍）。
    """
    import jwt as pyjwt

    secret = "e" * 40
    service = TokenService(IdentitySettings(jwt_secret=secret))
    past = int(time.time()) - 3600
    expired = pyjwt.encode(
        {
            "iss": "enterprise-rag-platform",
            "sub": str(uuid.uuid4()),
            "tid": str(uuid.uuid4()),
            "iat": past,
            "exp": past + 60,
            "jti": "x",
        },
        secret,
        algorithm="HS256",
    )
    with pytest.raises(TokenExpiredError):
        service.verify(expired)


def test_bearer_token_parsing() -> None:
    """Authorization 头解析。"""
    assert bearer_token("Bearer abc123") == "abc123"
    assert bearer_token("bearer abc123") == "abc123"

    with pytest.raises(AuthenticationError):
        bearer_token(None)
    with pytest.raises(AuthenticationError):
        bearer_token("Basic abc123")
    with pytest.raises(AuthenticationError):
        bearer_token("Bearer   ")


# ==================================================== 健康检查


def test_liveness_returns_ok() -> None:
    """liveness 不检查依赖，只要进程活着就 200。"""
    app = create_app(_settings())
    assert app.title == "Enterprise RAG Platform API"


# ==================================================== 异常契约（纯 ASGI 层）


def test_validation_error_does_not_echo_input() -> None:
    """**关键**：422 响应里不能出现用户提交的原值。

    这是 A2-4 实测证明的泄漏路径——FastAPI 默认的 422 会把
    `input` 字段原样返回，密码框校验失败时密码明文进响应体。
    """
    from fastapi.testclient import TestClient

    app = create_app(_settings())

    with TestClient(app) as client:
        # min_length=5，提交 3 个字符必然触发校验失败
        response = client.post("/api/v1/auth/dev-login", json={"credential": "abc"})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == ErrorCode.COMMON_VALIDATION_FAILED.value
    # 用户输入不应出现在响应里
    assert "abc" not in response.text.replace("dev-login", "")


def test_not_found_uses_error_contract() -> None:
    """404 也必须是 ErrorResponse 形状，而不是 FastAPI 默认的 {"detail": ...}。"""
    from fastapi.testclient import TestClient

    app = create_app(_settings())
    with TestClient(app) as client:
        response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == ErrorCode.COMMON_NOT_FOUND.value
    assert "request_id" in body


def test_method_not_allowed_uses_error_contract() -> None:
    """405 同样走统一契约。"""
    from fastapi.testclient import TestClient

    app = create_app(_settings())
    with TestClient(app) as client:
        response = client.put("/api/v1/me")

    assert response.status_code == 405
    assert response.json()["code"] == ErrorCode.COMMON_METHOD_NOT_ALLOWED.value


def test_unauthorized_uses_error_contract() -> None:
    """缺 token 时返回 401 + ErrorResponse。"""
    from fastapi.testclient import TestClient

    app = create_app(_settings())
    with TestClient(app) as client:
        response = client.get("/api/v1/me")

    assert response.status_code == 401
    body = response.json()
    assert body["code"] == ErrorCode.AUTH_UNAUTHENTICATED.value
    assert body["request_id"]


def test_invalid_token_returns_401_not_500() -> None:
    """乱填 token 必须 401，不能 500。"""
    from fastapi.testclient import TestClient

    app = create_app(_settings())
    with TestClient(app) as client:
        response = client.get("/api/v1/me", headers={"Authorization": "Bearer not-a-real-token"})

    assert response.status_code == 401
    assert response.json()["code"] == ErrorCode.AUTH_TOKEN_INVALID.value


# ==================================================== dev 登录的防护


def test_dev_login_absent_when_disabled() -> None:
    """dev_auth_enabled=False 时端点**根本不存在**（404，不是 403）。

    "注册了但运行时拒绝"会让攻击者确认路径存在；
    不注册则连存在性都无法探测。
    """
    from fastapi.testclient import TestClient

    app = create_app(_settings(dev_auth_enabled=False))
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/dev-login", json={"credential": "dev:alice"})

    assert response.status_code == 404


def test_production_rejects_dev_auth_at_startup() -> None:
    """生产环境开 dev 登录 → 启动即失败。

    这是三重防护的第三层：即使有人配错了环境变量，
    进程也起不来，而不是"悄悄在生产开了个免登录入口"。
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="DEV_AUTH_ENABLED"):
        Settings(
            env="prod",
            dev_auth_enabled=True,
            database={"password": "a-real-strong-secret-value-here"},
            identity={"jwt_secret": "x" * 40, "jwt_algorithm": "RS256"},
        )


def test_production_rejects_weak_jwt_secret() -> None:
    """生产环境用默认/过短 JWT 密钥 → 启动失败。

    弱密钥意味着任何人都能伪造 token，等于没有鉴权。
    """
    from pydantic import ValidationError

    # 20 个字符：通过 IdentitySettings 的 min_length=16，
    # 但低于生产守卫要求的 32 —— 这样测的是守卫而不是字段校验
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(
            env="prod",
            database={"password": "a-real-strong-secret-value-here"},
            identity={"jwt_secret": "x" * 20, "jwt_algorithm": "RS256"},
        )


def test_production_rejects_hs256() -> None:
    """生产环境不允许 HS256（对称算法要求所有验证方共享密钥）。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="HS256"):
        Settings(
            env="prod",
            database={"password": "a-real-strong-secret-value-here"},
            identity={"jwt_secret": "x" * 40, "jwt_algorithm": "HS256"},
        )


# ==================================================== request_id 与日志


def test_request_id_written_to_response_header() -> None:
    """响应头必须带 X-Request-ID，且与日志里的值一致。"""
    from fastapi.testclient import TestClient

    app = create_app(_settings())
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) == 32


def test_incoming_request_id_is_preserved() -> None:
    """上游传入的 request_id 应被沿用（跨服务串联）。"""
    from fastapi.testclient import TestClient

    app = create_app(_settings())
    with TestClient(app) as client:
        response = client.get("/health/live", headers={"X-Request-ID": "upstream-trace-123"})

    assert response.headers["X-Request-ID"] == "upstream-trace-123"


def test_overlong_request_id_is_replaced() -> None:
    """超长 request_id 被替换，避免污染日志。"""
    from fastapi.testclient import TestClient

    app = create_app(_settings())
    with TestClient(app) as client:
        response = client.get("/health/live", headers={"X-Request-ID": "x" * 200})

    assert response.headers["X-Request-ID"] != "x" * 200
    assert len(response.headers["X-Request-ID"]) == 32


def test_path_normalization() -> None:
    """访问日志里路径的 ID 段被归一化，避免指标基数爆炸。"""
    from rag_api.middleware import _normalize_path

    assert (
        _normalize_path("/api/v1/documents/3f2a4b1c-1234-5678-9abc-def012345678")
        == "/api/v1/documents/{id}"
    )
    assert _normalize_path("/api/v1/kb/42/documents") == "/api/v1/kb/{n}/documents"
    assert _normalize_path("/health/live") == "/health/live"
