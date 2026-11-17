"""问答限流依赖的守卫测试。

## 为什么单独测这个依赖

限流是安全控制，但它的失败模式很安静：没有挂到路由上、配置读错、
Redis 不可用时直接抛异常，都会让"限流"名存实亡。这里锁住三件事：

1. `/chat` 路由**真的声明了**这个依赖（防止重构时丢掉）；
2. 配额耗尽时抛出 `RateLimitedError`（429）；
3. 关闭开关 / 没有 Redis 时走 fail-open，不阻断服务。
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

import pytest
from fastapi.routing import APIRoute

from rag_api.deps.auth import Principal
from rag_api.deps.ratelimit import enforce_chat_ratelimit
from rag_core.errors import RateLimitedError
from rag_core.ratelimit import RateLimitResult
from rag_core.settings import ChatRateLimitSettings, RateLimitSettings, Settings


class _State:
    def __init__(self, settings: Settings, *, redis: object | None) -> None:
        self.settings = settings
        self.redis = redis
        self.metrics = None


class _App:
    def __init__(self, state: _State) -> None:
        self.state = state


class _Request:
    """只需要 `.app.state` 的最小请求替身。"""

    def __init__(self, state: _State) -> None:
        self.app = _App(state)


class _RedisStub:
    """`RateLimiter` 构造需要 `.client`；本测试用预置 limiter，不会真的用它。"""

    client: Any = object()


class _DenyingLimiter:
    async def check(self, *, policy: Any, key: str) -> RateLimitResult:
        return RateLimitResult(
            allowed=False,
            remaining=0,
            retry_after_ms=1500,
            limit_name=policy.name,
        )


class _ExplodingLimiter:
    async def check(self, *, policy: Any, key: str) -> RateLimitResult:
        raise AssertionError("限流被关闭 / Redis 缺失时不该调用 limiter")


def _settings(*, enabled: bool = True) -> Settings:
    return Settings(
        ratelimit=RateLimitSettings(
            chat=ChatRateLimitSettings(
                enabled=enabled,
                tenant_rate=20.0,
                tenant_burst=40,
                user_rate=2.0,
                user_burst=5,
            )
        )
    )


def _principal() -> Principal:
    return Principal(
        user_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        token_id="test-token",
    )


def _request(*, enabled: bool = True, redis: object | None = _RedisStub()) -> _Request:
    state = _State(_settings(enabled=enabled), redis=redis)
    state.chat_rate_limiter = _ExplodingLimiter()  # type: ignore[attr-defined]
    return _Request(state)


def test_chat_route_declares_ratelimit_dependency() -> None:
    """重构路由时不能把限流依赖丢掉。"""
    from rag_api.api.v1.chat import router

    chat_route = next(
        route for route in router.routes if isinstance(route, APIRoute) and route.path == "/chat"
    )
    assert any(
        dependency.dependency is enforce_chat_ratelimit for dependency in chat_route.dependencies
    )


async def test_denies_when_quota_exhausted() -> None:
    """配额耗尽 → `RateLimitedError`（HTTP 429）且带可重试时间。"""
    request = _request()
    request.app.state.chat_rate_limiter = _DenyingLimiter()  # type: ignore[attr-defined]

    with pytest.raises(RateLimitedError) as exc_info:
        await enforce_chat_ratelimit(request, _principal())  # type: ignore[arg-type]

    assert exc_info.value.http_status == 429
    assert exc_info.value.details["retry_after_s"] == 1.5


async def test_disabled_switch_skips_limiter() -> None:
    """显式关闭（本机调试/压测基线）时不触碰 Redis。"""
    request = _request(enabled=False)
    await enforce_chat_ratelimit(request, _principal())  # type: ignore[arg-type]


async def test_missing_redis_fails_open() -> None:
    """没有 Redis 时与 Redis 故障同策略：放行，不阻断问答。"""
    request = _request(redis=None)
    await enforce_chat_ratelimit(request, _principal())  # type: ignore[arg-type]


async def test_rate_limit_error_sets_retry_after_header() -> None:
    """429 必须带标准 `Retry-After` 头（OpenAPI 里已这样声明）。"""
    from rag_api.errors_handlers import app_error_handler

    class _URL:
        path = "/api/v1/chat"

    class _HandlerRequest:
        method = "POST"
        url = _URL()
        scope: ClassVar[dict[str, Any]] = {"state": {"request_id": "test-request"}}

    response = await app_error_handler(  # type: ignore[arg-type]
        _HandlerRequest(),
        RateLimitedError("请求过于频繁", details={"retry_after_s": 1.5}),
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "2"
