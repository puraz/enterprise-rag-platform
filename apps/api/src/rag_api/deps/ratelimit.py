"""限流依赖：按用途分文件（见 AGENTS.md 的 deps 分包约定）。

## 问答为什么是双层限流

- 只限租户：一个用户能把整个租户的额度吃光（其它同事全被限）；
- 只限用户：大量小号注册能把全平台打爆。

## 为什么问答限额比普通接口低

一次问答要调 3~4 次 LLM（改写 + 判答 + 生成），成本是普通接口的
几十倍。用同一个限额会让"刷查询"变相成为"刷模型配额"。

## Redis 故障时放行而不是拒绝

限流的目的是保护后端，不是拒绝用户。Redis 挂了的时候：
- 拒绝所有请求 → 全站不可用（比被刷更糟）；
- 放行 → 后端可能被打，但至少服务着。

选后者，并记 warning（可观测到"限流失效了"）。

## 配额从配置读

限额不能写死：不同部署的模型单价与并发规模差别很大。配置见
`RAG_RATELIMIT__CHAT__*`（`rag_core.settings.ratelimit`）。
"""

from __future__ import annotations

from fastapi import Depends, Request

from rag_api.deps.app import get_settings_from_app
from rag_api.deps.auth import Principal, get_current_principal
from rag_core.logging import get_logger
from rag_core.ratelimit import RateLimiter, RateLimitPolicy, tenant_key, user_key

log = get_logger(__name__)


async def enforce_chat_ratelimit(
    request: Request,
    principal: Principal = Depends(get_current_principal),
) -> None:
    """问答的双层限流（租户级 + 用户级）。

    `429 Too Many Requests` 由 `RateLimitedError` 表达；
    `details.retry_after_s` 告诉客户端多久后可以重试。
    """
    settings = get_settings_from_app(request)
    chat_limits = settings.ratelimit.chat
    if not chat_limits.enabled:
        return

    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        # 没有 Redis 连接（例如极简测试 app）：与 Redis 故障同样的策略——放行。
        log.warning(
            "ratelimit_unavailable",
            error_type="RedisNotConfigured",
            note="未装配 Redis，本次请求放行（限流是保护而非拒绝）",
        )
        return

    metrics = getattr(request.app.state, "metrics", None)
    limiter = _get_limiter(request, redis.client)

    tenant_policy = RateLimitPolicy(
        "chat_tenant",
        rate=chat_limits.tenant_rate,
        burst=chat_limits.tenant_burst,
    )
    user_policy = RateLimitPolicy(
        "chat_user",
        rate=chat_limits.user_rate,
        burst=chat_limits.user_burst,
    )

    try:
        tenant_result = await limiter.check(
            policy=tenant_policy, key=tenant_key(principal.tenant_id)
        )
        user_result = await limiter.check(
            policy=user_policy,
            key=user_key(principal.tenant_id, principal.user_id),
        )
    except Exception as exc:
        log.warning(
            "ratelimit_unavailable",
            error_type=type(exc).__name__,
            note="Redis 不可用，本次请求放行（限流是保护而非拒绝）",
        )
        return

    blocked = None
    if not tenant_result.allowed:
        blocked = tenant_result
    elif not user_result.allowed:
        blocked = user_result

    if blocked is not None:
        if metrics is not None:
            from rag_core.metrics import record_ratelimit_denied

            record_ratelimit_denied(metrics, policy=blocked.limit_name)
        from rag_core.errors import RateLimitedError

        raise RateLimitedError(
            f"请求过于频繁，请 {blocked.retry_after_s:.1f} 秒后重试",
            details={"retry_after_s": round(blocked.retry_after_s, 1)},
        )


def _get_limiter(request: Request, client: object) -> RateLimiter:
    """取进程级限流器（首次请求时构造）。

    `RateLimiter` 构造会 `register_script`，每请求重建虽然不产生 I/O，
    但没有必要；挂到 `app.state` 上复用即可。并发首次请求会各自构造
    一次（最终只有一个留下），无害。
    """
    limiter = getattr(request.app.state, "chat_rate_limiter", None)
    if limiter is None:
        limiter = RateLimiter(client)
        request.app.state.chat_rate_limiter = limiter
    return limiter


__all__ = ["enforce_chat_ratelimit"]
