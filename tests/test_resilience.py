"""韧性层测试。

重点不是覆盖率，而是锁死那些"看起来能用、出事时才发现是错的"行为：

- 非幂等操作被重试 → 重复扣费/重复建单
- 4xx 被计入熔断 → 攻击者用 401 打挂服务
- 无抖动的退避 → 重试风暴
- 单次超时不受总预算约束 → 一次尝试吃光全部时间
- 熔断打开后仍发起真实调用 → 保护失效
"""

from __future__ import annotations

import asyncio

import pytest

from rag_core.errors import (
    AppError,
    NotFoundError,
    ProviderAuthError,
    ProviderError,
    ProviderInvalidResponseError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    QuotaExceededError,
    ValidationFailedError,
)
from rag_core.resilience import (
    BreakerPolicy,
    BreakerState,
    BudgetExhaustedError,
    CircuitBreaker,
    CircuitOpenError,
    FailureKind,
    ResiliencePolicy,
    ResilientCaller,
    RetryPolicy,
    call_with_fallback,
    classify,
)

# ==================================================== 错误分类（最关键的规则）


@pytest.mark.parametrize(
    "error",
    [
        ProviderError(),
        ProviderTimeoutError(),
        ProviderRateLimitedError(),
        TimeoutError(),
        ConnectionResetError(),
        ConnectionAbortedError(),
    ],
)
def test_transient_failures_are_retryable(error: BaseException) -> None:
    """瞬时故障可重试，且计入熔断。"""
    assert classify(error) is FailureKind.RETRYABLE


@pytest.mark.parametrize(
    "error",
    [
        ValidationFailedError(),
        NotFoundError(),
        QuotaExceededError(),
    ],
)
def test_permanent_failures_are_not_retryable(error: BaseException) -> None:
    """调用方/业务状态问题不重试。

    注意 ProviderAuthError 不在这里：它虽然语义上"重试无用"，
    但属于 FATAL（我方配置故障，要熔断 + 告警），见下方专门用例。
    """
    assert classify(error) is FailureKind.PERMANENT


def test_provider_auth_failure_is_fatal_not_retryable() -> None:
    """供应商密钥失效必须归为 FATAL：不重试，但要熔断。

    这是最容易搞错的一条：它是 DEPENDENCY 分类（技术上属依赖故障），
    但重试一万次也不会好——我们的 key 就是错的。必须立刻熔断止损。

    反过来如果按分类直接判成 RETRYABLE，会变成"每 0.2s 打一次供应商
    认证接口"，既浪费又可能触发对方风控。
    """
    assert classify(ProviderAuthError()) is FailureKind.FATAL


def test_provider_invalid_response_is_fatal() -> None:
    """供应商协议漂移：重试无用，但要熔断并告警。"""
    assert classify(ProviderInvalidResponseError()) is FailureKind.FATAL


def test_unknown_exception_is_fatal() -> None:
    """未识别异常保守处理：不重试、计入熔断、留完整堆栈。

    判成 RETRYABLE 会让未知 bug 触发大量重试，把一个 bug 放大成故障。
    """
    assert classify(RuntimeError("没见过这个错")) is FailureKind.FATAL


# ==================================================== 非幂等保护


def test_non_idempotent_call_never_retries() -> None:
    """**本层最重要的安全约束**：非幂等操作只尝试一次。

    即使错误本身是可重试的（限流），也不许重试——"扣配额""创建文档"
    重试会产生重复副作用。宁可失败让用户重来，也不能悄悄多扣一次钱。
    """
    policy = ResiliencePolicy(
        retry=RetryPolicy(max_attempts=5, base_delay_s=0.001, max_delay_s=0.002),
    )
    caller = ResilientCaller("deduct-quota", policy)
    calls = {"n": 0}

    async def always_rate_limited() -> str:
        calls["n"] += 1
        raise ProviderRateLimitedError("busy")

    async def scenario() -> None:
        with pytest.raises(AppError):
            await caller.call(always_rate_limited, operation="quota.deduct", idempotent=False)

    asyncio.run(scenario())
    assert calls["n"] == 1, f"非幂等操作被重试了 {calls['n']} 次"


def test_idempotent_call_does_retry() -> None:
    """幂等操作按策略重试（对照组，证明上一条不是靠"永远不重试"实现的）。"""
    policy = ResiliencePolicy(
        retry=RetryPolicy(max_attempts=3, base_delay_s=0.001, max_delay_s=0.002),
    )
    caller = ResilientCaller("search", policy)
    calls = {"n": 0}

    async def eventually_ok() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderRateLimitedError("busy")
        return "ok"

    async def scenario() -> str:
        result, stats = await caller.call(eventually_ok, operation="search", idempotent=True)
        assert stats.attempts == 3
        return result

    assert asyncio.run(scenario()) == "ok"
    assert calls["n"] == 3


# ==================================================== 熔断器行为


def _fast_breaker_policy() -> BreakerPolicy:
    """小窗口策略，便于在测试里快速触发熔断。"""
    return BreakerPolicy(
        window_s=60.0,
        minimum_calls=4,
        failure_rate_threshold=0.5,
        open_timeout_s=0.05,
        half_open_max_calls=2,
    )


def test_breaker_does_not_open_below_minimum_calls() -> None:
    """样本不足时不熔断。

    1 次失败 = 100% 失败率。没有最小样本量的话，任何偶发错误都会
    误开熔断，变成"时不时全站 503"这种极难复现的故障。
    """
    breaker = CircuitBreaker("dep", _fast_breaker_policy())

    async def scenario() -> None:
        # 3 次失败 < minimum_calls=4，不熔断
        for _ in range(3):
            assert await breaker.allow() is True
            await breaker.record_failure()

    asyncio.run(scenario())
    assert breaker.state is BreakerState.CLOSED


def test_breaker_opens_after_threshold() -> None:
    """达到失败率阈值后打开。"""
    breaker = CircuitBreaker("dep", _fast_breaker_policy())

    async def scenario() -> None:
        for _ in range(4):
            await breaker.allow()
            await breaker.record_failure()

    asyncio.run(scenario())
    assert breaker.state is BreakerState.OPEN


def test_open_breaker_rejects_without_calling_downstream() -> None:
    """**熔断的核心价值**：打开后不给下游发请求。

    如果只是"记录失败率"但仍然转发请求，那就只是个监控指标，
    起不到保护作用——下游已经挂了，我们还在猛打。
    """
    policy = ResiliencePolicy(
        retry=RetryPolicy(max_attempts=1),
        breaker=_fast_breaker_policy(),
    )
    caller = ResilientCaller("dep", policy)
    calls = {"n": 0}

    async def always_fail() -> str:
        calls["n"] += 1
        raise ProviderError("down")

    async def scenario() -> int:
        # 打满 minimum_calls 让熔断打开
        for _ in range(4):
            with pytest.raises(AppError):
                await caller.call(always_fail, operation="probe")

        # **显式断言熔断已打开**：不能只依赖"调用次数够了就会打开"这个假设。
        # 不显式断言的话，若哪次窗口清理把记录清掉了，测试会以
        # "CircuitOpenError 没抛出"的形式失败，看不出真正原因。
        assert caller.breaker.state is BreakerState.OPEN

        before = calls["n"]
        # 熔断已打开，后续调用应快速失败且不再触达下游
        with pytest.raises(CircuitOpenError):
            await caller.call(always_fail, operation="probe")
        return calls["n"] - before

    extra_calls = asyncio.run(scenario())
    assert extra_calls == 0, "熔断打开后仍然发起了真实调用"


def test_client_errors_do_not_trip_breaker() -> None:
    """**安全关键**：4xx 类错误不能把熔断器打开。

    否则攻击者只需连发 100 个非法请求（401/422），就能让熔断器打开，
    使所有正常用户被拒——这等于给攻击者一个免费的 DoS 开关。
    """
    policy = ResiliencePolicy(
        retry=RetryPolicy(max_attempts=1),
        breaker=_fast_breaker_policy(),
    )
    caller = ResilientCaller("dep", policy)

    async def invalid_request() -> str:
        raise ValidationFailedError()

    async def scenario() -> None:
        for _ in range(20):
            with pytest.raises(ValidationFailedError):
                await caller.call(invalid_request, operation="probe")

    asyncio.run(scenario())
    assert caller.breaker.state is BreakerState.CLOSED, "客户端错误把熔断器打开了"


def test_breaker_recovers_through_half_open() -> None:
    """熔断后能通过半开试探恢复。

    不能恢复的熔断器等价于永久下线：下游恢复了我们也永远拒绝服务。
    """
    breaker = CircuitBreaker("dep", _fast_breaker_policy())

    async def scenario() -> None:
        for _ in range(4):
            await breaker.allow()
            await breaker.record_failure()
        assert breaker.state is BreakerState.OPEN

        # 冷却结束，进入半开
        await asyncio.sleep(0.06)
        assert await breaker.allow() is True
        assert breaker.state is BreakerState.HALF_OPEN

        # 半开下成功足够多次 → 恢复
        await breaker.record_success()
        await breaker.record_success()
        assert breaker.state is BreakerState.CLOSED

    asyncio.run(scenario())


def test_half_open_failure_reopens_immediately() -> None:
    """半开时一次失败就重新熔断：说明下游还没恢复。

    若允许半开下失败后继续放行，等于没熔断——故障期间会持续打下游。
    """
    breaker = CircuitBreaker("dep", _fast_breaker_policy())

    async def scenario() -> None:
        for _ in range(4):
            await breaker.allow()
            await breaker.record_failure()

        await asyncio.sleep(0.06)
        assert await breaker.allow() is True
        await breaker.record_failure()
        assert breaker.state is BreakerState.OPEN

    asyncio.run(scenario())


def test_half_open_limits_probe_count() -> None:
    """半开时只放行有限个试探请求，其余快速失败。"""
    breaker = CircuitBreaker("dep", _fast_breaker_policy())

    async def scenario() -> list[bool]:
        for _ in range(4):
            await breaker.allow()
            await breaker.record_failure()
        await asyncio.sleep(0.06)

        # half_open_max_calls=2，第 3 个应被拒
        return [await breaker.allow() for _ in range(3)]

    allowed = asyncio.run(scenario())
    assert allowed == [True, True, False]


# ==================================================== 超时与预算


def test_attempt_timeout_triggers() -> None:
    """单次尝试超时会被中断（不是无限等）。"""
    policy = ResiliencePolicy(
        attempt_timeout_s=0.05,
        total_timeout_s=1.0,
        retry=RetryPolicy(max_attempts=1),
    )
    caller = ResilientCaller("slow", policy)

    async def too_slow() -> str:
        await asyncio.sleep(1.0)
        return "never"

    async def scenario() -> None:
        with pytest.raises(AppError):
            await caller.call(too_slow, operation="slow")

    asyncio.run(scenario())


def test_total_budget_bounds_all_attempts() -> None:
    """总预算限制所有尝试的总耗时。

    没有总预算的话：3 次 × 30s = 90s，用户早关页面了，而连接和
    信号量被占住 90s，一个小故障会被放大成资源耗尽。
    """
    policy = ResiliencePolicy(
        attempt_timeout_s=0.1,
        total_timeout_s=0.25,
        retry=RetryPolicy(max_attempts=10, base_delay_s=0.01, max_delay_s=0.02),
    )
    caller = ResilientCaller("slow", policy)

    async def always_slow() -> str:
        await asyncio.sleep(1.0)
        return "never"

    async def scenario() -> float:
        import time

        started = time.monotonic()
        with pytest.raises(AppError):
            await caller.call(always_slow, operation="slow")
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert elapsed < 0.6, f"总预算失效，耗时 {elapsed:.2f}s"


def test_budget_exhausted_when_backoff_exceeds_deadline() -> None:
    """退避等待会超出总预算时应立刻放弃，不再空等。"""
    policy = ResiliencePolicy(
        attempt_timeout_s=0.05,
        total_timeout_s=0.12,
        retry=RetryPolicy(max_attempts=5, base_delay_s=0.2, max_delay_s=0.2, jitter=False),
    )
    caller = ResilientCaller("slow", policy)

    async def always_fail() -> str:
        raise ProviderError("down")

    async def scenario() -> None:
        with pytest.raises((BudgetExhaustedError, AppError)):
            await caller.call(always_fail, operation="probe")

    asyncio.run(scenario())


# ==================================================== 并发闸门


def test_semaphore_limits_concurrency() -> None:
    """信号量必须把在途请求数压在 max_concurrency 以内。

    这层保护是给供应商配额用的：并发超限会直接触发对方限流，
    表现为"平时好好的，一有流量就全 429"。
    """
    policy = ResiliencePolicy(
        attempt_timeout_s=2.0,
        total_timeout_s=5.0,
        max_concurrency=2,
        retry=RetryPolicy(max_attempts=1),
    )
    caller = ResilientCaller("limited", policy)
    inflight = {"current": 0, "peak": 0}

    async def tracked() -> str:
        inflight["current"] += 1
        inflight["peak"] = max(inflight["peak"], inflight["current"])
        await asyncio.sleep(0.02)
        inflight["current"] -= 1
        return "ok"

    async def scenario() -> None:
        await asyncio.gather(*(caller.call(tracked, operation="probe") for _ in range(8)))

    asyncio.run(scenario())
    assert inflight["peak"] <= 2, f"并发超限：峰值 {inflight['peak']}"


def test_semaphore_isolated_between_dependencies() -> None:
    """不同依赖的信号量互相隔离。

    共用一个实例会让"供应商 A 挂起"占满信号量，把"供应商 B"也拖死。
    """
    policy = ResiliencePolicy(
        attempt_timeout_s=2.0,
        total_timeout_s=5.0,
        max_concurrency=1,
        retry=RetryPolicy(max_attempts=1),
    )
    dep_a = ResilientCaller("a", policy)
    dep_b = ResilientCaller("b", policy)
    order: list[str] = []

    async def slow_a() -> str:
        await asyncio.sleep(0.05)
        order.append("a")
        return "a"

    async def fast_b() -> str:
        order.append("b")
        return "b"

    async def scenario() -> None:
        await asyncio.gather(
            dep_a.call(slow_a, operation="a"),
            dep_b.call(fast_b, operation="b"),
        )

    asyncio.run(scenario())
    # b 不受 a 的信号量限制，应该先完成
    assert order == ["b", "a"]


# ==================================================== 退避策略


def test_backoff_has_jitter() -> None:
    """退避必须带随机性，否则同时失败的客户端会同步重试。

    无抖动时 1000 个客户端在 1s 后同时重试，第二波把刚恢复的下游再打挂——
    这就是重试风暴。抖动把重试摊平到整个区间。
    """
    policy = ResiliencePolicy(retry=RetryPolicy(base_delay_s=1.0, max_delay_s=1.0, jitter=True))
    caller = ResilientCaller("dep", policy)

    delays = {caller._backoff_delay(1) for _ in range(50)}
    assert len(delays) > 1, "退避没有随机性，所有客户端会同步重试"
    assert all(0 <= value <= 1.0 for value in delays)


def test_backoff_grows_exponentially_with_cap() -> None:
    """退避时长随尝试次数增长，但不超过上限。"""
    policy = ResiliencePolicy(
        retry=RetryPolicy(base_delay_s=0.1, max_delay_s=0.4, multiplier=2.0, jitter=False)
    )
    caller = ResilientCaller("dep", policy)

    assert caller._backoff_delay(1) == pytest.approx(0.1)
    assert caller._backoff_delay(2) == pytest.approx(0.2)
    assert caller._backoff_delay(3) == pytest.approx(0.4)
    # 第 4 次会算到 0.8，被 cap 压回 0.4
    assert caller._backoff_delay(4) == pytest.approx(0.4)


# ==================================================== 降级链


def test_fallback_used_when_primary_fails() -> None:
    """主路径失败时走降级，并标记 degraded=True。

    degraded 标记必须回传：`docs/PROMPT.md` 要求降级时"有明确提示"，
    悄悄降级会让用户以为拿到的是完整结果。
    """
    policy = ResiliencePolicy(retry=RetryPolicy(max_attempts=1))
    primary = ResilientCaller("strong-model", policy)

    async def primary_call() -> str:
        raise ProviderError("strong model down")

    async def fallback_call() -> str:
        return "weak-model-result"

    async def scenario() -> tuple[str, dict[str, object]]:
        return await call_with_fallback(
            primary, primary_call, operation="chat", fallback=fallback_call
        )

    result, meta = asyncio.run(scenario())
    assert result == "weak-model-result"
    assert meta["degraded"] is True


def test_fallback_failure_raises_primary_error() -> None:
    """两条路都失败时，抛主路径异常（保留最初根因）。"""
    policy = ResiliencePolicy(retry=RetryPolicy(max_attempts=1))
    primary = ResilientCaller("strong-model", policy)

    async def primary_call() -> str:
        raise ProviderTimeoutError("primary timeout")

    async def fallback_call() -> str:
        raise ProviderError("fallback also down")

    async def scenario() -> None:
        await call_with_fallback(primary, primary_call, operation="chat", fallback=fallback_call)

    with pytest.raises(ProviderTimeoutError):
        asyncio.run(scenario())


def test_no_fallback_reraises_original() -> None:
    """没配降级时直接抛原异常。"""
    policy = ResiliencePolicy(retry=RetryPolicy(max_attempts=1))
    caller = ResilientCaller("dep", policy)

    async def fails() -> str:
        raise ProviderError("down")

    async def scenario() -> None:
        await call_with_fallback(caller, fails, operation="chat")

    with pytest.raises(ProviderError):
        asyncio.run(scenario())


# ==================================================== 统计与可观测


def test_stats_report_attempts_and_retries() -> None:
    """统计要能反映重试次数，供上层判断结果质量。"""
    policy = ResiliencePolicy(
        retry=RetryPolicy(max_attempts=3, base_delay_s=0.001, max_delay_s=0.002)
    )
    caller = ResilientCaller("dep", policy)
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderRateLimitedError("busy")
        return "ok"

    async def scenario() -> tuple[str, object]:
        return await caller.call(flaky, operation="probe")

    _, stats = asyncio.run(scenario())
    assert stats.attempts == 3  # type: ignore[attr-defined]
    assert stats.retries == 2  # type: ignore[attr-defined]


def test_snapshot_exposes_breaker_state() -> None:
    """快照能被指标系统读取（熔断状态是必需的告警信号）。"""
    caller = ResilientCaller("dep", ResiliencePolicy())
    snapshot = caller.snapshot()
    assert snapshot["name"] == "dep"
    assert snapshot["state"] == "closed"
    assert "failure_rate" in snapshot
