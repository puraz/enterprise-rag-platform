"""故障注入演示：证明韧性层真的能保护下游，而不只是有代码。

三个场景：
1. 偶发失败 → 重试后成功；
2. 持续故障 → 熔断打开，后续请求**不再触达下游**；
3. 客户端错误刷屏 → 熔断**不**打开（防攻击者 DoS）。

用法：uv run python scripts/demo_resilience.py
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from rag_core.errors import ProviderError, ValidationFailedError
from rag_core.logging import get_logger, setup_logging
from rag_core.resilience import (
    BreakerPolicy,
    ResiliencePolicy,
    ResilientCaller,
    RetryPolicy,
)
from rag_core.settings import Settings

log = get_logger("demo")

# 小窗口熔断策略：让演示在几次请求内就能看到状态变化
FAST_BREAKER = BreakerPolicy(
    window_s=60.0,
    minimum_calls=4,
    failure_rate_threshold=0.5,
    open_timeout_s=30.0,
)


async def scenario_transient() -> None:
    print("\n" + "=" * 62)
    print("场景 1：偶发失败 → 重试后成功")
    print("=" * 62)

    caller = ResilientCaller(
        "flaky-provider",
        ResiliencePolicy(
            attempt_timeout_s=1.0,
            total_timeout_s=5.0,
            retry=RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.02),
        ),
    )
    state = {"n": 0}

    async def flaky() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise ProviderError("provider hiccup")
        return "answer"

    result, stats = await caller.call(flaky, operation="chat")
    print(f"  结果: {result}")
    print(f"  尝试次数: {stats.attempts}，重试: {stats.retries}")
    print(f"  总耗时: {stats.total_elapsed_s * 1000:.1f}ms")


async def scenario_outage() -> None:
    print("\n" + "=" * 62)
    print("场景 2：持续故障 → 熔断打开，下游不再被打")
    print("=" * 62)

    caller = ResilientCaller(
        "down-provider",
        ResiliencePolicy(
            attempt_timeout_s=1.0,
            total_timeout_s=5.0,
            max_concurrency=5,
            retry=RetryPolicy(max_attempts=1),
        ),
        breaker_policy=FAST_BREAKER,
    )
    downstream_calls = {"n": 0}

    async def down() -> str:
        downstream_calls["n"] += 1
        raise ProviderError("connection refused")

    print(f"  {'请求':<6}{'结果':<22}{'是否触达下游'}")
    for i in range(8):
        before = downstream_calls["n"]
        try:
            await caller.call(down, operation="chat")
            outcome = "成功"
        except Exception as exc:
            outcome = type(exc).__name__
        reached = downstream_calls["n"] - before
        print(f"  #{i + 1:<5}{outcome:<22}{'是' if reached else '否（快速失败）'}")

    print(f"\n  熔断状态: {caller.breaker.state.value}")
    print(f"  下游实际被调用 {downstream_calls['n']} 次（若无熔断应为 8 次）")


async def scenario_client_errors() -> None:
    print("\n" + "=" * 62)
    print("场景 3：客户端错误刷屏 → 熔断不打开（防攻击者 DoS）")
    print("=" * 62)

    caller = ResilientCaller(
        "api-endpoint",
        ResiliencePolicy(
            attempt_timeout_s=1.0,
            total_timeout_s=5.0,
            retry=RetryPolicy(max_attempts=1),
        ),
        breaker_policy=FAST_BREAKER,
    )

    async def bad_request() -> str:
        raise ValidationFailedError("非法参数")

    for _ in range(20):
        # 预期内的客户端错误，吞掉继续刷——这正是要模拟的攻击者行为
        with contextlib.suppress(ValidationFailedError):
            await caller.call(bad_request, operation="query")

    print(f"  连续 20 次 422 错误后，熔断状态: {caller.breaker.state.value}")
    print("  结论: 熔断保持 closed —— 攻击者无法通过刷非法请求把服务熔断")


async def main() -> None:
    setup_logging(Settings(env="local", log_format="console", log_level="ERROR"))
    logging.getLogger().setLevel(logging.ERROR)  # demo 里只保留 ERROR，避免刷屏

    await scenario_transient()
    await scenario_outage()
    await scenario_client_errors()
    print("\n" + "=" * 62)
    print("演示完毕")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
