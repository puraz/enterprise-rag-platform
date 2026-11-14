"""韧性层：所有外部依赖调用的唯一入口。

使用约定：
- 每个外部依赖一个独立 `ResilientCaller` 实例（信号量与熔断按依赖隔离）；
- 业务代码**禁止**自己写 `for ... retry`、`asyncio.sleep` 重试、
  自建 Semaphore 限并发；
- 非幂等操作必须 `idempotent=False`，本层会强制只尝试一次。
"""

from rag_core.resilience.breaker import BreakerState, CircuitBreaker
from rag_core.resilience.caller import (
    BudgetExhaustedError,
    CallStats,
    CircuitOpenError,
    ResilientCaller,
    call_with_fallback,
)
from rag_core.resilience.classify import FailureKind, classify
from rag_core.resilience.policy import BreakerPolicy, ResiliencePolicy, RetryPolicy

__all__ = [
    "BreakerPolicy",
    "BreakerState",
    "BudgetExhaustedError",
    "CallStats",
    "CircuitBreaker",
    "CircuitOpenError",
    "FailureKind",
    "ResiliencePolicy",
    "ResilientCaller",
    "RetryPolicy",
    "call_with_fallback",
    "classify",
]
