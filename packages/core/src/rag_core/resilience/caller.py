"""ResilientCaller：所有外部依赖调用的唯一入口。

顺序刻意固定，每一步都有理由：

    信号量 → 熔断检查 → [单次超时 → 真实调用] → 记录结果 → 退避重试

为什么信号量在最外层：它是"这个依赖最多同时跑几个请求"的闸门。
放在重试内层的话，重试期间闸门会被释放又重占，实际并发数会超过上限。

为什么熔断在超时外层：熔断时**根本不发起调用**，所以不该占用超时预算，
更不该占用信号量名额（这里为了简单仍占信号量，但立即释放）。

不用 tenacity：它没有熔断、没有总预算、没有按错误分类决定是否重试，
硬套会写出"每层各配一遍参数"的散装方案，而 `docs/PROMPT.md` 要求
唯一入口。自己实现约 200 行，可控性远高于拼装。
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from rag_contracts.errors import ErrorCode
from rag_core.errors import AppError, ProviderError, ProviderTimeoutError
from rag_core.logging import get_logger
from rag_core.resilience.breaker import BreakerState, CircuitBreaker
from rag_core.resilience.classify import FailureKind, classify
from rag_core.resilience.policy import BreakerPolicy, ResiliencePolicy

log = get_logger(__name__)

T = TypeVar("T")


class CircuitOpenError(AppError):
    """熔断器打开，调用被快速拒绝。

    用 PROVIDER_UNAVAILABLE 而不是新的错误码：对调用方来说语义相同
    （依赖暂不可用，已走降级），没必要让前端多处理一种情况。
    """

    code = ErrorCode.PROVIDER_UNAVAILABLE
    default_message = "依赖服务暂时不可用"


class BudgetExhaustedError(AppError):
    """总预算耗尽。与"单次超时"不同：这表示重试也用光了时间。"""

    code = ErrorCode.PROVIDER_TIMEOUT
    default_message = "请求超时"


@dataclass
class CallStats:
    """一次 ResilientCaller 调用的执行统计。

    返回给调用方而不是只打日志：上层可据此决定"是否降级"
    （例如 2 次重试后才成功，可以给这次结果打个质量标记）。
    """

    attempts: int = 0
    retries: int = 0
    total_elapsed_s: float = 0.0
    breaker_state: str = BreakerState.CLOSED.value


class ResilientCaller:
    """统一的韧性包装器。

    每个外部依赖（每个供应商、每个下游服务）应持有**一个独立实例**：
    信号量按依赖隔离，熔断器按依赖隔离。共用一个实例会让"供应商 A 挂了"
    把"供应商 B"也熔断掉。
    """

    def __init__(
        self,
        name: str,
        policy: ResiliencePolicy | None = None,
        *,
        breaker_policy: BreakerPolicy | None = None,
    ) -> None:
        """构造韧性包装器。

        breaker_policy 可单独覆盖：个别依赖需要不同的熔断参数时，
        不必重建整个 ResiliencePolicy。不传则用 policy.breaker。
        """
        self.name = name
        self.policy = policy or ResiliencePolicy()
        self._semaphore = asyncio.Semaphore(self.policy.max_concurrency)
        self._breaker = CircuitBreaker(name=name, policy=breaker_policy or self.policy.breaker)

    @property
    def breaker(self) -> CircuitBreaker:
        """暴露熔断器，供指标采集与健康检查读取状态。"""
        return self._breaker

    async def call(
        self,
        func: Callable[[], Awaitable[T]],
        *,
        operation: str,
        idempotent: bool = True,
    ) -> tuple[T, CallStats]:
        """执行一次带韧性保护的调用。

        Args:
            func: 零参异步可调用。**必须自己携带好参数**——用 lambda 或
                functools.partial 绑定，这样重试时不会重复求值参数。
            operation: 操作名，用于日志与指标（如 "chat.completion"）。
            idempotent: 该操作是否幂等。

        **idempotent=False 时绝不重试**（无论错误类型）。这是本层的核心
        安全约束：重试"扣配额""创建文档"会产生重复副作用。宁可失败，
        也不能悄悄多扣一次钱、多建一条记录。

        返回值是 (结果, 统计)。用元组而非只返回结果，是因为上层需要知道
        "是否经过了重试"，以便决定是否给结果打降级标记。
        """
        started = time.monotonic()
        deadline = started + self.policy.total_timeout_s
        stats = CallStats()
        last_exc: BaseException | None = None

        # 非幂等操作强制单次尝试：由本层保证，不依赖调用方自觉
        max_attempts = self.policy.retry.max_attempts if idempotent else 1

        for attempt in range(1, max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BudgetExhaustedError(
                    f"{operation} 超出总预算 {self.policy.total_timeout_s}s"
                ) from last_exc

            if not await self._breaker.allow():
                stats.breaker_state = self._breaker.state.value
                stats.total_elapsed_s = time.monotonic() - started
                log.warning(
                    "circuit_open_rejected",
                    dependency=self.name,
                    operation=operation,
                    breaker_state=self._breaker.state.value,
                )
                raise CircuitOpenError(f"{self.name} 熔断中，快速失败")

            stats.attempts = attempt
            try:
                async with self._semaphore:
                    # 单次超时不能超过剩余总预算，否则一次尝试就吃光全部时间
                    attempt_timeout = min(self.policy.attempt_timeout_s, remaining)
                    async with asyncio.timeout(attempt_timeout):
                        result = await func()
            except Exception as exc:
                # 超时包装成我方异常类型。
                # 裸的 TimeoutError 会让上层无法用 AppError 统一处理，
                # 而且对外响应会暴露 "TimeoutError" 这种实现细节；
                # 更关键的是它不带错误码，前端无法据此决定是否重试。
                if isinstance(exc, TimeoutError):
                    exc = ProviderTimeoutError(f"{self.name} 在 {attempt_timeout:.2f}s 内未响应")
                last_exc = exc
                kind = classify(exc)

                if kind is FailureKind.PERMANENT:
                    # 调用方的问题：不重试，**不计入熔断**
                    await self._breaker.record_success()
                    stats.total_elapsed_s = time.monotonic() - started
                    log.info(
                        "call_failed_permanent",
                        dependency=self.name,
                        operation=operation,
                        attempt=attempt,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    # 注意必须 raise exc 而不是裸 raise：
                    # 上面可能把原始异常换成了 ProviderTimeoutError，
                    # 裸 raise 会重新抛出 except 捕获的**原始**异常
                    # （Python 的 re-raise 看的是异常上下文，不看局部变量）。
                    raise exc

                await self._breaker.record_failure()

                retryable = kind is FailureKind.RETRYABLE
                is_last = attempt >= max_attempts
                if not retryable or is_last:
                    stats.breaker_state = self._breaker.state.value
                    stats.total_elapsed_s = time.monotonic() - started
                    log.error(
                        "call_failed_final",
                        dependency=self.name,
                        operation=operation,
                        attempt=attempt,
                        attempts=max_attempts,
                        failure_kind=kind.value,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        exc_info=True,
                    )
                    raise exc

                delay = self._backoff_delay(attempt)
                # 退避时间不能把总预算吃穿
                if time.monotonic() + delay >= deadline:
                    stats.total_elapsed_s = time.monotonic() - started
                    raise BudgetExhaustedError(f"{operation} 重试退避将超出总预算") from exc

                stats.retries += 1
                log.warning(
                    "call_retrying",
                    dependency=self.name,
                    operation=operation,
                    attempt=attempt,
                    next_delay_s=round(delay, 3),
                    failure_kind=kind.value,
                    error_type=type(exc).__name__,
                )
                await asyncio.sleep(delay)
                continue

            else:
                await self._breaker.record_success()
                stats.total_elapsed_s = time.monotonic() - started
                stats.breaker_state = self._breaker.state.value
                if stats.retries:
                    log.info(
                        "call_succeeded_after_retry",
                        dependency=self.name,
                        operation=operation,
                        attempts=attempt,
                        total_elapsed_s=round(stats.total_elapsed_s, 3),
                    )
                return result, stats

        # 理论上不可达：循环内的分支都会 return 或 raise
        raise BudgetExhaustedError(f"{operation} 未完成调用")  # pragma: no cover

    def _backoff_delay(self, attempt: int) -> float:
        """计算第 attempt 次失败后的退避时长。

        full jitter：delay = random(0, min(cap, base * multiplier**(attempt-1)))
        用随机区间而不是固定值，避免所有客户端同时重试形成重试风暴。
        """
        raw = self.policy.retry.base_delay_s * (self.policy.retry.multiplier ** (attempt - 1))
        capped = min(raw, self.policy.retry.max_delay_s)
        if self.policy.retry.jitter:
            return random.uniform(0, capped)  # noqa: S311  # 非加密用途
        return capped

    def is_available(self) -> bool:
        """依赖当前是否可用（供健康检查/管理端展示）。"""
        return self._breaker.state is not BreakerState.OPEN

    def snapshot(self) -> dict[str, object]:
        """状态快照，供指标与排障。"""
        return self._breaker.snapshot()


# ---------------------------------------------------------------- 便捷包装


async def call_with_fallback[T](
    primary: ResilientCaller,
    primary_func: Callable[[], Awaitable[T]],
    *,
    operation: str,
    fallback: Callable[[], Awaitable[T]] | None = None,
    idempotent: bool = True,
) -> tuple[T, dict[str, object]]:
    """带降级的调用：主路径失败后尝试降级实现。

    返回 (结果, 元数据)。元数据里带 `degraded: bool`，
    上层据此决定是否在响应里告知用户"结果来自降级路径"——
    `docs/PROMPT.md` 要求降级时"有明确提示"，不能悄悄降级。
    """
    try:
        result, stats = await primary.call(primary_func, operation=operation, idempotent=idempotent)
        return result, {"degraded": False, "attempts": stats.attempts}
    except Exception as exc:
        if fallback is None:
            raise
        log.warning(
            "primary_failed_using_fallback",
            dependency=primary.name,
            operation=operation,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        try:
            fallback_result = await fallback()
        except Exception as fallback_exc:
            # 两条路都失败：抛主路径的异常，保留最初根因
            log.error(
                "fallback_also_failed",
                dependency=primary.name,
                operation=operation,
                error_type=type(fallback_exc).__name__,
                exc_info=True,
            )
            raise exc from fallback_exc
        return fallback_result, {"degraded": True, "primary_error": type(exc).__name__}


def provider_error_guard(
    provider_name: str,
) -> Callable[[BaseException], AppError]:
    """把供应商的原始异常转成我方 AppError。

    供应商 SDK 抛的异常里常带 endpoint、请求 ID、甚至回显的 prompt。
    统一转成 ProviderError 后，对外只显示"模型服务暂不可用"，
    细节留在日志里（且日志已脱敏）。

    这不是装饰器而是工厂：需要在 except 块里显式调用，保持控制流可见。
    """

    def convert(exc: BaseException) -> AppError:
        if isinstance(exc, AppError):
            return exc
        return ProviderError(f"{provider_name} 调用失败")

    return convert
