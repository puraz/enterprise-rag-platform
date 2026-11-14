"""熔断器：三态 + 滑动窗口。

为什么需要熔断而不是只有重试：
重试是"再试一次"，熔断是"暂时别试了"。下游彻底挂掉时，重试只会
把压力放大 N 倍并耗尽调用方的连接和信号量，让故障从下游蔓延到上游。
熔断的价值是**快速失败 + 给下游喘息时间**。

本实现刻意保持"单进程内"：
- 多副本时每个进程各自熔断，状态不共享。
  这是可接受的：熔断是保护性策略，各副本独立判断不会造成错误决策，
  只是恢复时间略有差异。
- 若要全局熔断（例如所有副本同时停调某个供应商），需要把状态放 Redis。
  那是 A13 的事；过早引入 Redis 依赖会让单测必须起 Redis。
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from rag_core.resilience.policy import BreakerPolicy


class BreakerState(StrEnum):
    """熔断器三态。"""

    CLOSED = "closed"
    """正常放行。"""

    OPEN = "open"
    """直接拒绝，不发起真实调用。"""

    HALF_OPEN = "half_open"
    """放少量试探请求，用来判断下游是否恢复。"""


@dataclass
class _CallRecord:
    """窗口内一次调用的结果。"""

    at: float
    success: bool


@dataclass
class CircuitBreaker:
    """单依赖熔断器。

    并发安全用 `asyncio.Lock` 保护状态转换。锁的范围**只覆盖状态判断
    与计数更新**，不覆盖真实调用——否则熔断器自己就变成了串行化瓶颈。
    """

    name: str
    policy: BreakerPolicy
    _state: BreakerState = BreakerState.CLOSED
    _records: deque[_CallRecord] = field(default_factory=deque)
    _opened_at: float = 0.0
    _half_open_inflight: int = 0
    _half_open_successes: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def state(self) -> BreakerState:
        """当前状态（只读，不触发转换）。

        时间驱动的 OPEN → HALF_OPEN 转换在 `allow()` 里做，
        因为需要拿锁；这里为了能被随意读取（指标、日志）不做副作用。
        """
        return self._state

    async def allow(self) -> bool:
        """判断本次调用是否放行。

        返回 False 表示应该**直接失败**，不发起真实调用。
        """
        async with self._lock:
            now = time.monotonic()

            if self._state is BreakerState.OPEN:
                if now - self._opened_at >= self.policy.open_timeout_s:
                    # 冷却结束，进入半开试探
                    self._state = BreakerState.HALF_OPEN
                    self._half_open_inflight = 0
                    self._half_open_successes = 0
                else:
                    return False

            if self._state is BreakerState.HALF_OPEN:
                if self._half_open_inflight >= self.policy.half_open_max_calls:
                    # 试探名额用满，其余请求继续快速失败
                    return False
                self._half_open_inflight += 1

            return True

    async def record_success(self) -> None:
        """记录成功。"""
        async with self._lock:
            if self._state is BreakerState.HALF_OPEN:
                self._half_open_successes += 1
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
                # 半开状态下有足够多次成功就完全恢复
                if self._half_open_successes >= self.policy.half_open_max_calls:
                    self._reset_to_closed()
                return

            self._append(now_success=True)

    async def record_failure(self) -> None:
        """记录失败（只对 RETRYABLE / FATAL 调用，见 classify）。"""
        async with self._lock:
            if self._state is BreakerState.HALF_OPEN:
                # 半开时一次失败就重新熔断：说明下游还没恢复
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
                self._open()
                return

            self._append(now_success=False)
            self._maybe_open()

    def _append(self, *, now_success: bool) -> None:
        """写入窗口记录并清理过期项。"""
        now = time.monotonic()
        self._records.append(_CallRecord(at=now, success=now_success))
        cutoff = now - self.policy.window_s
        while self._records and self._records[0].at < cutoff:
            self._records.popleft()

    def _maybe_open(self) -> None:
        """按窗口失败率决定是否熔断。"""
        total = len(self._records)
        # 样本不足时不评估：1 次失败就是 100% 失败率，会误判偶发抖动
        if total < self.policy.minimum_calls:
            return

        failures = sum(1 for record in self._records if not record.success)
        if failures / total >= self.policy.failure_rate_threshold:
            self._open()

    def _open(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = time.monotonic()

    def _reset_to_closed(self) -> None:
        self._state = BreakerState.CLOSED
        self._records.clear()
        self._half_open_inflight = 0
        self._half_open_successes = 0

    def snapshot(self) -> dict[str, object]:
        """给指标/日志用的状态快照。"""
        total = len(self._records)
        failures = sum(1 for record in self._records if not record.success)
        return {
            "name": self.name,
            "state": self._state.value,
            "window_calls": total,
            "window_failures": failures,
            "failure_rate": (failures / total) if total else 0.0,
        }
