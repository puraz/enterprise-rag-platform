"""并发压测：给 QPS / P95 / P99 与"池耗尽点"。

## 为什么自己写而不装 wrk / locust

- **wrk**：需要 Lua 脚本造带鉴权的请求，且不支持"先登录拿 token 再压"；
- **locust**：重（自带 Web UI 与事件循环），且它的统计口径与我们要的
  "分阶段加压找到饱和点"不一致。

本项目要压的是**带认证的真实业务接口**（登录 → 上传 → 检索 → 问答），
用一个 ~200 行的 asyncio 压测器更直接：完全控制"何时加压、压哪些接口"。

## 三个必须测出来的东西

### 1. 饱和点（池耗尽点）

逐步加压（并发 5 → 10 → 20 → 50 → 100），记录每个档位的
QPS 与 P99。**QPS 不再增长而 P99 暴涨的那个档位就是饱和点**。

不测这个的话，"系统能扛多少"只能拍脑袋。

### 2. 错误类型

饱和时错误不一定是"超时"，可能是：
- `pool_timeout`（数据库连接池等不到连接）→ 需要调 `pool_size`；
- `409/429`（限流生效）→ 这是**正确行为**，不该算失败；
- `500`（真的挂了）→ 必须修。

**必须区分这三类**，否则会把"限流正常工作"当成"系统撑不住"。

### 3. 分位数而不是平均值

平均值 50ms 可能是"一半请求 1ms、一半请求 99ms"。P95/P99 才能看出
真实用户体验。**P99 才是容量规划的依据**（平均值会让人高估容量）。

## 为什么不用 `asyncio.gather` 一次性发完

那样只能测"瞬时并发"，测不到"持续负载下的饱和"。
这里用**持续请求 + 固定并发数**（每个 worker 循环发请求直到时间到），
更接近真实流量。
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyStats:
    """一组请求的延迟分布。"""

    samples: list[float] = field(default_factory=list)
    """毫秒。"""

    def percentile(self, p: float) -> float:
        """分位数（p 取 0~100）。

        用"最近秩"而不是插值：插值会造出实际不存在的延迟值，
        而容量规划要看"真实存在的最差 N%"。
        """
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = min(int(len(ordered) * p / 100), len(ordered) - 1)
        return ordered[index]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples) if self.samples else 0.0

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0


@dataclass
class StageResult:
    """一个并发档位的结果。"""

    concurrency: int
    duration_s: float
    total_requests: int
    success: int
    errors: dict[str, int] = field(default_factory=dict)
    latency: LatencyStats = field(default_factory=LatencyStats)

    @property
    def qps(self) -> float:
        return self.total_requests / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total_requests if self.total_requests else 0.0

    @property
    def error_total(self) -> int:
        return sum(self.errors.values())


@dataclass
class LoadTestReport:
    """完整压测报告。"""

    stages: list[StageResult] = field(default_factory=list)
    target: str = ""

    @property
    def saturation_stage(self) -> StageResult | None:
        """饱和点：QPS 增速明显放缓且 P99 已抬高的档位。

        判据：与上一档相比 QPS 增长 < 20%（说明加并发没换来吞吐），
        或错误率超过 1%。
        """
        previous_qps = 0.0
        for stage in self.stages:
            if stage.error_total > 0 and stage.success_rate < 0.99:
                return stage
            if previous_qps > 0 and stage.qps < previous_qps * 1.2:
                return stage
            previous_qps = stage.qps
        return None


async def run_stage(
    *,
    request_fn: Any,
    concurrency: int,
    duration_s: float,
    warmup_s: float = 1.0,
) -> StageResult:
    """跑一个并发档位。

    Args:
        request_fn: `async () -> None`，发起一次请求。
            抛出异常即视为失败（错误类型从异常类名归类）。
        concurrency: 并发 worker 数。
        duration_s: 持续时间（不含预热）。
        warmup_s: 预热时长。**必须预热**：
            JIT、连接池填充、缓存冷启动都会让前几个请求异常慢，
            混进统计会让 P99 虚高。

    ## 为什么每个 worker 循环而不是"发 N 个然后 gather"

    持续请求更接近真实流量，且能测到"连接池用尽后的排队"——
    一次性 gather 时如果并发数小于池容量，永远测不到池耗尽。
    """
    latency = LatencyStats()
    errors: dict[str, int] = {}
    success = 0
    total = 0
    stop = asyncio.Event()

    async def worker() -> None:
        nonlocal success, total
        while not stop.is_set():
            started = time.perf_counter()
            try:
                await request_fn()
            except BaseException as exc:
                elapsed = (time.perf_counter() - started) * 1000
                total += 1
                # 用异常类名归类：不同失败模式的类名不同
                # （TimeoutError / PoolTimeoutError / HTTPStatusError）
                key = type(exc).__name__
                # 限流是**正确行为**，单独归类
                message = str(exc)
                if "429" in message:
                    key = "RateLimited429"
                elif "409" in message:
                    key = "Conflict409"
                errors[key] = errors.get(key, 0) + 1
                latency.samples.append(elapsed)
            else:
                elapsed = (time.perf_counter() - started) * 1000
                total += 1
                success += 1
                latency.samples.append(elapsed)

    # ---- 预热 ----
    warmup_stop = asyncio.Event()
    warmup_workers = [
        asyncio.create_task(_warmup_worker(request_fn, warmup_stop, warmup_s))
        for _ in range(min(concurrency, 5))
    ]
    try:
        await asyncio.sleep(warmup_s)
    finally:
        warmup_stop.set()
        await asyncio.gather(*warmup_workers, return_exceptions=True)

    # 预热结束后清空（预热的数据不参与统计）
    latency.samples.clear()
    total = 0
    success = 0
    errors.clear()

    # ---- 正式压测 ----
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    started_at = time.perf_counter()
    await asyncio.sleep(duration_s)
    stop.set()
    await asyncio.gather(*workers, return_exceptions=True)
    actual_duration = time.perf_counter() - started_at

    return StageResult(
        concurrency=concurrency,
        duration_s=actual_duration,
        total_requests=total,
        success=success,
        errors=errors,
        latency=latency,
    )


async def _warmup_worker(request_fn: Any, stop: asyncio.Event, duration_s: float) -> None:
    """预热 worker：静默发请求，失败不影响压测结论。

    预热的目的只是"把连接池填满、把 JIT 走热"，失败与否无关紧要。
    显式 `contextlib.suppress` 而不是 `except: pass`：意图更清楚
    （"我知道会抛，就是要忽略"），也让 linter 不必豁免。
    """
    import contextlib

    deadline = time.perf_counter() + duration_s
    while not stop.is_set() and time.perf_counter() < deadline:
        with contextlib.suppress(Exception):
            await request_fn()


async def run_load_test(
    *,
    request_fn: Any,
    stages: list[int] | None = None,
    duration_per_stage_s: float = 5.0,
    target: str = "",
) -> LoadTestReport:
    """分档加压。

    默认档位 5/10/20/50/100：5 起步能拿到"无竞争"基线，
    100 足以把本地开发环境（PG pool 10+10）压到饱和。
    """
    report = LoadTestReport(target=target)
    for concurrency in stages or [5, 10, 20, 50, 100]:
        stage = await run_stage(
            request_fn=request_fn,
            concurrency=concurrency,
            duration_s=duration_per_stage_s,
        )
        report.stages.append(stage)

        print(
            f"  并发 {concurrency:>3}: "
            f"QPS={stage.qps:>7.1f}  "
            f"P50={stage.latency.percentile(50):>6.1f}ms  "
            f"P95={stage.latency.percentile(95):>6.1f}ms  "
            f"P99={stage.latency.percentile(99):>6.1f}ms  "
            f"成功率={stage.success_rate:>6.1%}"
            + (f"  错误={stage.errors}" if stage.errors else "")
        )

    return report


def format_report(report: LoadTestReport) -> str:
    """生成 Markdown 报告（可直接贴进验收文档）。"""
    lines = [
        "| 并发 | QPS | P50 (ms) | P95 (ms) | P99 (ms) | 成功率 | 错误 |",
        "|---|---|---|---|---|---|---|",
    ]
    for stage in report.stages:
        errors = ", ".join(f"{k}×{v}" for k, v in stage.errors.items()) if stage.errors else "—"
        lines.append(
            f"| {stage.concurrency} | {stage.qps:.1f} | "
            f"{stage.latency.percentile(50):.1f} | "
            f"{stage.latency.percentile(95):.1f} | "
            f"{stage.latency.percentile(99):.1f} | "
            f"{stage.success_rate:.1%} | {errors} |"
        )

    saturation = report.saturation_stage
    if saturation is not None:
        lines.append("")
        lines.append(
            f"**饱和点**：并发 {saturation.concurrency}（QPS {saturation.qps:.1f}，"
            f"P99 {saturation.latency.percentile(99):.1f}ms，"
            f"成功率 {saturation.success_rate:.1%}）"
        )
    else:
        lines.append("")
        lines.append("**未测到饱和点**：所有档位的 QPS 都在增长且成功率 ≥99%。")

    return "\n".join(lines)


__all__ = [
    "LatencyStats",
    "LoadTestReport",
    "StageResult",
    "format_report",
    "run_load_test",
    "run_stage",
]
