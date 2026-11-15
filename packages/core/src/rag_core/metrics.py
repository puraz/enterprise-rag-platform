"""Prometheus 指标：定义集中在此，业务代码只调用记录函数。

## 为什么指标定义必须集中

Prometheus 客户端在**重复注册同名指标时会抛异常**
（`Duplicated timeseries in CollectorRegistry`）。

如果允许各业务模块自己 `Counter("rag_llm_calls_total", ...)`，会立刻遇到：
- 两个模块都定义同名指标 → 启动崩溃；
- 模块被 import 两次（循环依赖、测试重载）→ 崩溃；
- 改了 label 集合 → 旧指标还在 registry 里，新定义冲突。

所以：**指标的 name/labels 只在这里定义一次**，业务代码只调
`record_llm_call(...)`。这样 label 集合变更时只有一个改动点。

## 为什么不用 `REGISTRY` 默认注册器

默认注册器是**进程级全局**。测试里反复 import/创建 app 会累积注册，
而且无法在测试间"重置"。这里用显式 registry，并在 `build_metrics()`
里创建——测试可以每次都拿一个干净的。

代价：多进程（uvicorn workers）时每个进程有自己的 registry，
需要 `PROMETHEUS_MULTIPROC_DIR` 让各进程把数据写到共享目录再汇总。
本模块支持这个模式（见 `build_metrics` 的说明）。

## 指标设计：四条 USE/RED 原则

| 指标类型 | 用途 | 本项目对应 |
|---|---|---|
| Counter（只增） | 计数 | 请求数、LLM 调用数、token 数、成本 |
| Histogram（分布） | 延迟/大小 | 请求延迟、LLM 延迟、检索延迟 |
| Gauge（可升降） | 瞬时值 | 熔断器状态、连接池使用、队列积压 |
| Info（元信息） | 版本/配置 | 构建信息 |

**不要用 Gauge 记"累计值"**（如总调用数）：进程重启就归零，
Prometheus 会用 `rate()` 算出负增长。累计量一律用 Counter。

## label 的基数问题（最常见的生产事故）

label 值会被建成独立的时间序列。给 `document_id` 或 `query` 打 label
会让时间序列数量随数据量爆炸（十万文档 = 十万序列），
Prometheus 内存被打爆。

**本模块所有 label 都是低基数**：状态、模型名、任务类型、错误码。
绝不把 ID、URL、用户输入放进 label。

需要按 ID 排查时用日志/trace，不用指标。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

from rag_core.logging import get_logger

log = get_logger(__name__)

# 延迟桶：覆盖 5ms ~ 30s。
#
# 为什么这样分：
# - 缓存命中/向量检索在 5~50ms 量级，需要细桶才能看出差异；
# - 一次完整问答（含 LLM）在 1~10s 量级；
# - P99 超出 30s 的场景必须单独看（通常是超时，不是真实延迟）。
#
# 桶太少会让 P95/P99 只能落在粗粒度边界上，"优化了 200ms"看不出来。
LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

# token 数桶：LLM 调用的输入/输出规模差异很大（几十到几万）
TOKEN_BUCKETS = (50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000)

# 成本桶（USD）：单次调用通常 1e-5 ~ 1e-1
COST_BUCKETS = (0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0)


@dataclass(slots=True)
class Metrics:
    """一组指标定义 + 记录方法。

    用类而不是模块级全局：测试可以创建独立实例（干净 registry），
    app 也可以在自己的 lifespan 里建一个。
    """

    registry: CollectorRegistry

    # ---- HTTP ----
    http_requests: Counter = field(init=False)
    http_request_duration: Histogram = field(init=False)
    http_errors: Counter = field(init=False)

    # ---- 检索 ----
    retrieval_total: Counter = field(init=False)
    retrieval_duration: Histogram = field(init=False)
    retrieval_results: Histogram = field(init=False)

    # ---- LLM ----
    llm_calls: Counter = field(init=False)
    llm_duration: Histogram = field(init=False)
    llm_tokens: Histogram = field(init=False)
    llm_cost: Histogram = field(init=False)
    llm_breaker_state: Gauge = field(init=False)

    # ---- 引用 ----
    citations_total: Counter = field(init=False)

    # ---- 限流 ----
    ratelimit_denied: Counter = field(init=False)

    # ---- Worker ----
    tasks_processed: Counter = field(init=False)
    tasks_inflight: Gauge = field(init=False)
    task_duration: Histogram = field(init=False)

    # ---- 缓存 ----
    cache_operations: Counter = field(init=False)
    cache_hit_ratio: Gauge = field(init=False)

    def __post_init__(self) -> None:
        """定义全部指标。

        **必须在 __post_init__ 里定义**（而不是类体里）：类体在 import 时
        执行，无法把 registry 传进去。放这里让每个实例绑定自己的 registry。
        """
        r = self.registry

        # ---- HTTP ----
        self.http_requests = Counter(
            "rag_http_requests_total",
            "HTTP 请求总数",
            # route 用**路由模板**（`/api/v1/documents/{doc_id}`）
            # 而不是实际路径——实际路径会让基数随文档数爆炸。
            ["method", "route", "status"],
            registry=r,
        )
        self.http_request_duration = Histogram(
            "rag_http_request_duration_seconds",
            "HTTP 请求耗时",
            ["method", "route"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.http_errors = Counter(
            "rag_http_errors_total",
            "HTTP 错误数（按业务错误码分类）",
            ["code_name"],
            registry=r,
        )

        # ---- 检索 ----
        self.retrieval_total = Counter(
            "rag_retrieval_total",
            "检索次数",
            ["strategy"],
            """strategy: vector / hybrid / with_rerank。低基数。""",
            registry=r,
        )
        self.retrieval_duration = Histogram(
            "rag_retrieval_duration_seconds",
            "检索耗时",
            ["strategy"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.retrieval_results = Histogram(
            "rag_retrieval_results_count",
            "单次检索返回的结果数",
            buckets=(1, 3, 5, 10, 20, 50, 100),
            registry=r,
        )

        # ---- LLM ----
        self.llm_calls = Counter(
            "rag_llm_calls_total",
            "LLM 调用次数",
            # task 是 ModelTaskType（低基数枚举）；model 是配置里的模型名
            # （几十个以内）。**不要加 prompt/query/user 维度**。
            #
            # is_fallback 是布尔值（2 个基数），加它是为了能算"降级比例"——
            # 没有这个 label，`LLMDegradedToFallback` 告警无从写起。
            ["task", "provider", "model", "status", "is_fallback"],
            registry=r,
        )
        self.llm_duration = Histogram(
            "rag_llm_duration_seconds",
            "LLM 调用耗时",
            ["task", "model"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.llm_tokens = Histogram(
            "rag_llm_tokens",
            "LLM token 用量",
            ["task", "model", "kind"],
            """kind: input / output / cached。""",
            buckets=TOKEN_BUCKETS,
            registry=r,
        )
        self.llm_cost = Histogram(
            "rag_llm_cost_usd",
            "单次 LLM 调用成本（USD）",
            ["task", "model"],
            buckets=COST_BUCKETS,
            registry=r,
        )
        self.llm_breaker_state = Gauge(
            "rag_llm_breaker_state",
            "熔断器状态：0=closed 1=half_open 2=open",
            ["provider"],
            registry=r,
        )

        # ---- 引用 ----
        self.citations_total = Counter(
            "rag_citations_total",
            "引用产出情况",
            # located: 定位成功 / failed；这个比值反映"引用是否靠谱"
            ["located"],
            registry=r,
        )

        # ---- 限流 ----
        self.ratelimit_denied = Counter(
            "rag_ratelimit_denied_total",
            "被限流的请求数",
            ["policy"],
            registry=r,
        )

        # ---- Worker ----
        self.tasks_processed = Counter(
            "rag_tasks_processed_total",
            "Worker 处理的任务数",
            ["task_type", "outcome"],
            """outcome: succeeded / failed / dead。""",
            registry=r,
        )
        self.tasks_inflight = Gauge(
            "rag_tasks_inflight",
            "当前正在执行的任务数",
            registry=r,
        )
        self.task_duration = Histogram(
            "rag_task_duration_seconds",
            "任务执行耗时",
            ["task_type"],
            buckets=(0.1, 0.5, 1.0, 5.0, 30.0, 60.0, 300.0, 600.0),
            registry=r,
        )

        # ---- 缓存 ----
        self.cache_operations = Counter(
            "rag_cache_operations_total",
            "缓存操作数",
            ["namespace", "result"],
            """result: hit / miss / error。""",
            registry=r,
        )
        self.cache_hit_ratio = Gauge(
            "rag_cache_hit_ratio",
            "缓存命中率（0~1，由业务代码定期更新）",
            ["namespace"],
            registry=r,
        )


def build_metrics() -> Metrics:
    """构造指标集合。

    ## 多进程模式

    若环境变量 `PROMETHEUS_MULTIPROC_DIR` 已设置（uvicorn 多 worker 场景），
    各进程会把指标写到该目录下的文件中，由 `MultiProcessCollector`
    汇总。**必须在创建指标之前设置这个环境变量**，否则指标会留在
    各进程内存里，抓取时只能看到其中一个进程的数据。

    未设置时用普通 registry（单进程/测试）。
    """
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if multiproc_dir:
        log.info(
            "metrics_multiprocess_mode",
            dir=multiproc_dir,
            note="多进程模式：指标写入共享目录后汇总",
        )
        registry = CollectorRegistry()
        # prometheus_client 的这个类没标类型，但它就是构造函数。
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
        # 注意：多进程模式下 Counter/Histogram 会自动走 multiprocess 逻辑，
        # 但 Gauge 需要显式用 `multiprocess_mode` 参数指定聚合方式
        # （见 `set_gauge` 的说明）。这里不做事后补丁——
        # 需要聚合 Gauge 的场景（如 tasks_inflight）在多进程下语义本就模糊
        # （是"单进程在跑几个"还是"全部进程合计"？），
        # 所以我们只把它当单进程观测，汇总交给 Prometheus 的 sum()。
        return Metrics(registry=registry)

    return Metrics(registry=CollectorRegistry())


def render_metrics(metrics: Metrics) -> bytes:
    """渲染成 Prometheus 文本格式（给 `/metrics` 端点用）。"""
    return generate_latest(metrics.registry)


# ---------------------------------------------------------------- 便捷记录函数


def record_llm_call(
    metrics: Metrics,
    *,
    task: str,
    provider: str,
    model: str,
    status: str,
    duration_s: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    cost_usd: float = 0.0,
    is_fallback: bool = False,
) -> None:
    """记录一次 LLM 调用（同时更新计数、延迟、token、成本四个指标）。

    集中成一个函数的原因：一次调用要更新四个指标，分散写必然有人漏。
    漏记成本指标会让"成本看板"与"对账报告"数字不一致——
    而这类不一致会让人怀疑两个数字都不可信。
    """
    metrics.llm_calls.labels(
        task=task,
        provider=provider,
        model=model,
        status=status,
        is_fallback=str(is_fallback).lower(),
    ).inc()
    metrics.llm_duration.labels(task=task, model=model).observe(duration_s)

    if input_tokens:
        metrics.llm_tokens.labels(task=task, model=model, kind="input").observe(input_tokens)
    if output_tokens:
        metrics.llm_tokens.labels(task=task, model=model, kind="output").observe(output_tokens)
    if cached_tokens:
        metrics.llm_tokens.labels(task=task, model=model, kind="cached").observe(cached_tokens)
    if cost_usd:
        metrics.llm_cost.labels(task=task, model=model).observe(cost_usd)


def record_retrieval(
    metrics: Metrics,
    *,
    strategy: str,
    duration_s: float,
    result_count: int,
) -> None:
    metrics.retrieval_total.labels(strategy=strategy).inc()
    metrics.retrieval_duration.labels(strategy=strategy).observe(duration_s)
    metrics.retrieval_results.observe(result_count)


def record_cache_op(metrics: Metrics, *, namespace: str, result: str) -> None:
    """result: hit / miss / error。"""
    metrics.cache_operations.labels(namespace=namespace, result=result).inc()


def record_task(
    metrics: Metrics,
    *,
    task_type: str,
    outcome: str,
    duration_s: float,
) -> None:
    """outcome: succeeded / failed / dead。"""
    metrics.tasks_processed.labels(task_type=task_type, outcome=outcome).inc()
    metrics.task_duration.labels(task_type=task_type).observe(duration_s)


def record_citation(metrics: Metrics, *, located: bool) -> None:
    metrics.citations_total.labels(located="located" if located else "failed").inc()


def record_ratelimit_denied(metrics: Metrics, *, policy: str) -> None:
    metrics.ratelimit_denied.labels(policy=policy).inc()


# ---------------------------------------------------------------- 熔断器状态


_BREAKER_STATE_VALUES = {"closed": 0, "half_open": 1, "open": 2}


def update_breaker_state(metrics: Metrics, *, provider: str, state: str) -> None:
    """更新熔断器状态 Gauge。

    用数值而不是字符串：Prometheus 的 Gauge 只支持数值，
    字符串得用 label（但 label 变化会产生"多条序列"，查询时要 filter，
    不如直接用数值 + 固定的阈值告警 `> 1` 表示熔断）。
    """
    value = _BREAKER_STATE_VALUES.get(state)
    if value is None:
        log.warning("unknown_breaker_state", state=state)
        return
    metrics.llm_breaker_state.labels(provider=provider).set(value)


# ---------------------------------------------------------------- 健康检查


def metrics_snapshot(metrics: Metrics) -> dict[str, Any]:
    """取关键指标的当前值（供 `/health` 或管理端展示）。

    只返回少量关键值——**不是**指标系统的替代品。
    详细指标走 `/metrics`，这里只为"运维一眼看健康度"。
    """
    registry = metrics.registry
    snapshot: dict[str, Any] = {}

    for metric in registry.collect():
        if not metric.samples:
            continue
        # 只取 Counter 的 _total 与 Gauge 的直接值，跳过 Histogram 的
        # _bucket/_sum/_count（它们数量多且需要聚合才有意义）
        for sample in metric.samples:
            name = sample.name
            if name.endswith("_total") or name.startswith("rag_llm_breaker_state"):
                snapshot[name] = sample.value

    return snapshot


__all__ = [
    "COST_BUCKETS",
    "LATENCY_BUCKETS",
    "TOKEN_BUCKETS",
    "Metrics",
    "build_metrics",
    "metrics_snapshot",
    "record_cache_op",
    "record_citation",
    "record_llm_call",
    "record_ratelimit_denied",
    "record_retrieval",
    "record_task",
    "render_metrics",
    "update_breaker_state",
]
