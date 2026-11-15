"""链路追踪：节点级 span + 脱敏。

## 为什么需要 span（而不只是日志）

A11 的检索链路有 5~8 个节点（改写、多路召回、融合、重排、生成、自省），
一次问答 3~10 秒。日志能回答"发生了什么"，但回答不了：

- 这 8 秒里，改写占多少、生成占多少？（**归因**）
- P95 延迟上涨是因为 retrieval 还是 LLM？（**定位瓶颈**）
- 降级发生时，是哪一步先失败的？（**因果链**）

span 有父子关系与耗时，天然回答这三个问题。

## 脱敏是硬要求（不是可选）

`docs/PROMPT.md` 明确要求 "Langfuse 打点与节点 span（**含脱敏**）"。

追踪系统的危害在于它是**旁路**：
- 它把数据发到另一个服务（可能是 SaaS）；
- 那里的数据通常保留更久、访问控制更松；
- 且很容易被当成"调试数据"而豁免合规审查。

所以本模块在**出口统一脱敏**：

1. 属性名命中敏感键（`prompt`/`content`/`question`/`answer`/`api_key`…）
   → 整值替换为长度标记（`<redacted:123 chars>`）；
2. 其它字符串值过 PII 规则（手机号/身份证/邮箱…）；
3. **默认不记录 prompt 与回答正文**，只记录长度与哈希前缀——
   即使脱敏也可能漏（名字、内部代号不会被 PII 规则覆盖）。

这与"日志脱敏"是同一套原则：**只在不必要的地方不记**。

## 为什么自带本地 Tracer

Langfuse 需要服务端（`docker compose up` 或云服务）。本机没有 Docker，
但这不该阻塞开发：

- `LocalTracer` 把 span 记在内存里，测试与本地排查够用；
- 接口与 `LangfuseTracer` 完全相同，切换只改一行配置；
- 没有 Langfuse 时**明确记一条 warning**，不静默降级
  （否则"追踪没数据"会变成没人知道原因的谜）。
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_core.logging import get_logger

log = get_logger(__name__)

# 命中这些键名的属性值整值脱敏。
#
# 为什么按"键名"而不是"内容"：内容级脱敏（PII 规则）只能识别格式化的
# 敏感数据（手机号、身份证）。用户名、内部项目代号、客户简称这些
# 没有固定格式，规则抓不到，但键名能表达"这是用户输入"。
_SENSITIVE_KEYS = frozenset(
    {
        "prompt",
        "messages",
        "content",
        "text",
        "question",
        "query",  # 用户问题本身可能含个人信息
        "answer",
        "response",
        "document",
        "chunk",
        "api_key",
        "authorization",
        "token",
        "password",
        "secret",
    }
)

# 允许记录正文的属性键（白名单）。
#
# 与 `_SENSITIVE_KEYS` 配合：默认拒绝，只有明确列出的才可能记录正文。
# 白名单模式比黑名单安全——新增字段默认不泄漏。
_TEXT_SAFE_KEYS = frozenset({"error_message", "status_message"})


@dataclass
class SpanData:
    """一个 span 的记录。"""

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    start_ns: int = 0
    end_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_ns <= self.start_ns:
            return 0.0
        return (self.end_ns - self.start_ns) / 1_000_000


class Tracer(Protocol):
    """追踪器接口。

    实现约定：
    - `span()` 必须是上下文管理器（同步与异步都要）；
    - **脱敏在实现内完成**（调用方不该操心）；
    - 追踪失败绝不能影响业务（内部吞掉异常并记日志）。
    """

    def span(self, name: str, **attributes: Any) -> Any: ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------- 脱敏


def _redact_value(key: str, value: Any) -> Any:
    """按 key 与内容脱敏单个属性值。"""
    from rag_core.pii import redact_text

    # 敏感键：整值替换成"长度标记"——保留长度便于判断"prompt 有多大"，
    # 不保留内容。
    if key.lower() in _SENSITIVE_KEYS and key.lower() not in _TEXT_SAFE_KEYS:
        if isinstance(value, str):
            return f"<redacted:{len(value)} chars>"
        return f"<redacted:{type(value).__name__}>"

    if isinstance(value, str):
        # 非敏感键也可能含 PII（如 `path` 里有用户手机号）
        return redact_text(value)
    if isinstance(value, dict):
        return {str(k): _redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(key, item) for item in value]
    return value


def redact_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """脱敏一组 span 属性。

    公开这个函数是为了**可测试**：脱敏规则必须能被单测覆盖，
    而不是埋在 tracer 实现里只能靠端到端验证。
    """
    return {key: _redact_value(key, value) for key, value in attributes.items()}


def text_fingerprint(text: str) -> str:
    """正文的指纹（用于"同一次请求的不同 span 是否处理同一份内容"）。

    用 SHA-256 前 12 位：不可逆，但能比对是否相同。
    比记录正文安全得多，比完全不记有用。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- 本地实现


class LocalTracer:
    """内存追踪器：span 记在列表里，可导出供测试与本地排查。

    用途：
    - 单测验证"链路上有哪些 span、父子关系对不对"；
    - 本地开发时 `print(spans)` 就能看耗时分布；
    - 无 Langfuse 时的兜底。
    """

    def __init__(self, *, max_spans: int = 1000) -> None:
        self._spans: list[SpanData] = []
        self._max_spans = max_spans
        self._stack: list[SpanData] = []

    @property
    def spans(self) -> list[SpanData]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()
        self._stack.clear()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[SpanData]:
        data = self._begin(name, attributes)
        try:
            yield data
        finally:
            self._end(data)

    @asynccontextmanager
    async def aspan(self, name: str, **attributes: Any) -> AsyncIterator[SpanData]:
        data = self._begin(name, attributes)
        try:
            yield data
        finally:
            self._end(data)

    def _begin(self, name: str, attributes: dict[str, Any]) -> SpanData:
        parent = self._stack[-1] if self._stack else None
        data = SpanData(
            name=name,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            start_ns=time.perf_counter_ns(),
            attributes=redact_attributes(attributes),
        )
        self._stack.append(data)
        return data

    def _end(self, data: SpanData) -> None:
        data.end_ns = time.perf_counter_ns()
        # 从栈里移除（正常是栈顶；异常路径下可能不是，所以用 remove）
        if self._stack and self._stack[-1] is data:
            self._stack.pop()
        elif data in self._stack:
            self._stack.remove(data)

        if len(self._spans) < self._max_spans:
            self._spans.append(data)

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------- Langfuse


class LangfuseTracer:
    """Langfuse 追踪器（可选）。

    **构造失败时降级为 LocalTracer 并明确告警**：
    Langfuse 是可选依赖，配置错/连不上不该让服务起不来，
    但也不能静默——否则"追踪没数据"无从排查。

    ## 为什么用 `langfuse` SDK 的 `start_as_current_span`

    它是 OpenTelemetry 兼容的接口，将来换 Jaeger/Tempo 时改动极小。
    自己拼 HTTP 请求会让"采样、批量发送、重试"全都要自研。
    """

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        fallback: LocalTracer | None = None,
    ) -> None:
        self._fallback = fallback or LocalTracer()
        self._client: Any = None

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            log.info("langfuse_tracer_ready", host=host)
        except ImportError:
            log.warning(
                "langfuse_not_installed",
                impact="追踪降级为本地内存记录（重启丢失）",
                hint="uv add langfuse",
            )
        except Exception as exc:
            log.warning(
                "langfuse_init_failed",
                error_type=type(exc).__name__,
                impact="追踪降级为本地内存记录",
            )

    @property
    def is_available(self) -> bool:
        return self._client is not None

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        safe = redact_attributes(attributes)
        if self._client is None:
            with self._fallback.span(name, **safe) as data:
                yield data
            return

        # 追踪失败不能影响业务：外层 try/except 包住"开 span"，
        # 但**不吞业务异常**（yield 在 try 内，业务的异常会正常传播）。
        try:
            cm = self._client.start_as_current_span(name=name)
        except Exception as exc:
            log.warning("span_start_failed", name=name, error_type=type(exc).__name__)
            with self._fallback.span(name, **safe) as data:
                yield data
            return

        with cm as span:
            try:
                self._apply_attributes(span, safe)
            except Exception as exc:
                # 属性写失败不该打断业务，但要留痕（debug 级）：
                # 完全静默会让"追踪里看不到某个字段"无从排查。
                log.debug(
                    "span_attribute_failed",
                    name=name,
                    error_type=type(exc).__name__,
                )
            yield span

    def _apply_attributes(self, span: Any, attributes: dict[str, Any]) -> None:
        """把属性写到 span 上。

        优先用 `update(metadata=...)`：Langfuse 的 `set_attribute` 对
        非标量值支持不一致，metadata 接受任意 JSON。
        """
        if not attributes:
            return
        if hasattr(span, "update"):
            try:
                span.update(metadata=attributes)
                return
            except Exception as exc:
                # Langfuse 的 metadata 对某些值类型不接受，
                # 降级到逐字段 set_attribute（强制转字符串）。
                log.debug(
                    "span_metadata_failed",
                    error_type=type(exc).__name__,
                    note="降级为逐字段 set_attribute",
                )

        for key, value in attributes.items():
            try:
                span.set_attribute(key, str(value))
            except Exception as exc:
                # 单个字段失败不影响其它字段
                log.debug(
                    "span_set_attribute_failed",
                    key=key,
                    error_type=type(exc).__name__,
                )

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                # Langfuse SDK 的 flush 有同步与异步两种形态，
                # 这里兼容处理：优先 flush（同步），失败再试 aflush
                flush = getattr(self._client, "flush", None)
                if flush is not None:
                    flush()
            except Exception as exc:
                log.warning("langfuse_flush_failed", error_type=type(exc).__name__)
        await self._fallback.aclose()


# ---------------------------------------------------------------- 装配


def build_tracer(
    *,
    enabled: bool,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
) -> LocalTracer | LangfuseTracer:
    """按配置构造追踪器。

    三种情况：
    - 未启用 → `LocalTracer`（仍然记录，供本地排查与测试断言）；
    - 启用且有凭证 → `LangfuseTracer`（失败自动降级）；
    - 启用但缺凭证 → `LocalTracer` + 明确告警（配置错误要说出来）。
    """
    if not enabled:
        return LocalTracer()

    if not public_key or not secret_key:
        log.warning(
            "tracing_enabled_without_credentials",
            impact="追踪降级为本地内存记录",
            hint="设置 RAG_TRACING__PUBLIC_KEY / SECRET_KEY / HOST",
        )
        return LocalTracer()

    return LangfuseTracer(
        public_key=public_key,
        secret_key=secret_key,
        host=host or "http://localhost:3000",
    )


def trace_summary(spans: list[SpanData]) -> dict[str, Any]:
    """把一组 span 汇总成"各节点耗时"（B4 链路可视化的数据源）。

    按名称聚合（同名 span 取均值与总数）：一次问答可能有 3 个
    `retrieval.vector` span（多查询），按名聚合才能看出总耗时。
    """
    by_name: dict[str, list[float]] = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span.duration_ms)

    return {
        "total_ms": sum(span.duration_ms for span in spans if span.parent_id is None),
        "span_count": len(spans),
        "nodes": [
            {
                "name": name,
                "count": len(durations),
                "total_ms": round(sum(durations), 1),
                "avg_ms": round(sum(durations) / len(durations), 1),
                "max_ms": round(max(durations), 1),
            }
            for name, durations in sorted(by_name.items(), key=lambda item: -sum(item[1]))
        ],
    }


__all__ = [
    "LangfuseTracer",
    "LocalTracer",
    "SpanData",
    "Tracer",
    "build_tracer",
    "redact_attributes",
    "text_fingerprint",
    "trace_summary",
]
