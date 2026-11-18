"""链路追踪：节点级 span + 脱敏 + 可选 Langfuse 导出。

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

Langfuse 需要服务端（`docker compose --profile langfuse` 或云服务）。
服务端不可用不该阻塞开发：

- `LocalTracer` 把 span 记在内存里，测试与本地排查够用；
- 接口与 `LangfuseTracer` 完全相同，切换只改一行配置；
- 没有 Langfuse 时**明确记一条 warning**，不静默降级
  （否则"追踪没数据"会变成没人知道原因的谜）。

## 与 Langfuse Python SDK 的版本绑定（实测，不是猜）

本文件按 **langfuse>=5.2,<6** 的 API 编写。实测差异：

- `start_as_current_span` / `start_as_current_generation`
  → `start_as_current_observation(as_type="span" | "generation")`
- `update_current_trace` → `langfuse.propagate_attributes(...)`
- `span.set_attribute(k, v)` → `span.update(metadata={...})`
- `client.flush()` → `client.shutdown()`（flush 仍存在，
  但 shutdown 才会停掉后台导出线程）

初版代码用的是左列那套接口，在 v5 上 `start_as_current_span` 是 `None`，
每次开 span 都抛 `AttributeError` 被 catch 吞掉 → **全部降级成本地记录，
外部看就是"Langfuse 一条数据都没有"**。这类问题必须在代码里显式对齐，
并靠一个假的 SDK 单测锁住（见 `tests/test_tracing.py`）。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from rag_core.logging import get_logger
from rag_core.settings.tracing import DEFAULT_LANGFUSE_HOST, TracingSettings

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
        "secret_key",
        "public_key",
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
    kind: str = "span"
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

    - 所有 span 方法都是上下文管理器（同步 `with` 与异步 `async with` 各一套）；
    - **脱敏在实现内完成**（调用方不该操心）；
    - 追踪失败绝不能影响业务（内部吞掉异常并记日志）；
    - `trace()` 建立"一次业务操作"的根（一次问答 / 一个任务），
      子 span 通过 `span()`/`aspan()` 挂在它下面。
    """

    def span(self, name: str, **attributes: Any) -> Any: ...

    def aspan(self, name: str, **attributes: Any) -> Any: ...

    def generation(self, name: str, **attributes: Any) -> Any: ...

    def ageneration(self, name: str, **attributes: Any) -> Any: ...

    def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        **attributes: Any,
    ) -> Any: ...

    def atrace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        **attributes: Any,
    ) -> Any: ...

    def annotate(self, span: Any, **attributes: Any) -> None: ...

    def record_usage(
        self,
        span: Any,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None = None,
    ) -> None: ...

    def current_trace_id(self) -> str | None: ...

    async def averify(self) -> bool: ...

    async def score_trace(
        self,
        *,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None: ...

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

    ## 为什么栈放在 ContextVar 里

    初版用实例属性 `self._stack`。在 API 里一个进程同时处理几十个请求，
    A 请求开的 span 会被 B 请求当成父节点（互相污染），
    表现为"trace 里父子关系乱七八糟"，而本地串行测试完全复现不了。

    ContextVar 天然按任务隔离：asyncio 的每个 Task 复制一份上下文，
    跨 `await` 传播，并发请求互不干扰。
    """

    def __init__(self, *, max_spans: int = 1000) -> None:
        self._spans: list[SpanData] = []
        self._max_spans = max_spans
        self._stack: ContextVar[tuple[SpanData, ...]] = ContextVar(
            f"rag_trace_stack_{id(self)}", default=()
        )

    @property
    def spans(self) -> list[SpanData]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()
        self._stack.set(())

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[SpanData]:
        data = self._begin(name, attributes, kind="span")
        try:
            yield data
        finally:
            self._end(data)

    @asynccontextmanager
    async def aspan(self, name: str, **attributes: Any) -> AsyncIterator[SpanData]:
        data = self._begin(name, attributes, kind="span")
        try:
            yield data
        finally:
            self._end(data)

    @contextmanager
    def generation(self, name: str, **attributes: Any) -> Iterator[SpanData]:
        data = self._begin(name, attributes, kind="generation")
        try:
            yield data
        finally:
            self._end(data)

    @asynccontextmanager
    async def ageneration(self, name: str, **attributes: Any) -> AsyncIterator[SpanData]:
        data = self._begin(name, attributes, kind="generation")
        try:
            yield data
        finally:
            self._end(data)

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        **attributes: Any,
    ) -> Iterator[SpanData]:
        data = self._begin(
            name,
            {"user_id": user_id, "session_id": session_id, **attributes},
            kind="trace",
        )
        try:
            yield data
        finally:
            self._end(data)

    @asynccontextmanager
    async def atrace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        **attributes: Any,
    ) -> AsyncIterator[SpanData]:
        data = self._begin(
            name,
            {"user_id": user_id, "session_id": session_id, **attributes},
            kind="trace",
        )
        try:
            yield data
        finally:
            self._end(data)

    def annotate(self, span: Any, **attributes: Any) -> None:
        if isinstance(span, SpanData):
            span.attributes.update(redact_attributes(attributes))

    def record_usage(
        self,
        span: Any,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None = None,
    ) -> None:
        if not isinstance(span, SpanData):
            return
        span.attributes.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                **({"cost_usd": round(cost_usd, 8)} if cost_usd is not None else {}),
            }
        )

    def current_trace_id(self) -> str | None:
        stack = self._stack.get()
        return stack[0].trace_id if stack else None

    async def averify(self) -> bool:
        return True

    async def score_trace(
        self,
        *,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        # 本地实现只记日志：内存 tracer 没有"按 trace_id 追加 score"的语义，
        # 造一个半成品接口比不实现更容易误导。
        log.info("trace_score_local", trace_id=trace_id, name=name, value=value)

    async def aclose(self) -> None:
        return None

    # ------------------------------------------------------------ 内部

    def _begin(self, name: str, attributes: dict[str, Any], *, kind: str) -> SpanData:
        stack = self._stack.get()
        parent = stack[-1] if stack else None
        data = SpanData(
            name=name,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            kind=kind,
            start_ns=time.perf_counter_ns(),
            attributes=redact_attributes(attributes),
        )
        self._stack.set((*stack, data))
        return data

    def _end(self, data: SpanData) -> None:
        data.end_ns = time.perf_counter_ns()
        # 从栈里移除（正常是栈顶；异常路径下可能不是，所以按 identity 过滤）
        stack = self._stack.get()
        self._stack.set(tuple(item for item in stack if item is not data))

        if len(self._spans) < self._max_spans:
            self._spans.append(data)


# ---------------------------------------------------------------- Langfuse


class LangfuseTracer:
    """Langfuse 追踪器。

    **构造失败时降级为 LocalTracer 并明确告警**：
    配置错/连不上不该让服务起不来，但也不能静默——
    否则"追踪没数据"无从排查。

    ## 为什么用 SDK 的 observation 接口而不是自己发 HTTP

    SDK 负责批量、重试、采样、上下文传播（OTel contextvars），
    自己实现这些的复杂度远高于收益，且容易在异步上下文里丢父子关系。
    """

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str = DEFAULT_LANGFUSE_HOST,
        environment: str | None = None,
        release: str | None = None,
        sample_rate: float = 1.0,
        flush_at: int = 64,
        flush_interval_s: float = 2.0,
        timeout_s: int = 10,
        fallback: LocalTracer | None = None,
    ) -> None:
        self._fallback = fallback or LocalTracer()
        self._client: Any = None
        self._host = host
        self._environment = environment
        self._public_key = public_key
        self._secret_key = secret_key
        self._verified = False

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                environment=environment,
                release=release,
                # 采样在 SDK 内完成：被丢弃的 trace 依然有 trace_id，
                # 所以 model_calls 里的关联不会出现"有 id 但查不到"的困惑，
                # 而是"本来就没采样"——这是可接受且可解释的。
                sample_rate=sample_rate,
                flush_at=flush_at,
                flush_interval=flush_interval_s,
                timeout=timeout_s,
                # 显式开启导出：默认值在不同版本里变过，写明避免依赖默认。
                tracing_enabled=True,
            )
            log.info(
                "langfuse_tracer_ready",
                host=host,
                environment=environment,
                release=release,
                sample_rate=sample_rate,
            )
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

    # ------------------------------------------------------------ span

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        with self._open_sync(name, "span", attributes) as span:
            yield span

    @asynccontextmanager
    async def aspan(self, name: str, **attributes: Any) -> AsyncIterator[Any]:
        # **用同步 `with` 打开 SDK 的 observation**：
        # 实测 langfuse 5.2 的 `start_as_current_observation` 返回
        # `_AgnosticContextManager`，它只实现了 `__enter__/__exit__`；
        # 用 `async with` 会抛 "does not support the asynchronous
        # context manager protocol"（worker 里表现为任务全部进死信）。
        #
        # 外层仍是 async context manager：调用方写 `async with` 不受影响，
        # 且降级到 LocalTracer 时仍走它的异步实现。
        with self._open_sync(name, "span", attributes) as span:
            yield span

    @contextmanager
    def generation(self, name: str, **attributes: Any) -> Iterator[Any]:
        with self._open_sync(name, "generation", attributes) as span:
            yield span

    @asynccontextmanager
    async def ageneration(self, name: str, **attributes: Any) -> AsyncIterator[Any]:
        with self._open_sync(name, "generation", attributes) as span:
            yield span

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        **attributes: Any,
    ) -> Iterator[Any]:
        with (
            self._open_sync(name, "span", attributes) as span,
            self._propagate(
                user_id=user_id, session_id=session_id, name=name, attributes=attributes
            ),
        ):
            yield span

    @asynccontextmanager
    async def atrace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        **attributes: Any,
    ) -> AsyncIterator[Any]:
        with (
            self._open_sync(name, "span", attributes) as span,
            self._propagate(
                user_id=user_id, session_id=session_id, name=name, attributes=attributes
            ),
        ):
            yield span

    def annotate(self, span: Any, **attributes: Any) -> None:
        if span is None or not attributes:
            return
        safe = redact_attributes(attributes)
        if isinstance(span, SpanData):
            span.attributes.update(safe)
            return
        update = getattr(span, "update", None)
        if update is None:
            return
        try:
            update(metadata=safe)
        except Exception as exc:
            # 属性写失败不该打断业务，但要留痕（debug 级）：
            # 完全静默会让"追踪里看不到某个字段"无从排查。
            log.debug("span_annotate_failed", error_type=type(exc).__name__)

    def record_usage(
        self,
        span: Any,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None = None,
    ) -> None:
        """把 token 用量写到 generation 上。

        **成本以 `model_calls` 为准**：这里只做"在 trace 里能直接看到贵不贵"
        的展示，不参与对账。两边都算一遍容易口径漂移，
        而对账的唯一真相必须是数据库（见 AGENTS.md 4.16）。
        """
        if span is None:
            return
        if isinstance(span, SpanData):
            span.attributes.update(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    **({"cost_usd": round(cost_usd, 8)} if cost_usd is not None else {}),
                }
            )
            return
        update = getattr(span, "update", None)
        if update is None:
            return
        payload: dict[str, Any] = {
            "usage_details": {"input": input_tokens, "output": output_tokens}
        }
        if cost_usd is not None:
            payload["cost_details"] = {"total": round(float(cost_usd), 8)}
        try:
            update(**payload)
        except Exception as exc:
            log.debug("span_usage_failed", error_type=type(exc).__name__)

    def current_trace_id(self) -> str | None:
        if self._client is None:
            return self._fallback.current_trace_id()
        try:
            trace_id = self._client.get_current_trace_id()
        except Exception as exc:  # pragma: no cover - SDK 内部异常
            log.debug("get_trace_id_failed", error_type=type(exc).__name__)
            return None
        return str(trace_id) if trace_id else None

    # ------------------------------------------------------------ 生命周期

    async def averify(self) -> bool:
        """主动探测服务端连通性与凭证（只在启动时调一次）。

        ## 为什么必须有这一步

        SDK 的导出是**后台异步**的：host 写错、凭证错误、服务没起，
        业务侧一次异常都不会看到，只会在 Langfuse 里"没有数据"。
        这类失效最难排查，所以启动时必须主动问一次服务端。
        """
        if self._client is None:
            return False
        if self._verified:
            return True
        try:
            await asyncio.to_thread(self._client.auth_check)
        except Exception as exc:
            log.warning(
                "langfuse_auth_check_failed",
                host=self._host,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
                impact="trace 会持续发送失败，业务不受影响但观测数据缺失",
                hint="确认 RAG_TRACING__HOST / PUBLIC_KEY / SECRET_KEY 与 Langfuse 项目一致",
            )
            return False
        self._verified = True
        log.info("langfuse_auth_check_ok", host=self._host)
        return True

    async def score_trace(
        self,
        *,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """给已有 trace 追加一个 score（用户反馈 → 评估数据闭环）。

        ## 为什么直接调 REST 而不用 SDK 的 `create_score`

        实测：`langfuse==5.2.0` 对 `langfuse/langfuse:3`（3.227.1）调
        `create_score` 会被服务端拒绝（HTTP 400），而 SDK 只把错误**打印到
        stderr、不抛异常**——于是业务侧看起来"成功"，Langfuse 里却一条
        score 都没有。

        `POST /api/public/scores` 是稳定接口（与 server 版本解耦，
        直接 curl 验证过 200）。这里用它，失败时我们自己能拿到状态码。

        失败只记日志：反馈已经落库，评分是旁路增强。
        """
        if not self._public_key:
            return
        payload: dict[str, Any] = {
            "traceId": trace_id,
            "name": name,
            "value": value,
            # 显式声明类型：不传时服务端按 JSON 类型推断，行为随版本变过
            "dataType": "NUMERIC",
        }
        if comment:
            payload["comment"] = comment
        if self._environment:
            payload["environment"] = self._environment

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self._host}/api/public/scores",
                    json=payload,
                    auth=httpx.BasicAuth(self._public_key, self._secret_key),
                )
        except Exception as exc:
            log.warning(
                "langfuse_score_failed",
                name=name,
                trace_id=trace_id,
                error_type=type(exc).__name__,
            )
            return

        if response.status_code >= 400:
            log.warning(
                "langfuse_score_rejected",
                name=name,
                trace_id=trace_id,
                status=response.status_code,
                body=response.text[:200],
            )

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                # v5：shutdown() 会 flush + 停掉后台导出线程。
                # 只调 flush() 的话，进程退出时后台线程可能还在发最后一批。
                # 放线程里执行：导出是同步 HTTP，会阻塞事件循环。
                await asyncio.to_thread(self._client.shutdown)
            except Exception as exc:
                log.warning("langfuse_shutdown_failed", error_type=type(exc).__name__)
        await self._fallback.aclose()

    # ------------------------------------------------------------ 内部

    def _open_sync(self, name: str, kind: str, attributes: dict[str, Any]) -> Any:
        """打开 span 上下文。失败时降级到 LocalTracer。

        **只有这一条打开路径**：SDK 返回的 `_AgnosticContextManager`
        在该版本只支持同步 `with`，所以异步调用方也复用它
        （见 `aspan` 的注释）。
        """
        safe = redact_attributes(attributes)
        fallback_attrs = dict(safe)
        extra: dict[str, Any] = {}
        if kind == "generation":
            # `model` 是 Langfuse 的专用字段（不是 metadata）：只有写在这里，
            # 才能按模型聚合 token 与成本。放在 metadata 里就只是“看一下”。
            model = safe.pop("model", None)
            if isinstance(model, str):
                extra["model"] = model
        if self._client is not None:
            try:
                return self._client.start_as_current_observation(
                    name=name, as_type=kind, metadata=safe or None, **extra
                )
            except Exception as exc:
                log.warning("span_start_failed", name=name, error_type=type(exc).__name__)
        # 降级：**保留 span 结构**，好用测试与本地排查
        return self._fallback.span(name, **{**fallback_attrs, **extra})

    def _propagate(
        self,
        *,
        user_id: str | None,
        session_id: str | None,
        name: str,
        attributes: dict[str, Any],
    ) -> Any:
        """把 trace 级属性（user/session）传播给后续所有子 span。"""
        if self._client is None or not (user_id or session_id):
            return nullcontext()
        try:
            from langfuse import propagate_attributes

            return propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                trace_name=name,
                metadata=redact_attributes(attributes) or None,
            )
        except Exception as exc:  # pragma: no cover - SDK 版本差异
            log.debug("trace_attributes_failed", error_type=type(exc).__name__)
            return nullcontext()


# ---------------------------------------------------------------- 装配


def build_tracer(
    settings: TracingSettings | None = None,
    *,
    enabled: bool | None = None,
) -> LocalTracer | LangfuseTracer:
    """按配置构造追踪器。

    四种情况：

    - 未启用 → `LocalTracer`（仍然记录，供本地排查与测试断言）；
    - 启用且有凭证 → `LangfuseTracer`（构造失败自动降级）；
    - 启用但缺凭证 → `LocalTracer` + 明确告警（配置错误要说出来）；
    - 传 `settings=None` → `LocalTracer`（测试/脚本的默认值）。

    `enabled` 参数只用于测试覆盖，生产以配置为准。
    """
    if settings is None:
        return LocalTracer()

    if not (settings.enabled if enabled is None else enabled):
        return LocalTracer()

    public_key = settings.public_key.get_secret_value() if settings.public_key else ""
    secret_key = settings.secret_key.get_secret_value() if settings.secret_key else ""

    if not public_key or not secret_key:
        log.warning(
            "tracing_enabled_without_credentials",
            impact="追踪降级为本地内存记录",
            hint="设置 RAG_TRACING__PUBLIC_KEY / RAG_TRACING__SECRET_KEY",
        )
        return LocalTracer()

    return LangfuseTracer(
        public_key=public_key,
        secret_key=secret_key,
        host=settings.host,
        environment=settings.environment,
        release=settings.release,
        sample_rate=settings.sample_rate,
        flush_at=settings.flush_at,
        flush_interval_s=settings.flush_interval_s,
        timeout_s=settings.timeout_s,
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
