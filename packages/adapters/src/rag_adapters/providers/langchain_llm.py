"""基于 LangChain ChatOpenAI 的 LLM Provider 实现。

## 为什么用 langchain-openai 而不是自研 httpx

一开始我自研了 355 行 httpx 实现，理由是"LangChain 会丢失缓存/推理
token 的细分"。**实测证明这个判断是错的**：

    usage_metadata = {
        "input_tokens": 39,
        "output_tokens": 31,
        "total_tokens": 70,
        "input_token_details": {"cache_read": 0},      ← 缓存命中
        "output_token_details": {"reasoning": 29},     ← 推理 token
    }

LangChain 的 `InputTokenDetails` / `OutputTokenDetails` 完整覆盖了
成本核算需要的所有字段。

### 选型的真实权衡

| 维度 | LangChain | 自研 httpx |
|---|---|---|
| 用量字段 | ✅ 完整 | ✅ 手写解析 |
| 流式 SSE | ✅ 成熟（边界、[DONE]、增量） | ⚠️ 手写易出坑 |
| 与 LangGraph（A11）集成 | ✅ 原生 | ❌ 需适配层 |
| 依赖体积 | ❌ 67 个传递依赖 | ✅ 1 个 httpx |
| 重试控制 | ⚠️ 自带（`max_retries=0` 可关） | ✅ 天然无 |

**决定性理由是 A11**：`docs/PROMPT.md` 要求检索用 LangGraph 状态图，
节点里需要把 LLM 当可调用对象传递。LangChain 的 `BaseChatModel` 是
LangGraph 的一等公民，自研 Provider 得写适配器。

依赖体积的代价在这个项目里可接受：后端服务镜像大几十 MB 不是瓶颈。

## 必须做对的三件事

1. **`max_retries=0`**：LangChain 默认重试 2 次。不关掉的话
   会出现 "LangChain 重试 × ResilientCaller 重试" 的乘数效应，
   且熔断器拿不到真实失败信号（A2-7 已强调统一重试入口）。
2. **`reasoning_effort` 走 `extra_body`**：这是 DeepSeek 特有参数，
   `ChatOpenAI` 的具名参数里没有，必须放 extra_body。
3. **用 `response_metadata["model_name"]` 记账**：DeepSeek 会把已下线的
   模型名路由到新模型（实测 `deepseek-chat` → `deepseek-flash`），
   按请求名记账会与账单不符。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_adapters.providers.llm import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    LLMRole,
    ProviderCapabilities,
    TokenUsage,
)
from rag_core.logging import get_logger
from rag_core.settings.llm import LLMSettings

log = get_logger(__name__)


class LangChainLLMError(LLMProviderError):
    """LLM 调用失败。"""


def _to_lc_messages(messages: list[LLMMessage]) -> list[Any]:
    """把抽象消息转成 LangChain 消息。

    映射表集中在这里：新增角色（如 tool）只改这一处。
    """
    mapping = {
        LLMRole.SYSTEM: SystemMessage,
        LLMRole.USER: HumanMessage,
        LLMRole.ASSISTANT: AIMessage,
    }
    return [mapping[message.role](content=message.content) for message in messages]


def _to_usage(message: AIMessage) -> TokenUsage:
    """把 LangChain 的 usage_metadata 转成我们的 TokenUsage。

    字段语义（实测确认）：
    - `input_tokens` **已包含** `input_token_details.cache_read`；
    - `output_tokens` **已包含** `output_token_details.reasoning`。

    所以这里做的是**提取子集而非相加**。相加会让成本翻倍。
    """
    usage: dict[str, Any] = message.usage_metadata or {}  # type: ignore[assignment]
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)

    input_details: dict[str, Any] = usage.get("input_token_details") or {}
    output_details: dict[str, Any] = usage.get("output_token_details") or {}

    cached = int(input_details.get("cache_read") or 0)
    reasoning = int(output_details.get("reasoning") or 0)

    # 一致性检查：字段语义若变化，不能让成本算出负值
    if cached > input_tokens:
        log.warning(
            "usage_cached_exceeds_input",
            input_tokens=input_tokens,
            cached_tokens=cached,
        )
        cached = input_tokens
    if reasoning > output_tokens:
        reasoning = output_tokens

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )


def _usage_from_metadata(metadata: dict[str, Any]) -> TokenUsage:
    """从流式 chunk 的 `usage_metadata` 转 TokenUsage。

    与 `_to_usage` 同一套字段语义（见其 docstring），
    但入参是裸 dict 而不是 AIMessage —— 流式 chunk 的
    `usage_metadata` 是 dict，没有 message 包装。

    **不复用 `_to_usage`** 是为了避免构造一个假的 AIMessage：
    那会让"这里为什么要造一个空 message"变成需要解释的困惑。
    """

    class _Fake:
        usage_metadata = metadata

    return _to_usage(_Fake())  # type: ignore[arg-type]


def _classify_error(exc: BaseException) -> LLMProviderError:
    """把 LangChain/OpenAI SDK 的异常翻译成带可重试标记的异常。

    为什么要分类：`ResilientCaller` 靠 `retryable` 决定是否重试。
    把 401（密钥失效）标成可重试会导致每 0.2s 打一次认证接口，
    既浪费又可能触发供应商风控。
    """
    name = type(exc).__name__
    text = str(exc)[:200]

    # 认证/权限：重试无用
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return LangChainLLMError("供应商认证失败", retryable=False)
    # 限流
    if name == "RateLimitError":
        return LangChainLLMError("供应商限流", retryable=True)
    # 超时
    if name in ("APITimeoutError", "TimeoutError", "ReadTimeout"):
        return LangChainLLMError("供应商超时", retryable=True)
    # 连接错误
    if name in ("APIConnectionError", "ConnectError"):
        return LangChainLLMError("供应商网络错误", retryable=True)
    # 服务端 5xx
    if name == "InternalServerError":
        return LangChainLLMError("供应商服务端错误", retryable=True)
    # 请求非法（400、上下文超长、参数错）：不重试
    if name in ("BadRequestError", "UnprocessableEntityError"):
        return LangChainLLMError(f"请求被拒绝: {text}", retryable=False)

    # 未知异常：保守判定不重试（避免放大未知 bug），但保留原始类型名便于排查
    log.warning("llm_unclassified_error", error_type=name)
    return LangChainLLMError(f"调用失败: {name}", retryable=False)


class LangChainLLMProvider(LLMProvider):
    """LangChain `ChatOpenAI` 的适配器。

    适配范围仅限：
    - 消息类型转换（`LLMMessage` ↔ LangChain 消息）；
    - 用量字段提取（`usage_metadata` → `TokenUsage`）；
    - 异常分类（SDK 异常 → `retryable` 标记）；
    - 能力声明。

    协议本身（HTTP、SSE、JSON 解析、重试语义）由 LangChain 负责。
    """

    def __init__(
        self,
        settings: LLMSettings,
        *,
        model: str | None = None,
        llm: ChatOpenAI | None = None,
    ) -> None:
        if not settings.api_key and llm is None:
            raise LangChainLLMError("未配置 RAG_LLM__API_KEY", retryable=False)

        self._settings = settings
        self.name = settings.provider
        self._model_name = model or settings.chat_model

        self._http_client: httpx.AsyncClient | None = None

        if llm is not None:
            # 测试注入：已经构造好的 ChatOpenAI（可用 mock transport）
            self._llm = llm
        else:
            # **必须传独立的 httpx client**：
            # openai SDK 在不传 http_client 时会用模块级共享 client，
            # 多个 ChatOpenAI 实例共享它。一旦某个实例 aclose() 关掉它，
            # 其它实例全部失效（实测：`Event loop is closed`）。
            self._http_client = httpx.AsyncClient(timeout=settings.timeout_s)
            self._llm = self._build_llm(self._model_name)

    def _build_llm(self, model: str) -> ChatOpenAI:
        """按配置构造 ChatOpenAI。

        三个关键参数：
        - `max_retries=0`：关闭 SDK 内置重试，重试统一由 ResilientCaller 负责；
        - `extra_body={"reasoning_effort": ...}`：DeepSeek 特有参数，
          不在 ChatOpenAI 的具名参数里；
        - `streaming=False`：非流式调用时明确关闭（流式用单独的实例行为）。

        **`extra_body` 只在配置了值时才传**：不支持的厂商会因未知参数
        返回 400（比如某些 OpenAI 兼容实现严格校验 schema）。
        """
        extra: dict[str, Any] = {}
        if self._settings.reasoning_effort:
            extra["reasoning_effort"] = self._settings.reasoning_effort

        return ChatOpenAI(
            model=model,
            api_key=self._settings.api_key,  # type: ignore[arg-type]
            base_url=self._settings.base_url,
            timeout=self._settings.timeout_s,
            max_retries=0,  # 关键：不让 SDK 自己重试
            streaming=False,
            extra_body=extra or None,
            # 独立 client：避免与其它实例共享并被误关
            http_async_client=self._http_client,
        )

    @property
    def model_name(self) -> str:
        """配置的 chat_model（用于查单价、写 model_calls）。

        注意与 `LLMResponse.model` 的区别：那是**响应里**供应商实际
        使用的模型名（DeepSeek 可能把下线模型路由到新模型）。
        查单价用配置名（本地要知道"我们想用哪个"），
        记账用响应名（"实际按哪个计费"）——两者都记。
        """
        return self._model_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        """能力声明。

        `max_context` 按配置里模型的定价表无法推断，这里用 settings 的
        保守值。真实裁剪由 A11 用 token 计数做，不依赖这个声明。
        """
        return ProviderCapabilities(
            supports_stream=True,
            supports_vision=self._settings.provider == "deepseek",
            supports_json_mode=True,
            is_local=False,
            max_context=1_048_576 if self._settings.provider == "deepseek" else 128_000,
            cost_tier="standard",
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """非流式补全。"""
        llm = self._llm
        if model is not None and model != self._model_name:
            # 按次指定模型：用 bind 覆盖 model 参数，而不是重建实例。
            # 重建会丢掉注入的 client（测试注入的 fake、或共享的连接池），
            # 也让"注入 llm"这个扩展点失效。
            llm = llm.bind(model=model)  # type: ignore[assignment]
        if temperature != 0.7:
            llm = llm.bind(temperature=temperature)  # type: ignore[assignment]
        if max_tokens is not None:
            llm = llm.bind(max_tokens=max_tokens)  # type: ignore[assignment]
        if json_mode:
            llm = llm.bind(response_format={"type": "json_object"})  # type: ignore[assignment]

        started = time.perf_counter()
        try:
            # 按次超时用 `with_config(timeout=...)` 覆盖：
            # `ainvoke(timeout=...)` 不是合法 kwarg，LangChain 会抛
            # RuntimeError（实测踩到，且错误信息完全看不出是超时参数的问题）
            callable_llm = llm.with_config(timeout=timeout_s) if timeout_s is not None else llm
            message = await callable_llm.ainvoke(_to_lc_messages(messages))
        except Exception as exc:
            raise _classify_error(exc) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        if not isinstance(message, AIMessage):
            raise LangChainLLMError("返回类型异常", retryable=False)

        content = message.content if isinstance(message.content, str) else str(message.content)
        finish_reason = str(message.response_metadata.get("finish_reason") or "")
        # **用响应里的实际模型名记账**（供应商可能改写模型名）
        actual_model = str(
            message.response_metadata.get("model_name")
            or message.response_metadata.get("model")
            or model
            or self._model_name
        )

        usage = _to_usage(message)

        # 思考模型的通病：max_tokens 被思考耗尽 → content 空 + length。
        # 静默返回空串会被误判为"模型拒答"，掩盖真实原因。
        if not content and finish_reason == "length":
            log.warning(
                "llm_output_empty_truncated",
                model=actual_model,
                max_tokens=max_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                note="max_tokens 被思考过程耗尽，需提高 max_tokens 或降低 reasoning_effort",
            )
            raise LangChainLLMError(
                f"输出被截断：{usage.reasoning_tokens} 个 token 消耗在思考过程，"
                f"max_tokens={max_tokens} 不足以产出正文",
                retryable=True,
            )

        return LLMResponse(
            content=content,
            model=actual_model,
            usage=usage,
            finish_reason=finish_reason or None,
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
        usage_callback: Callable[[TokenUsage], None] | None = None,
    ) -> AsyncIterator[str]:
        """流式补全，逐块产出文本增量。

        SSE 的边界处理（`data:` 前缀、`[DONE]`、跨 chunk 的半行）
        都由 LangChain 负责——这正是自研实现最容易出坑的地方。

        ## usage 只出现在最后一个 chunk

        OpenAI 兼容协议（含 DeepSeek）在**最后一个** chunk 的
        `usage` 字段给总用量，中间的 chunk 是 null。所以：
        - 生成过程中拿不到成本（只能等结束）；
        - 必须显式取最后一个 chunk 的 `usage_metadata`，
          否则流式路径的成本永远是 0（A5 实现漏了这个）。
        """
        last_usage: TokenUsage | None = None
        llm = self._llm
        if model is not None and model != self._model_name:
            llm = llm.bind(model=model)  # type: ignore[assignment]
        if temperature != 0.7:
            llm = llm.bind(temperature=temperature)  # type: ignore[assignment]
        if max_tokens is not None:
            llm = llm.bind(max_tokens=max_tokens)  # type: ignore[assignment]

        try:
            async for chunk in llm.astream(_to_lc_messages(messages)):
                # 每个 chunk 都可能带 usage_metadata，最后一个才是完整值。
                # 用 `=` 而不是 `+=`：LangChain 在流式里给的是**累计值**
                # （最后一个 chunk 含全部），累加会翻倍。
                metadata = getattr(chunk, "usage_metadata", None)
                if metadata:
                    last_usage = _usage_from_metadata(metadata)

                if isinstance(chunk.content, str) and chunk.content:
                    yield chunk.content
                elif isinstance(chunk.content, list):
                    # 多模态内容块：只取文本部分
                    for part in chunk.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            yield str(part.get("text") or "")
        except Exception as exc:
            raise _classify_error(exc) from exc
        finally:
            # 无论正常结束还是异常中断，都把已拿到的 usage 回调出去。
            # 异常时也回调：供应商可能已生成部分 token 并计费。
            if usage_callback is not None and last_usage is not None:
                usage_callback(last_usage)

    async def health(self) -> bool:
        """探活：一次极小的真实调用。

        `max_tokens` 留余量——思考模式会先消耗预算，
        给 1 会必然截断导致误判为不健康。
        """
        try:
            await self.complete(
                [LLMMessage(role=LLMRole.USER, content="hi")],
                model=self._settings.cheap_model,
                max_tokens=100,
                timeout_s=10.0,
            )
            return True
        except Exception as exc:
            log.warning("provider_health_failed", error=type(exc).__name__)
            return False

    async def aclose(self) -> None:
        """关闭底层 HTTP 连接。

        **必须关 `root_async_client`**：`ChatOpenAI.async_client` 是
        `AsyncCompletions`（API 分组对象），不是 HTTP 客户端；
        真正的 `AsyncOpenAI`（内含 httpx.AsyncClient）在
        `root_async_client`。

        不关的后果（实测）：httpx client 绑定在创建它的事件循环上。
        pytest-asyncio 每个用例用新的事件循环，复用旧 Provider 会抛
        `RuntimeError: Event loop is closed` —— 且这个错误被 SDK 包了一层，
        错误信息完全看不出是"连接属于旧事件循环"。
        生产里同理：Provider 若在 import 时创建、被多个事件循环使用就会崩。
        """
        # 只关自己创建的 client。
        # **不能关 root_async_client**：openai SDK 默认共享模块级 httpx client，
        # 关掉会让进程内其它实例全部失效（实测导致后续调用抛
        # "Event loop is closed"，且错误信息完全指不到根因）。
        if self._http_client is not None:
            import contextlib

            with contextlib.suppress(Exception):
                await self._http_client.aclose()
            self._http_client = None


class DeepSeekProvider(LangChainLLMProvider):
    """DeepSeek 专用别名（保留以兼容既有引用与测试）。

    实现完全复用 `LangChainLLMProvider`——DeepSeek 只是 OpenAI 兼容端点，
    差异全在 `LLMSettings` 的配置里（base_url、model、reasoning_effort）。
    """

    name = "deepseek"
