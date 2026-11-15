"""LLM 供应商抽象。

## 抽象边界

`LLMProvider` 只负责"把消息发出去、把文本收回来、报告用量"：

    complete(messages, ...) -> LLMResponse

它**不负责**：
- 重试/熔断/限流（那是 `ResilientCaller`，A2-7）；
- 成本计算（那是 service 层，用 `model_calls` 表）；
- 模型选择（那是 `ModelGateway` 的路由逻辑）。

这样切分的原因：换供应商只应改"怎么发请求"，而重试策略、成本口径、
路由规则对所有供应商都一样。把它们塞进 Provider 会让每个实现重复一遍。

## 能力自描述

`capabilities` 让路由层按能力匹配而不是 `if provider == "xxx"`：
- `supports_stream`：能不能流式；
- `supports_vision`：能不能读图；
- `max_context`：上下文窗口，用于裁剪；
- `supports_json_mode`：能不能强制 JSON 输出（结构化抽取要用）；
- `is_local`：本地模型（成本 0，但慢）；
- `cost_tier`：成本档位（cheap / standard / premium），用于分级路由。

`docs/PROMPT.md` 明确要求："见到 `if provider == "xxx"` 一律改成能力匹配"。

## 消息类型

统一用 `LLMMessage` 而不是直接传 OpenAI 的 dict：
- 换供应商时不用改业务代码的消息构造；
- 可以在抽象层统一做敏感信息脱敏（A14 的 trace 打点要脱敏 Authorization）。

注意：这里**不做 prompt 模板**。模板属于 prompts 层（A11），
Provider 只负责传输。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum


class LLMRole(StrEnum):
    """消息角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """一条对话消息。"""

    role: LLMRole
    content: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """token 用量。

    **字段语义必须与供应商的计费口径一致**，否则成本会算错。
    DeepSeek 的字段实测（2026-11）：

        prompt_tokens=31, completion_tokens=5,
        prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=31,
        completion_tokens_details={reasoning_tokens: 5}

    关键点：
    - `prompt_tokens` **已包含**缓存命中的部分，不要再相加；
    - `reasoning_tokens` **已包含在** `completion_tokens` 里（思考模式的输出），
      不要重复计费；
    - `cached_tokens` 是从 `prompt_tokens` 里划出来的子集。
    """

    input_tokens: int = 0
    """输入 token 总数（含缓存命中部分）。"""

    output_tokens: int = 0
    """输出 token 总数（含 reasoning 部分）。"""

    cached_tokens: int = 0
    """其中命中 prompt cache 的输入 token 数。"""

    reasoning_tokens: int = 0
    """其中思考过程消耗的输出 token（已含在 output_tokens 内）。"""

    @property
    def uncached_input(self) -> int:
        """未命中缓存的输入 token（按常规单价计费的部分）。"""
        return max(0, self.input_tokens - self.cached_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """供应商/模型的能力声明。路由层按这些字段匹配，不做名字判断。"""

    supports_stream: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    is_local: bool = False
    """本地模型：无 API 成本，但吞吐低、可能占满 CPU/GPU。"""

    max_context: int = 8192
    """上下文窗口（token）。超过必须裁剪，否则供应商直接报错。"""

    cost_tier: str = "standard"
    """成本档位：cheap / standard / premium。分级路由用。"""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """一次补全的结果。"""

    content: str
    model: str
    """供应商实际使用的模型名。

    **必须用响应里的值，而不是请求里写的**：DeepSeek 会把已下线的模型名
    （如 `deepseek-chat`）路由到新模型并返回新名字。用请求名记账会导致
    "成本按旧价算、实际按新模型计费"。
    """

    usage: TokenUsage
    finish_reason: str | None = None
    """停止原因：stop / length / content_filter。`length` 说明被截断。"""

    latency_ms: int | None = None
    raw_metadata: dict[str, str] = field(default_factory=dict)
    """可公开的元数据（不含量级、不含 prompt）。"""


class LLMProviderError(Exception):
    """Provider 层错误的基类。

    **不继承 AppError**：Provider 活在 rag_api 内，但它的错误需要
    被 service 层翻译成业务异常（`ProviderTimeoutError` 等）。
    这样 Provider 不依赖 core 的异常体系，可以独立测试。
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


async def _empty_stream() -> AsyncIterator[str]:
    """空异步迭代器（供基类 stream 默认实现使用）。

    用 `for ... in ()` 而不是 `return` + `yield`：
    后者会被 mypy 的 warn_unreachable 判为不可达代码。
    遍历空元组是"合法但产出 0 个元素"的写法。
    """
    empty: tuple[str, ...] = ()
    for item in empty:
        yield item


class LLMProvider(ABC):
    """LLM 供应商接口。

    实现约定：
    - **不做重试**：重试由 ResilientCaller 统一负责。Provider 自己重试
      会导致重试次数不可控（客户端 × 业务 × 网关），且熔断器收不到真实失败信号；
    - **不吞异常**：失败必须抛出 `LLMProviderError` 或其子类；
    - **不记录 prompt**：日志与异常里绝不能出现用户文档内容或对话正文。
    """

    #: 供应商标识，进日志与 model_calls
    name: str = "unknown"

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """当前模型的能力。路由层据此匹配。"""
        raise NotImplementedError

    @abstractmethod
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
        """执行一次非流式补全。

        Args:
            model: 覆盖默认模型。None 用配置里的 `chat_model`。
            json_mode: 要求输出合法 JSON（结构化抽取用）。
            timeout_s: 覆盖默认超时。
        """
        raise NotImplementedError

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[str]:
        """流式补全，逐块产出文本。

        默认实现：**返回空流**（什么都不产出）。

        为什么默认不抛异常：基类方法要同时满足"是异步生成器"与
        "mypy 的 warn_unreachable"。抛 `NotImplementedError` 后再写
        `yield` 会被判为不可达。返回空流是类型正确且行为安全的：
        调用方本应先查 `capabilities.supports_stream`；
        真调了不支持的实现得到空流，比拿到半截内容更容易发现。

        需要流式的实现（如 DeepSeek）直接覆盖本方法。
        """
        if not self.capabilities.supports_stream:
            # 返回空异步迭代器：调用方遍历它不会得到任何元素。
            # 用 `_empty_stream()` 而不是 `yield`：本方法声明为普通 def
            # （返回 AsyncIterator），这样基类不必是异步生成器，
            # 也就不用为了满足类型检查写不可达的 yield。
            return _empty_stream()
        raise NotImplementedError("子类声明了 supports_stream 但没有实现 stream()")

    async def health(self) -> bool:
        """供应商是否可用。默认 True（本地实现无需健康检查）。"""
        return True
