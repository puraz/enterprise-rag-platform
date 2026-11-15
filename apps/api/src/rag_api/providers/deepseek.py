"""DeepSeek Provider（OpenAI 兼容协议）。

## 为什么用 httpx 而不是 openai SDK

| 方案 | 问题 |
|---|---|
| `openai` SDK | 引入重量依赖；SDK 自带重试（与统一 ResilientCaller 冲突）；
  版本升级会改变错误类型 |
| **httpx 直连（选这个）** | 完全可控；重试只在 ResilientCaller 一处；
  换供应商只是换 base_url + 少量字段 |

DeepSeek 的 API 与 OpenAI Chat Completions 兼容，
所以这套实现可以直接指向任何 OpenAI 兼容端点（通义、智谱、Moonshot、
本地 vLLM、Ollama 的 /v1），**换供应商只改配置**。

## 用量字段的坑（实测）

DeepSeek 返回的 usage（2026-11 实测）：

    {
      "prompt_tokens": 31,
      "completion_tokens": 5,
      "total_tokens": 36,
      "prompt_tokens_details": {"cached_tokens": 0},
      "completion_tokens_details": {"reasoning_tokens": 5},
      "prompt_cache_hit_tokens": 0,
      "prompt_cache_miss_tokens": 31
    }

三个必须注意的点：

1. **`cached_tokens` 是 `prompt_tokens` 的子集**，不是额外量。
   写成 `input = prompt_tokens + cached_tokens` 会让成本翻倍；
2. **`reasoning_tokens` 已含在 `completion_tokens` 里**，不要重复计费；
3. 缓存命中数以 `prompt_cache_hit_tokens` 为准（顶层字段更明确），
   `prompt_tokens_details.cached_tokens` 作为回退。

## 模型名的坑

请求 `deepseek-chat` 时响应返回 `model: "deepseek-flash"` ——
已下线的模型名会被路由到新模型。**记账必须用响应里的 model**，
否则会按旧价算、按新模型计费，账永远对不上。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from rag_api.providers.llm import (
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


class DeepSeekError(LLMProviderError):
    """DeepSeek 调用失败。"""


class DeepSeekProvider(LLMProvider):
    """DeepSeek（及任意 OpenAI 兼容端点）实现。"""

    name = "deepseek"

    def __init__(
        self,
        settings: LLMSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not settings.api_key:
            raise DeepSeekError("未配置 RAG_LLM__API_KEY", retryable=False)
        self._settings = settings
        self._base_url = settings.base_url.rstrip("/")
        # 注入 client 便于测试（可以传入 mock transport）
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            timeout=settings.timeout_s,
        )
        self._owns_client = client is None

    @property
    def capabilities(self) -> ProviderCapabilities:
        """能力声明。

        `max_context` 取 1M（官方文档：两个模型都是 1048576）。
        保守一点按 **100 万** 声明，实际裁剪时还会留出输出空间。
        """
        return ProviderCapabilities(
            supports_stream=True,
            supports_vision=True,  # flash 支持图像输入
            supports_json_mode=True,  # 用 response_format 实现
            is_local=False,
            max_context=1_048_576,
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
        """非流式补全。

        **不做重试**：失败直接抛出，由 ResilientCaller 决定是否重试。
        """
        target_model = model or self._settings.chat_model
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": [
                {"role": message.role.value, "content": message.content} for message in messages
            ],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # 限制思考深度：模型默认 high，会把 max_tokens 大量消耗在思考上，
        # 导致正文为空且 finish_reason=length（实测验证）
        if self._settings.reasoning_effort:
            payload["reasoning_effort"] = self._settings.reasoning_effort
        if json_mode:
            # OpenAI 兼容的 JSON 模式。DeepSeek 支持 response_format
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
                timeout=timeout_s or self._settings.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise DeepSeekError(f"{target_model} 调用超时", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise DeepSeekError(f"{target_model} 网络错误", retryable=True) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code == 429:
            raise DeepSeekError("供应商限流", retryable=True)
        if response.status_code in (401, 403):
            # 密钥失效：不重试（重试一万次也一样），但要能触发熔断
            raise DeepSeekError("供应商认证失败", retryable=False)
        if response.status_code >= 500:
            raise DeepSeekError(f"供应商 {response.status_code}", retryable=True)
        if response.status_code >= 400:
            # 4xx 其他：请求本身有问题（上下文超长、参数非法），不重试
            raise DeepSeekError(f"请求被拒绝（{response.status_code}）", retryable=False)

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise DeepSeekError("供应商返回非 JSON", retryable=False) from exc

        parsed = self._parse_response(data, latency_ms=latency_ms, requested_model=target_model)

        # 思考模式下 max_tokens 可能被思考过程耗尽，导致 content 为空
        # 且 finish_reason=length（实测：同一问题思考长度会波动，
        # 300 token 有时够有时不够）。
        # 这不是错误——模型确实执行了并被截断，但调用方拿不到答案。
        # 明确警告而不是静默返回空字符串：空答案在 RAG 里会被当成
        # "模型拒答"，掩盖真实原因。
        if not parsed.content and parsed.finish_reason == "length":
            log.warning(
                "llm_output_empty_truncated",
                model=parsed.model,
                max_tokens=max_tokens,
                reasoning_tokens=parsed.usage.reasoning_tokens,
                note="max_tokens 被思考过程耗尽，需提高 max_tokens 或降低 reasoning_effort",
            )
            raise DeepSeekError(
                f"输出被截断：{parsed.usage.reasoning_tokens} 个 token 消耗在思考过程，"
                f"max_tokens={max_tokens} 不足以产出正文",
                retryable=True,
            )

        return parsed

    def _parse_response(
        self, data: dict[str, Any], *, latency_ms: int, requested_model: str
    ) -> LLMResponse:
        """解析响应体。

        单独成方法便于单测：不需要真实网络就能覆盖各种畸形响应。
        """
        choices = data.get("choices") or []
        if not choices:
            raise DeepSeekError("供应商返回空 choices", retryable=False)

        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        finish_reason = choices[0].get("finish_reason")

        # **用响应里的 model，不用请求里的**（见模块 docstring 的"模型名的坑"）
        actual_model = str(data.get("model") or requested_model)
        if actual_model != requested_model:
            log.info(
                "model_redirected",
                requested=requested_model,
                actual=actual_model,
                note="供应商把请求路由到了另一个模型，成本按实际模型计算",
            )

        usage = self._parse_usage(data.get("usage") or {})

        return LLMResponse(
            content=content,
            model=actual_model,
            usage=usage,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )

    def _parse_usage(self, usage: dict[str, Any]) -> TokenUsage:
        """解析 token 用量。

        字段优先级与坑见模块 docstring。这里的关键是**不做加法**：
        `prompt_tokens` 已经包含缓存部分，`completion_tokens` 已包含 reasoning。
        """
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)

        # 缓存命中：优先顶层字段（更明确），回退到 details
        cached = int(usage.get("prompt_cache_hit_tokens") or 0)
        if cached == 0:
            cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)

        reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)

        # 一致性检查：缓存数不能超过输入总数（超了说明字段语义变了）
        if cached > input_tokens:
            log.warning(
                "usage_cached_exceeds_input",
                input_tokens=input_tokens,
                cached_tokens=cached,
                note="供应商字段语义可能变化，已按输入总数截断",
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

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[str]:
        """流式补全（SSE）。

        只产出文本增量，不产出用量——SSE 的 usage 在最后一个 chunk 里，
        需要的话调用方自己收集。A11 的 SSE 接口会用这个。
        """
        target_model = model or self._settings.chat_model
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": [
                {"role": message.role.value, "content": message.content} for message in messages
            ],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self._settings.reasoning_effort:
            payload["reasoning_effort"] = self._settings.reasoning_effort

        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                timeout=timeout_s or self._settings.timeout_s,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise DeepSeekError(
                        f"流式请求失败（{response.status_code}）: {body[:200]!r}",
                        retryable=response.status_code >= 500,
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:].strip()
                    if chunk == "[DONE]":
                        break
                    import json

                    try:
                        parsed = json.loads(chunk)
                    except ValueError:
                        continue
                    for choice in parsed.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            yield str(text)
        except httpx.HTTPError as exc:
            raise DeepSeekError("流式请求网络错误", retryable=True) from exc

    async def health(self) -> bool:
        """探活：调一次最小请求。

        用 `max_tokens=1` 让成本可忽略（约 $0.000001）。
        不做缓存：健康检查必须反映真实可用性。
        """
        try:
            await self.complete(
                [LLMMessage(role=LLMRole.USER, content="hi")],
                model=self._settings.cheap_model,
                # 思考模式会先消耗预算，max_tokens=1 必然被截断导致报错。
                # 留足余量：健康检查的真实成本仍可忽略（约 $0.0001）
                max_tokens=100,
                timeout_s=10.0,
            )
            return True
        except Exception as exc:
            log.warning("provider_health_failed", error=type(exc).__name__)
            return False

    async def aclose(self) -> None:
        """关闭 HTTP 客户端（只关自己创建的）。"""
        if self._owns_client:
            await self._client.aclose()
