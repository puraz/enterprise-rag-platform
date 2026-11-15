"""LLM Provider 与 ModelGateway 测试。

## mock 策略

自研 httpx 版本用 `httpx.MockTransport` 拦截请求。改用 LangChain 后，
拦截点变成 `ChatOpenAI.ainvoke` 的返回值（`AIMessage`）。

这里用**最简 stub**：一个带 `ainvoke`/`astream` 的假对象注入 Provider。
行为完全可控，不依赖网络或 SDK 内部实现——SDK 升级时测试不会莫名失败。

真实调用测试（integration）保留，验证真实 API 契约。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from rag_api.providers.gateway import ModelGateway, compute_cost
from rag_api.providers.langchain_llm import (
    LangChainLLMError,
    LangChainLLMProvider,
    _classify_error,
    _to_lc_messages,
    _to_usage,
)
from rag_api.providers.llm import LLMMessage, LLMRole, TokenUsage
from rag_contracts.enums import ModelTaskType
from rag_core.settings import get_settings
from rag_core.settings.llm import LLMSettings, TokenPricing, off_peak_time, peak_time

# ==================================================== 测试工具


def _settings(**overrides: object) -> LLMSettings:
    base: dict[str, object] = {
        "provider": "deepseek",
        "api_key": "sk-test-not-real",
        "base_url": "https://api.deepseek.com/v1",
        "chat_model": "deepseek-v4-pro",
        "cheap_model": "deepseek-flash",
    }
    base.update(overrides)
    return LLMSettings(**base)


def _ai_message(
    *,
    content: str = "OK",
    model: str = "deepseek-flash",
    input_tokens: int = 31,
    output_tokens: int = 5,
    cache_read: int = 0,
    reasoning: int = 5,
    finish_reason: str = "stop",
) -> AIMessage:
    """构造带 usage_metadata 的 AIMessage（与真实响应结构一致）。"""
    return AIMessage(
        content=content,
        response_metadata={"model_name": model, "finish_reason": finish_reason},
        usage_metadata={  # type: ignore[arg-type]
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read},
            "output_token_details": {"reasoning": reasoning},
        },
    )


class _FakeChatModel:
    """最小 ChatOpenAI 替身：只需要 `ainvoke` 与 `astream`。"""

    def __init__(
        self,
        *,
        message: AIMessage | None = None,
        error: BaseException | None = None,
        chunks: list[str] | None = None,
    ) -> None:
        self._message = message
        self._error = error
        self._chunks = chunks or []
        self.calls: list[Any] = []

    def bind(self, **_: Any) -> _FakeChatModel:
        return self

    async def ainvoke(self, messages: Any, **_: Any) -> AIMessage:
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        assert self._message is not None
        return self._message

    async def astream(self, messages: Any, **_: Any) -> AsyncIterator[AIMessage]:
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        for text in self._chunks:
            yield AIMessage(content=text)


def _provider(fake: _FakeChatModel, settings: LLMSettings | None = None) -> LangChainLLMProvider:
    return LangChainLLMProvider(settings or _settings(), llm=fake)  # type: ignore[arg-type]


# ==================================================== 用量解析（核心）


def test_usage_extracts_cache_and_reasoning() -> None:
    """从 usage_metadata 提取缓存与推理 token。

    实测 LangChain 字段：
        input_token_details: {cache_read: 800}
        output_token_details: {reasoning: 27}

    这两个是成本核算的全部依据：
    - cache_read 决定有多少输入按折扣价算（DeepSeek 缓存价是 1/50）；
    - reasoning 已含在 output 内，用于诊断思考消耗了多少预算。
    """
    message = _ai_message(input_tokens=1000, output_tokens=100, cache_read=800, reasoning=60)
    usage = _to_usage(message)

    assert usage.input_tokens == 1000
    assert usage.output_tokens == 100
    assert usage.cached_tokens == 800
    assert usage.reasoning_tokens == 60
    assert usage.uncached_input == 200


def test_usage_does_not_double_count() -> None:
    """**关键**：input/output 已包含缓存与推理，不能再相加。

    写成 `input = input_tokens + cache_read` 会让成本翻倍。
    """
    message = _ai_message(input_tokens=100, output_tokens=50, cache_read=80, reasoning=30)
    usage = _to_usage(message)

    assert usage.input_tokens == 100, "input_tokens 不应被加上 cached"
    assert usage.output_tokens == 50, "output_tokens 不应被加上 reasoning"
    assert usage.total_tokens == 150


def test_usage_clamps_inconsistent_cache() -> None:
    """缓存数超过输入总数时截断，不产生负数成本。"""
    message = _ai_message(input_tokens=10, output_tokens=5, cache_read=999)
    usage = _to_usage(message)
    assert usage.cached_tokens == 10
    assert usage.uncached_input == 0


def test_usage_handles_missing_metadata() -> None:
    """没有 usage_metadata 时返回全 0，不崩。

    部分 OpenAI 兼容实现不返回 usage（尤其流式的中间 chunk）。
    """
    usage = _to_usage(AIMessage(content="x"))
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


# ==================================================== 消息转换


def test_message_conversion_covers_all_roles() -> None:
    """三种角色都能转换。新增角色时这里会失败，提醒同步映射表。"""
    converted = _to_lc_messages(
        [
            LLMMessage(role=LLMRole.SYSTEM, content="sys"),
            LLMMessage(role=LLMRole.USER, content="u"),
            LLMMessage(role=LLMRole.ASSISTANT, content="a"),
        ]
    )
    assert [type(m).__name__ for m in converted] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
    ]


# ==================================================== 模型名改写


async def test_uses_actual_model_from_response() -> None:
    """**记账必须用响应里的 model**。

    实测：请求 `deepseek-chat` 返回 `model_name: deepseek-flash`。
    按请求名记账会与供应商账单不符。
    """
    fake = _FakeChatModel(message=_ai_message(model="deepseek-flash"))
    provider = _provider(fake)

    response = await provider.complete(
        [LLMMessage(role=LLMRole.USER, content="hi")], model="deepseek-chat"
    )

    assert response.model == "deepseek-flash"


# ==================================================== 空输出防护


async def test_empty_content_with_length_raises() -> None:
    """思考耗尽 max_tokens 时报错，不静默返回空串。

    实测：max_tokens 太小时预算全用于思考，content 空且
    finish_reason=length。静默返回会被误判为"模型拒答"。
    """
    fake = _FakeChatModel(
        message=_ai_message(content="", output_tokens=20, reasoning=20, finish_reason="length")
    )
    provider = _provider(fake)

    with pytest.raises(LangChainLLMError, match="被截断"):
        await provider.complete([LLMMessage(role=LLMRole.USER, content="hi")], max_tokens=20)


async def test_empty_content_with_stop_is_allowed() -> None:
    """正常结束的空内容合法（模型回了空行），不能一律当错误。"""
    fake = _FakeChatModel(message=_ai_message(content="", finish_reason="stop"))
    provider = _provider(fake)

    response = await provider.complete([LLMMessage(role=LLMRole.USER, content="hi")])
    assert response.content == ""


# ==================================================== 异常分类


@pytest.mark.parametrize(
    ("error_name", "retryable"),
    [
        ("RateLimitError", True),
        ("APITimeoutError", True),
        ("APIConnectionError", True),
        ("InternalServerError", True),
        ("AuthenticationError", False),
        ("PermissionDeniedError", False),
        ("BadRequestError", False),
    ],
)
def test_error_classification(error_name: str, retryable: bool) -> None:
    """SDK 异常正确映射为是否可重试。

    把 401 标成可重试会导致每 0.2s 打一次认证接口，
    既浪费又可能触发供应商风控。
    """
    exc_type = type(error_name, (Exception,), {})
    classified = _classify_error(exc_type("boom"))
    assert classified.retryable is retryable


def test_unknown_error_is_not_retryable() -> None:
    """未知异常保守判定不重试（避免把未知 bug 放大成大量重试）。"""
    assert _classify_error(RuntimeError("未知错误")).retryable is False


# ==================================================== 能力声明


def test_capabilities_for_deepseek() -> None:
    """DeepSeek 声明支持流式、视觉、JSON 模式。"""
    caps = _provider(_FakeChatModel(message=_ai_message())).capabilities

    assert caps.supports_stream is True
    assert caps.supports_vision is True
    assert caps.supports_json_mode is True
    assert caps.max_context == 1_048_576


# ==================================================== 流式


async def test_stream_yields_text_chunks() -> None:
    """流式产出文本增量。"""
    provider = _provider(_FakeChatModel(chunks=["你", "好", "世界"]))

    collected = [
        text async for text in provider.stream([LLMMessage(role=LLMRole.USER, content="hi")])
    ]
    assert collected == ["你", "好", "世界"]


async def test_stream_propagates_classified_error() -> None:
    """流式中的异常也要被分类，不能裸抛。"""
    provider = _provider(_FakeChatModel(error=type("RateLimitError", (Exception,), {})("busy")))

    with pytest.raises(LangChainLLMError) as exc_info:
        async for _ in provider.stream([LLMMessage(role=LLMRole.USER, content="hi")]):
            pass

    assert exc_info.value.retryable is True


# ==================================================== 成本计算


def test_compute_cost_basic() -> None:
    """基础成本：1M 输入 + 0.5M 输出。"""
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
    pricing = TokenPricing(input=Decimal("0.15"), output=Decimal("0.6"))

    total, _, _ = compute_cost(usage, pricing)
    assert total == Decimal("0.45")


def test_compute_cost_with_cache_discount() -> None:
    """缓存命中按折扣价。

    DeepSeek flash 缓存价是常规价的 1/50（$0.003 vs $0.15）。
    不区分会让长系统提示场景成本估算偏高几十倍。
    """
    usage = TokenUsage(input_tokens=1_000_000, cached_tokens=1_000_000)
    pricing = TokenPricing(
        input=Decimal("0.15"), input_cached=Decimal("0.003"), output=Decimal("0.6")
    )

    total, _, _ = compute_cost(usage, pricing)
    assert total == Decimal("0.003")


def test_compute_cost_cache_without_price_uses_full_price() -> None:
    """未配缓存价时按常规价算（宁可高估不可低估）。"""
    usage = TokenUsage(input_tokens=1_000_000, cached_tokens=1_000_000)
    pricing = TokenPricing(input=Decimal("0.15"), output=Decimal("0.6"))

    total, _, _ = compute_cost(usage, pricing)
    assert total == Decimal("0.15")


def test_compute_cost_is_exact_decimal() -> None:
    """成本用 Decimal 精确计算（float 累加会有可见误差）。"""
    usage = TokenUsage(input_tokens=333, output_tokens=777)
    pricing = TokenPricing(input=Decimal("0.123456789"), output=Decimal("0.987654321"))

    total, _, _ = compute_cost(usage, pricing)
    assert isinstance(total, Decimal)
    assert total == (
        Decimal(333) / Decimal(1_000_000) * Decimal("0.123456789")
        + Decimal(777) / Decimal(1_000_000) * Decimal("0.987654321")
    )


# ==================================================== 峰谷计价


def test_peak_pricing_is_double() -> None:
    """峰时单价是谷时 2 倍。"""
    pricing = _settings().pricing_for("deepseek-flash")
    assert pricing is not None

    off = pricing.price_at(off_peak_time())
    peak = pricing.price_at(peak_time())
    assert peak.input == off.input * 2
    assert peak.output == off.output * 2


def test_weekend_is_off_peak() -> None:
    """周末全天谷时。"""
    from datetime import UTC, datetime

    pricing = _settings().pricing_for("deepseek-flash")
    assert pricing is not None
    assert pricing.is_peak(datetime(2026, 11, 21, 2, 0, tzinfo=UTC)) is False


def test_peak_judged_in_utc_not_local() -> None:
    """峰时按 UTC 判断，不受部署地时区影响。

    UTC+8 部署时，本地 09:00-12:00 实际是 UTC 01:00-04:00 的峰时。
    """
    from datetime import UTC, datetime, timedelta, timezone

    pricing = _settings().pricing_for("deepseek-flash")
    assert pricing is not None

    utc_peak = datetime(2026, 11, 16, 2, 0, tzinfo=UTC)
    shanghai = utc_peak.astimezone(timezone(timedelta(hours=8)))
    assert shanghai.hour == 10

    assert pricing.is_peak(utc_peak) is True
    assert pricing.is_peak(shanghai) is True, "同一时刻换时区表示，结果必须一致"


# ==================================================== 路由


def test_routing_uses_cheap_model_for_simple_tasks() -> None:
    """简单任务路由到便宜模型（两模型输出价差 3.3 倍）。"""
    gateway = ModelGateway(
        providers={"deepseek": _provider(_FakeChatModel(message=_ai_message()))},
        settings=_settings(),
    )
    for task in (
        ModelTaskType.QUERY_REWRITE,
        ModelTaskType.MULTI_QUERY,
        ModelTaskType.HYDE,
        ModelTaskType.CONTEXT_COMPRESS,
    ):
        assert gateway.model_for(task) == "deepseek-flash"


def test_routing_uses_strong_model_for_quality_tasks() -> None:
    """质量敏感任务用强模型。"""
    gateway = ModelGateway(
        providers={"deepseek": _provider(_FakeChatModel(message=_ai_message()))},
        settings=_settings(),
    )
    assert gateway.model_for(ModelTaskType.CHAT) == "deepseek-v4-pro"
    assert gateway.model_for(ModelTaskType.EVALUATION) == "deepseek-v4-pro"


def test_routing_is_overridable() -> None:
    """路由表可被覆盖，不写死。"""
    gateway = ModelGateway(
        providers={"deepseek": _provider(_FakeChatModel(message=_ai_message()))},
        settings=_settings(),
        routing={ModelTaskType.CHAT: "deepseek-flash"},
    )
    assert gateway.model_for(ModelTaskType.CHAT) == "deepseek-flash"


# ==================================================== Gateway 端到端（mock）


async def test_gateway_cost_uses_actual_model() -> None:
    """网关成本基于响应里的实际模型。"""
    fake = _FakeChatModel(
        message=_ai_message(model="deepseek-flash", input_tokens=1_000_000, output_tokens=0)
    )
    gateway = ModelGateway(providers={"deepseek": _provider(fake)}, settings=_settings())

    _response, cost = await gateway.complete(
        ModelTaskType.QUERY_REWRITE, [LLMMessage(role=LLMRole.USER, content="hi")]
    )

    assert cost.model == "deepseek-flash"
    assert cost.total_usd == Decimal("0.15")
    assert cost.pricing_tier == "off_peak"


async def test_gateway_unknown_model_costs_zero() -> None:
    """未知模型不猜价格：成本 0 但保留用量，补上定价后能重算。"""
    fake = _FakeChatModel(message=_ai_message(model="brand-new-model"))
    gateway = ModelGateway(providers={"deepseek": _provider(fake)}, settings=_settings())

    _response, cost = await gateway.complete(
        ModelTaskType.CHAT, [LLMMessage(role=LLMRole.USER, content="hi")]
    )

    assert cost.total_usd == Decimal("0")
    assert cost.pricing_tier == "unknown"
    assert cost.input_tokens > 0


async def test_gateway_rejects_unknown_provider() -> None:
    """未知供应商名明确报错，不静默回退。"""
    gateway = ModelGateway(
        providers={"deepseek": _provider(_FakeChatModel(message=_ai_message()))},
        settings=_settings(),
    )
    with pytest.raises(ValueError, match="未知供应商"):
        await gateway.complete(
            ModelTaskType.CHAT,
            [LLMMessage(role=LLMRole.USER, content="hi")],
            provider_name="nope",
        )


# ==================================================== 真实调用（integration）


def _real_provider() -> LangChainLLMProvider:
    settings = get_settings().llm
    if not settings.api_key:
        pytest.skip("未配置 RAG_LLM__API_KEY")
    return LangChainLLMProvider(settings)


@pytest.mark.integration
async def test_real_flash_call() -> None:
    """真实调用 flash：验证与 API 契约一致。

    `max_tokens` 不能太小——思考模式会先消耗预算（实测踩过）。
    """
    settings = get_settings().llm
    provider = _real_provider()
    try:
        response = await provider.complete(
            [LLMMessage(role=LLMRole.USER, content="1+1等于几？只回答数字")],
            model=settings.cheap_model,
            max_tokens=500,
        )
    finally:
        await provider.aclose()

    assert response.model == settings.cheap_model
    assert "2" in response.content
    assert response.usage.input_tokens > 0
    assert response.finish_reason == "stop"


@pytest.mark.integration
async def test_real_pro_call() -> None:
    """真实调用 pro：max_tokens 留足余量（pro 思考更重）。"""
    settings = get_settings().llm
    provider = _real_provider()
    try:
        response = await provider.complete(
            [LLMMessage(role=LLMRole.USER, content="用一句话说明什么是向量检索")],
            model=settings.chat_model,
            max_tokens=1000,
        )
    finally:
        await provider.aclose()

    assert response.model == settings.chat_model
    assert len(response.content) > 5
    assert response.finish_reason == "stop"


@pytest.mark.integration
async def test_real_stream() -> None:
    """真实流式：能逐块收到文本。"""
    settings = get_settings().llm
    provider = _real_provider()
    chunks: list[str] = []
    try:
        async for text in provider.stream(
            [LLMMessage(role=LLMRole.USER, content="数到5，只输出数字")],
            model=settings.cheap_model,
            max_tokens=600,
        ):
            chunks.append(text)
    finally:
        await provider.aclose()

    assert chunks, "流式应至少收到一个块"
    assert any(ch.isdigit() for ch in "".join(chunks))


@pytest.mark.integration
async def test_real_gateway_cost_difference() -> None:
    """真实调用：验证路由到正确模型，且**单价**符合配置。

    **不断言总成本谁更高**：单次调用的成本 = token 数 × 单价，
    而 token 数波动很大（思考长度不定）。实测出现过 pro 只花 6 个输入
    token、flash 花了 39 个的情况，此时总成本 pro 反而更低。
    断言总成本会变成偶发失败的脆弱测试。

    正确做法：断言**单价**（配置决定，稳定）与模型路由（逻辑决定，稳定）。
    """
    settings = get_settings().llm
    provider = _real_provider()
    gateway = ModelGateway(providers={"deepseek": provider}, settings=settings)
    try:
        prompt = [LLMMessage(role=LLMRole.USER, content="回复OK")]
        _r1, flash_cost = await gateway.complete(
            ModelTaskType.QUERY_REWRITE, prompt, max_tokens=500
        )
        _r2, pro_cost = await gateway.complete(ModelTaskType.CHAT, prompt, max_tokens=500)
    finally:
        await gateway.aclose()

    # 路由正确
    assert flash_cost.model == settings.cheap_model
    assert pro_cost.model == settings.chat_model
    # 成本被真实计算（非 0）
    assert flash_cost.total_usd > Decimal("0")
    assert pro_cost.total_usd > Decimal("0")
    # **单价**必须满足配置：pro 的输入价高于 flash（稳定断言）
    assert pro_cost.unit_price_input > flash_cost.unit_price_input
    assert pro_cost.unit_price_output > flash_cost.unit_price_output


@pytest.mark.integration
async def test_real_health() -> None:
    """Provider 健康检查。"""
    provider = _real_provider()
    try:
        assert await provider.health() is True
    finally:
        await provider.aclose()


@pytest.mark.integration
async def test_real_json_mode() -> None:
    """JSON 模式返回可解析的 JSON。"""
    settings = get_settings().llm
    provider = _real_provider()
    try:
        response = await provider.complete(
            [
                LLMMessage(
                    role=LLMRole.USER,
                    content='把这句话抽取为JSON：{"topic": ..., "intent": ...}。内容：如何申请年假',
                )
            ],
            model=settings.cheap_model,
            max_tokens=600,
            json_mode=True,
        )
    finally:
        await provider.aclose()

    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    assert isinstance(json.loads(text), dict)
