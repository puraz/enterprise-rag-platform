"""LLM Provider 与 ModelGateway 测试。

分两类：

- **单测（mock）**：覆盖用量解析、成本计算、路由、异常映射、空输出防护。
  这类测试必须不依赖网络，才能在任何环境跑。
- **真实调用（integration）**：验证实现与真实 API 的契约一致。
  这类会产生少量费用，用 `max_tokens` 极小 + 便宜模型控制成本。

为什么两类都要：
- 只有 mock：无法发现"供应商字段改名"这类真实契约漂移（本项目已踩到
  模型名被改写、思考 token 消耗 max_tokens 两个坑）；
- 只有真实调用：慢、要钱、不稳定，无法覆盖所有边界。
"""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from rag_api.providers.deepseek import DeepSeekError, DeepSeekProvider
from rag_api.providers.gateway import ModelGateway, compute_cost
from rag_api.providers.llm import (
    LLMMessage,
    LLMProviderError,
    LLMRole,
    TokenUsage,
)
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


def _mock_response(payload: dict[str, object], *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=payload)


def _provider_with(handler: object, settings: LLMSettings | None = None) -> DeepSeekProvider:
    """用 mock transport 构造 Provider（不发真实请求）。"""
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    client = httpx.AsyncClient(
        base_url="https://api.deepseek.com/v1",
        transport=transport,
        headers={"Authorization": "Bearer sk-test"},
    )
    return DeepSeekProvider(settings or _settings(), client=client)


def _chat_payload(
    *,
    content: str = "OK",
    model: str = "deepseek-flash",
    input_tokens: int = 31,
    output_tokens: int = 5,
    cached: int = 0,
    reasoning: int = 5,
    finish_reason: str = "stop",
) -> dict[str, object]:
    return {
        "id": "test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "prompt_cache_hit_tokens": cached,
            "prompt_cache_miss_tokens": input_tokens - cached,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


# ==================================================== 用量解析


async def test_usage_parsing_does_not_double_count() -> None:
    """**用量解析不能做加法**：prompt_tokens 已含缓存，completion 已含 reasoning。

    实测字段：
        prompt_tokens=31, completion_tokens=5,
        prompt_cache_hit_tokens=0, reasoning_tokens=5

    如果写成 `input = prompt_tokens + cached_tokens`，成本会翻倍。
    """
    provider = _provider_with(lambda request: _mock_response(_chat_payload()))
    try:
        response = await provider.complete([LLMMessage(LLMRole.USER, "hi")])
    finally:
        await provider.aclose()

    assert response.usage.input_tokens == 31
    assert response.usage.output_tokens == 5
    assert response.usage.cached_tokens == 0
    assert response.usage.reasoning_tokens == 5
    # reasoning 已含在 output 内，不应相加
    assert response.usage.total_tokens == 36


async def test_usage_with_cache_hit() -> None:
    """缓存命中时，cached_tokens 是 input 的子集。"""
    provider = _provider_with(
        lambda request: _mock_response(
            _chat_payload(input_tokens=1000, cached=800, output_tokens=100)
        )
    )
    try:
        response = await provider.complete([LLMMessage(LLMRole.USER, "hi")])
    finally:
        await provider.aclose()

    assert response.usage.input_tokens == 1000
    assert response.usage.cached_tokens == 800
    assert response.usage.uncached_input == 200


async def test_usage_clamps_inconsistent_cache_value() -> None:
    """缓存数超过输入总数时截断并告警，不产生负数成本。

    供应商字段语义变化时（比如以后改成"额外赠送"），
    不截断会让 uncached_input 变负数，成本算出负值。
    """
    provider = _provider_with(
        lambda request: _mock_response(_chat_payload(input_tokens=10, cached=999))
    )
    try:
        response = await provider.complete([LLMMessage(LLMRole.USER, "hi")])
    finally:
        await provider.aclose()

    assert response.usage.cached_tokens == 10
    assert response.usage.uncached_input == 0


# ==================================================== 模型名改写


async def test_uses_actual_model_from_response() -> None:
    """**记账必须用响应里的 model**，不能用请求的。

    实测：请求 `deepseek-chat` 返回 `model: deepseek-flash`。
    按请求名记账会与供应商账单不符（按新模型计费、按旧价核算）。
    """
    provider = _provider_with(lambda request: _mock_response(_chat_payload(model="deepseek-flash")))
    try:
        response = await provider.complete([LLMMessage(LLMRole.USER, "hi")], model="deepseek-chat")
    finally:
        await provider.aclose()

    assert response.model == "deepseek-flash", "应使用响应中的实际模型名"


# ==================================================== 空输出防护


async def test_empty_content_with_length_raises() -> None:
    """思考耗尽 max_tokens 时必须报错，不能静默返回空串。

    实测：max_tokens=20 时 20 个 token 全在思考，content 为空
    且 finish_reason=length。静默返回空串会被误判为"模型拒答"。
    """
    provider = _provider_with(
        lambda request: _mock_response(
            _chat_payload(content="", output_tokens=20, reasoning=20, finish_reason="length")
        )
    )
    try:
        with pytest.raises(DeepSeekError, match="被截断"):
            await provider.complete([LLMMessage(LLMRole.USER, "hi")], max_tokens=20)
    finally:
        await provider.aclose()


async def test_empty_content_with_stop_is_allowed() -> None:
    """正常结束但内容为空是合法响应（比如模型只回了空行）。

    不能一律当错误——那会让合法空回答变成 500。
    """
    provider = _provider_with(
        lambda request: _mock_response(_chat_payload(content="", finish_reason="stop"))
    )
    try:
        response = await provider.complete([LLMMessage(LLMRole.USER, "hi")])
    finally:
        await provider.aclose()

    assert response.content == ""
    assert response.finish_reason == "stop"


# ==================================================== 错误映射


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (429, True),  # 限流：可重试
        (500, True),  # 服务端错误：可重试
        (503, True),
        (401, False),  # 密钥失效：重试无用
        (400, False),  # 请求非法：重试无用
    ],
)
async def test_http_error_maps_to_retryable_flag(status: int, retryable: bool) -> None:
    """HTTP 错误正确映射为是否可重试。

    这个标记决定 ResilientCaller 是否重试：
    把 401 标成可重试会导致每 0.2s 打一次认证接口，
    既浪费又可能触发供应商风控。
    """
    provider = _provider_with(lambda request: httpx.Response(status_code=status))
    try:
        with pytest.raises(LLMProviderError) as exc_info:
            await provider.complete([LLMMessage(LLMRole.USER, "hi")])
    finally:
        await provider.aclose()

    assert exc_info.value.retryable is retryable


async def test_timeout_is_retryable() -> None:
    """超时可重试。"""

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout", request=request)

    provider = _provider_with(timeout_handler)
    try:
        with pytest.raises(DeepSeekError) as exc_info:
            await provider.complete([LLMMessage(LLMRole.USER, "hi")])
    finally:
        await provider.aclose()

    assert exc_info.value.retryable is True


async def test_empty_choices_raises() -> None:
    """空 choices 是畸形响应，不能当成功。"""
    provider = _provider_with(
        lambda request: _mock_response({"model": "x", "choices": [], "usage": {}})
    )
    try:
        with pytest.raises(DeepSeekError, match="空 choices"):
            await provider.complete([LLMMessage(LLMRole.USER, "hi")])
    finally:
        await provider.aclose()


# ==================================================== 成本计算


def test_compute_cost_basic() -> None:
    """基础成本：1M 输入 + 0.5M 输出。"""
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
    pricing = TokenPricing(input=Decimal("0.15"), output=Decimal("0.6"))

    total, _, _ = compute_cost(usage, pricing)
    # 1M * 0.15 + 0.5M * 0.6 = 0.15 + 0.30 = 0.45
    assert total == Decimal("0.45")


def test_compute_cost_with_cache_discount() -> None:
    """缓存命中按折扣价，不按常规输入价。

    DeepSeek flash 的缓存价是常规价的 1/50（$0.003 vs $0.15）。
    不区分会让长系统提示场景的成本估算偏高几十倍。
    """
    usage = TokenUsage(input_tokens=1_000_000, cached_tokens=1_000_000, output_tokens=0)
    pricing = TokenPricing(
        input=Decimal("0.15"), input_cached=Decimal("0.003"), output=Decimal("0.6")
    )

    total, _, _ = compute_cost(usage, pricing)
    assert total == Decimal("0.003"), f"全缓存命中应只花 0.003，实际 {total}"


def test_compute_cost_cache_without_price_uses_full_price() -> None:
    """未配置缓存价时，缓存部分按常规价算（宁可高估不可低估）。

    低估会导致预算超支：以为花了 $10 实际花了 $50，
    等发现时已经超了。
    """
    usage = TokenUsage(input_tokens=1_000_000, cached_tokens=1_000_000)
    pricing = TokenPricing(input=Decimal("0.15"), output=Decimal("0.6"))

    total, _, _ = compute_cost(usage, pricing)
    assert total == Decimal("0.15")


def test_compute_cost_is_exact_decimal() -> None:
    """成本用 Decimal 精确计算。

    float 累加会产生可见误差（A3-5 已证明 3×0.1+3×0.2 != 0.9），
    而成本账必须精确到分。
    """
    usage = TokenUsage(input_tokens=333, output_tokens=777)
    pricing = TokenPricing(input=Decimal("0.123456789"), output=Decimal("0.987654321"))

    total, _, _ = compute_cost(usage, pricing)
    # 结果必须是 Decimal 且能精确表示
    assert isinstance(total, Decimal)
    assert total == (
        Decimal(333) / Decimal(1_000_000) * Decimal("0.123456789")
        + Decimal(777) / Decimal(1_000_000) * Decimal("0.987654321")
    )


# ==================================================== 峰谷计价


def test_peak_pricing_is_double() -> None:
    """峰时单价是谷时的 2 倍（DeepSeek 官方定价结构）。"""
    settings = _settings()
    pricing = settings.pricing_for("deepseek-flash")
    assert pricing is not None

    off = pricing.price_at(off_peak_time())
    peak = pricing.price_at(peak_time())
    assert peak.input == off.input * 2
    assert peak.output == off.output * 2


def test_weekend_is_always_off_peak() -> None:
    """周末全天算谷时（官方文档：峰时仅工作日）。"""
    from datetime import UTC, datetime

    settings = _settings()
    pricing = settings.pricing_for("deepseek-flash")
    assert pricing is not None

    # 2026-11-21 是周六
    saturday_2am = datetime(2026, 11, 21, 2, 0, tzinfo=UTC)
    assert pricing.is_peak(saturday_2am) is False


def test_peak_windows_are_utc() -> None:
    """峰时按 UTC 判断，不受部署地时区影响。

    在 UTC+8 部署时，本地 09:00-12:00 实际是 UTC 01:00-04:00——
    用本地时区判断会错位。
    """
    from datetime import UTC, datetime, timedelta, timezone

    settings = _settings()
    pricing = settings.pricing_for("deepseek-flash")
    assert pricing is not None

    utc_peak = datetime(2026, 11, 16, 2, 0, tzinfo=UTC)  # 周一 02:00 UTC
    # 同一时刻用 UTC+8 表示
    shanghai = utc_peak.astimezone(timezone(timedelta(hours=8)))
    assert shanghai.hour == 10

    assert pricing.is_peak(utc_peak) is True
    assert pricing.is_peak(shanghai) is True, "换成 +08:00 表示仍是同一时刻，结果必须一致"


# ==================================================== 路由


def test_routing_uses_cheap_model_for_simple_tasks() -> None:
    """简单任务路由到便宜模型。

    成本差异是真实的：本项目两模型输出价差 3.3 倍。
    """
    settings = _settings()
    gateway = ModelGateway(
        providers={"deepseek": _provider_with(lambda r: _mock_response(_chat_payload()))},
        settings=settings,
    )

    assert gateway.model_for(ModelTaskType.QUERY_REWRITE) == "deepseek-flash"
    assert gateway.model_for(ModelTaskType.MULTI_QUERY) == "deepseek-flash"
    assert gateway.model_for(ModelTaskType.HYDE) == "deepseek-flash"
    assert gateway.model_for(ModelTaskType.CONTEXT_COMPRESS) == "deepseek-flash"


def test_routing_uses_strong_model_for_quality_tasks() -> None:
    """质量敏感任务用强模型。"""
    settings = _settings()
    gateway = ModelGateway(
        providers={"deepseek": _provider_with(lambda r: _mock_response(_chat_payload()))},
        settings=settings,
    )

    assert gateway.model_for(ModelTaskType.CHAT) == "deepseek-v4-pro"
    assert gateway.model_for(ModelTaskType.EVALUATION) == "deepseek-v4-pro"


def test_routing_is_overridable() -> None:
    """路由表可被配置覆盖（不写死）。"""
    settings = _settings()
    gateway = ModelGateway(
        providers={"deepseek": _provider_with(lambda r: _mock_response(_chat_payload()))},
        settings=settings,
        routing={ModelTaskType.CHAT: "deepseek-flash"},
    )
    assert gateway.model_for(ModelTaskType.CHAT) == "deepseek-flash"


# ==================================================== Gateway 端到端（mock）


async def test_gateway_returns_cost_with_actual_model() -> None:
    """网关返回的成本基于**响应里的实际模型**。"""
    settings = _settings()
    provider = _provider_with(
        lambda request: _mock_response(
            _chat_payload(model="deepseek-flash", input_tokens=1_000_000, output_tokens=0, cached=0)
        )
    )
    gateway = ModelGateway(providers={"deepseek": provider}, settings=settings)
    try:
        _response, cost = await gateway.complete(
            ModelTaskType.QUERY_REWRITE, [LLMMessage(LLMRole.USER, "hi")]
        )
    finally:
        await gateway.aclose()

    assert cost.model == "deepseek-flash"
    assert cost.input_tokens == 1_000_000
    # 谷时输入价 $0.15
    assert cost.total_usd == Decimal("0.15")
    assert cost.pricing_tier == "off_peak"


async def test_gateway_unknown_model_costs_zero() -> None:
    """未知模型不猜价格：成本记 0 但保留 token 用量。

    比"算不出就不记"好：至少能看到用量，补上定价后能重算历史。
    """
    settings = _settings()
    provider = _provider_with(lambda request: _mock_response(_chat_payload(model="some-new-model")))
    gateway = ModelGateway(providers={"deepseek": provider}, settings=settings)
    try:
        _response, cost = await gateway.complete(
            ModelTaskType.CHAT, [LLMMessage(LLMRole.USER, "hi")]
        )
    finally:
        await gateway.aclose()

    assert cost.model == "some-new-model"
    assert cost.total_usd == Decimal("0")
    assert cost.pricing_tier == "unknown"
    assert cost.input_tokens > 0, "用量必须保留"


async def test_gateway_rejects_unknown_provider() -> None:
    """未知供应商名要明确报错，不静默回退。"""
    settings = _settings()
    gateway = ModelGateway(
        providers={"deepseek": _provider_with(lambda r: _mock_response(_chat_payload()))},
        settings=settings,
    )
    with pytest.raises(ValueError, match="未知供应商"):
        await gateway.complete(
            ModelTaskType.CHAT, [LLMMessage(LLMRole.USER, "hi")], provider_name="nope"
        )


# ==================================================== 真实调用（integration）


@pytest.mark.integration
async def test_real_deepseek_flash_call() -> None:
    """真实调用 flash 模型，验证与 API 契约一致。

    用 flash + 小 max_tokens 控制成本（单次约 $0.0001）。
    `max_tokens` 不能太小：思考模式会先消耗预算。
    """
    settings = get_settings().llm
    if not settings.api_key:
        pytest.skip("未配置 RAG_LLM__API_KEY")

    provider = DeepSeekProvider(settings)
    try:
        response = await provider.complete(
            [LLMMessage(LLMRole.USER, "1+1等于几？只回答数字")],
            model=settings.cheap_model,
            max_tokens=500,
        )
    finally:
        await provider.aclose()

    assert response.model == settings.cheap_model
    assert "2" in response.content
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.finish_reason == "stop"


@pytest.mark.integration
async def test_real_deepseek_pro_call() -> None:
    """真实调用 pro 模型。max_tokens 留足余量——pro 的思考更重。"""
    settings = get_settings().llm
    if not settings.api_key:
        pytest.skip("未配置 RAG_LLM__API_KEY")

    provider = DeepSeekProvider(settings)
    try:
        response = await provider.complete(
            [LLMMessage(LLMRole.USER, "用一句话说明什么是向量检索")],
            model=settings.chat_model,
            max_tokens=1000,
        )
    finally:
        await provider.aclose()

    assert response.model == settings.chat_model
    assert len(response.content) > 5
    assert response.finish_reason == "stop"


@pytest.mark.integration
async def test_real_gateway_routing_and_cost() -> None:
    """真实调用的端到端：路由 + 成本计算。

    验证两个模型的真实成本差异（flash 应显著便宜）。
    """
    settings = get_settings().llm
    if not settings.api_key:
        pytest.skip("未配置 RAG_LLM__API_KEY")

    provider = DeepSeekProvider(settings)
    gateway = ModelGateway(providers={"deepseek": provider}, settings=settings)
    try:
        prompt = [LLMMessage(LLMRole.USER, "回复OK")]

        _flash_resp, flash_cost = await gateway.complete(
            ModelTaskType.QUERY_REWRITE, prompt, max_tokens=500
        )
        _pro_resp, pro_cost = await gateway.complete(ModelTaskType.CHAT, prompt, max_tokens=500)
    finally:
        await gateway.aclose()

    assert flash_cost.model == settings.cheap_model
    assert pro_cost.model == settings.chat_model
    assert pro_cost.total_usd > flash_cost.total_usd, "强模型应更贵"
    # 价格比约 4.4 倍（输入 0.66/0.15）到 3.3 倍（输出 1.98/0.6）
    assert flash_cost.total_usd > Decimal("0")


@pytest.mark.integration
async def test_real_provider_health() -> None:
    """Provider 健康检查（真实探活）。"""
    settings = get_settings().llm
    if not settings.api_key:
        pytest.skip("未配置 RAG_LLM__API_KEY")

    provider = DeepSeekProvider(settings)
    try:
        assert await provider.health() is True
    finally:
        await provider.aclose()


@pytest.mark.integration
async def test_real_json_mode() -> None:
    """JSON 模式能返回可解析的 JSON（结构化抽取要用）。"""
    settings = get_settings().llm
    if not settings.api_key:
        pytest.skip("未配置 RAG_LLM__API_KEY")

    provider = DeepSeekProvider(settings)
    try:
        response = await provider.complete(
            [
                LLMMessage(
                    LLMRole.USER,
                    '把这句话抽取为JSON：{"topic": ..., "intent": ...}。内容：如何申请年假',
                )
            ],
            model=settings.cheap_model,
            max_tokens=600,
            json_mode=True,
        )
    finally:
        await provider.aclose()

    # 思考模式下 content 可能带 markdown 包裹，先剥掉
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
