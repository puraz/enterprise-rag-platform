"""ModelGateway：按任务分级路由 + 成本计算 + 用量落库。

## 为什么需要网关而不是直接用 Provider

`docs/PROMPT.md` 的要求：

> 统一管理界面是「模型网关 ModelGateway」：按任务类型（改写 / 多查询生成 /
> HyDE / 压缩 / 生成 / 评估 judge）选模型，支持按任务分级路由
> （简单任务走便宜小模型，复杂任务走强模型）、主备降级、权重灰度。

成本差异是真实的：本项目两个模型的输出价差 **3.3 倍**
（flash $1.2 vs pro $3.96 / 1M）。改写一次查询用 pro 是纯浪费。

## 路由规则（可配置，不写死）

| 任务 | 默认模型 | 理由 |
|---|---|---|
| `chat`（生成答案） | pro | 质量最重要，用户直接看到 |
| `query_rewrite` | flash | 机械改写，便宜模型足够 |
| `multi_query` | flash | 生成变体，容错高 |
| `hyde` | flash | 假设文档，只需语义相近 |
| `context_compress` | flash | 抽取式压缩 |
| `evaluation` | pro | judge 需要判断力 |

路由表用 `dict[ModelTaskType, str]` 表达，换模型只改配置。

## 成本计算必须用"实际模型 + 当时单价"

两个实测踩过的坑：

1. **模型名会被改写**：请求 `deepseek-chat` 返回 `model: deepseek-flash`。
   按请求名算账会与账单不符；
2. **峰谷价差 2 倍**：同一模型 UTC 峰时 / 谷时单价不同。
   必须按调用时刻取单价，不能用固定值。

所以 `record_call` 里：`pricing.price_at(call_time)` + `response.model`。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from rag_api.providers.llm import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    TokenUsage,
)
from rag_contracts.enums import ModelTaskType
from rag_core.logging import get_logger
from rag_core.resilience import ResilientCaller
from rag_core.settings.llm import LLMSettings, TokenPricing

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CallCost:
    """一次调用的成本明细。

    保留中间量（单价、各部分 token）而不是只给总额：
    出账目问题时能一眼看出是哪一项算错了。
    """

    model: str
    pricing_tier: str
    """`peak` 或 `off_peak`。"""
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    unit_price_input: Decimal
    unit_price_cached: Decimal | None
    unit_price_output: Decimal
    total_usd: Decimal


def compute_cost(usage: TokenUsage, pricing: TokenPricing) -> tuple[Decimal, Decimal, Decimal]:
    """按 TokenPricing 计算成本，返回 (总额, 输入单价, 输出单价)。

    公式（单位 USD/1M tokens）：

        uncached_input = input_tokens - cached_tokens
        cost = uncached_input / 1e6 * price.input
             + cached_tokens   / 1e6 * price.input_cached   （若配置了）
             + output_tokens   / 1e6 * price.output

    **用 Decimal 而不是 float**：float 累加十万条会有可见误差，
    账目对不上（A3-5 已用测试证明 3×0.1+3×0.2 在 float 下 != 0.9）。

    缓存未配置单价时，缓存部分按常规输入价计：
    宁可高估成本，也不要因为漏算而低估（低估会导致预算超支）。
    """
    million = Decimal(1_000_000)
    uncached = Decimal(usage.uncached_input)
    cached = Decimal(usage.cached_tokens)
    output = Decimal(usage.output_tokens)

    cached_price = pricing.input_cached if pricing.input_cached is not None else pricing.input

    total = (
        uncached / million * pricing.input
        + cached / million * cached_price
        + output / million * pricing.output
    )
    return total, pricing.input, pricing.output


class ModelGateway:
    """模型调用网关：分级路由 + 韧性包装 + 成本计算。

    **不直接落库**：它返回 `CallCost`，由 service 层配合
    `ModelCallRepository` 写入 `model_calls`。这样网关可以独立单测，
    不需要数据库。
    """

    def __init__(
        self,
        *,
        providers: dict[str, LLMProvider],
        settings: LLMSettings,
        callers: dict[str, ResilientCaller] | None = None,
        routing: dict[ModelTaskType, str] | None = None,
    ) -> None:
        """
        Args:
            providers: 供应商名 → 实例。通常只有主供应商，A5 后期可加备用。
            callers: 供应商名 → 韧性包装器。未提供则为每个供应商建一个。
        """
        if not providers:
            raise ValueError("至少需要一个 LLM provider")

        self._providers = providers
        self._settings = settings
        self._callers = callers or {name: ResilientCaller(f"llm:{name}") for name in providers}
        # 默认路由表：简单任务走便宜模型
        self._routing: dict[ModelTaskType, str] = routing or {
            ModelTaskType.CHAT: settings.chat_model,
            ModelTaskType.QUERY_REWRITE: settings.cheap_model,
            ModelTaskType.MULTI_QUERY: settings.cheap_model,
            ModelTaskType.HYDE: settings.cheap_model,
            ModelTaskType.CONTEXT_COMPRESS: settings.cheap_model,
            ModelTaskType.EVALUATION: settings.chat_model,
            # embedding / rerank 走别的 Provider 类型（A5 后续）
            ModelTaskType.EMBEDDING: settings.cheap_model,
            ModelTaskType.RERANK: settings.cheap_model,
        }

    def model_for(self, task: ModelTaskType) -> str:
        """按任务类型选模型。未知任务回退到 chat_model（宁可贵不可错）。"""
        return self._routing.get(task, self._settings.chat_model)

    async def complete(
        self,
        task: ModelTaskType,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        provider_name: str | None = None,
        idempotent: bool = True,
    ) -> tuple[LLMResponse, CallCost]:
        """执行一次调用，返回 (响应, 成本明细)。

        `idempotent=False` 时**不重试**（例如带副作用的调用）。
        普通补全是幂等的（同样的输入得到同样输出，不计费重复）——
        注意：重试会**产生多次计费**，所以成本要按实际调用次数累加，
        不能只记第一次。
        """
        resolved_provider = provider_name or self._settings.provider
        provider = self._providers.get(resolved_provider)
        if provider is None:
            raise ValueError(f"未知供应商 {resolved_provider!r}")
        caller = self._callers[resolved_provider]

        target_model = model or self.model_for(task)

        async def call_once() -> LLMResponse:
            return await provider.complete(
                messages,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )

        try:
            response, stats = await caller.call(
                call_once, operation=f"llm.{task.value}", idempotent=idempotent
            )
        except LLMProviderError:
            # Provider 异常直接上抛：由 service 层翻译成业务异常
            raise
        except Exception as exc:
            # ResilientCaller 的包装异常（熔断/预算耗尽）保留原语义
            log.error(
                "llm_call_failed",
                task=task.value,
                provider=resolved_provider,
                model=target_model,
                error_type=type(exc).__name__,
            )
            raise

        cost = self._compute_call_cost(response, at=datetime.now(UTC))
        if stats.retries:
            log.info(
                "llm_call_needed_retry",
                task=task.value,
                model=response.model,
                attempts=stats.attempts,
                note="重试会产生多次计费，成本需按实际调用累加",
            )
        return response, cost

    def _compute_call_cost(self, response: LLMResponse, *, at: datetime) -> CallCost:
        """按**实际模型 + 调用时刻**算成本。

        用 `response.model`（实际模型）而不是请求的模型名：
        DeepSeek 会把已下线的模型名路由到新模型并返回新名字。
        """
        pricing = self._settings.pricing_for(response.model)
        if pricing is None:
            # 未知模型：不猜价格。成本记 0 但保留 token 用量，
            # 等补上定价后可以重算历史（A3-5 的设计）。
            log.warning(
                "unknown_model_pricing",
                model=response.model,
                note="未配置单价，成本记 0；补上定价后可重算历史",
            )
            return CallCost(
                model=response.model,
                pricing_tier="unknown",
                input_tokens=response.usage.input_tokens,
                cached_tokens=response.usage.cached_tokens,
                output_tokens=response.usage.output_tokens,
                unit_price_input=Decimal("0"),
                unit_price_cached=None,
                unit_price_output=Decimal("0"),
                total_usd=Decimal("0"),
            )

        tier = "peak" if pricing.is_peak(at) else "off_peak"
        token_pricing = pricing.price_at(at)
        total, price_in, price_out = compute_cost(response.usage, token_pricing)

        return CallCost(
            model=response.model,
            pricing_tier=tier,
            input_tokens=response.usage.input_tokens,
            cached_tokens=response.usage.cached_tokens,
            output_tokens=response.usage.output_tokens,
            unit_price_input=price_in,
            unit_price_cached=token_pricing.input_cached,
            unit_price_output=price_out,
            total_usd=total,
        )

    async def aclose(self) -> None:
        """关闭所有供应商的 HTTP 客户端。"""
        for provider in self._providers.values():
            closer = getattr(provider, "aclose", None)
            if callable(closer):
                await closer()
