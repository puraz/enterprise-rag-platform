"""LLM 供应商配置。

## 为什么单价要写进配置

`docs/PROMPT.md` 要求"成本账要平"，而成本 = token × 单价。
单价如果只写在前端或文档里，后端算账时就得硬编码——供应商调价后
两处必然不一致。

写进配置的好处：
- 调价只改环境变量，不用发版；
- `model_calls` 记录**当时的单价快照**，历史账单可重算验证；
- 多供应商各自配单价，成本拆解时不需要外部映射表。

## 峰谷计价（DeepSeek 特有）

DeepSeek 的定价按 UTC 时段分峰谷，**价差 2 倍**：
- 峰值：UTC 工作日 01:00–04:00、06:00–10:00；
- 谷值：其余时间（含周末与节假日）。

如果只配一套单价，成本会系统性地偏高一倍或偏低一半。
这里拆成 peak / off_peak 两组，由 `ModelPricing.price_at()` 按调用时间选。

## 缓存计价

DeepSeek 的 prompt cache 命中价是未命中的 **1/50**（flash: $0.006 vs $0.30）。
不区分缓存会让成本估算严重偏高——长系统提示场景下偏差可达几十倍。
"""

from __future__ import annotations

from datetime import UTC, datetime, time
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class TokenPricing(BaseModel):
    """某一档的单价的（USD / 1M tokens）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: Decimal = Field(description="输入 token 单价（未命中缓存）")
    input_cached: Decimal | None = Field(
        default=None,
        description="输入 token 命中 prompt cache 的单价。None 表示不区分缓存",
    )
    output: Decimal = Field(description="输出 token 单价")


class ModelPricing(BaseModel):
    """单个模型的定价（支持峰谷两档）。

    单价单位统一为 **USD / 1M tokens**，与主流供应商报价口径一致。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 谷时（默认档）
    off_peak: TokenPricing
    # 峰时。None 表示无峰谷差异
    peak: TokenPricing | None = None

    # DeepSeek 峰时段（UTC）：工作日 01:00-04:00 与 06:00-10:00
    peak_windows_utc: tuple[tuple[int, int], ...] = ((1, 4), (6, 10))

    def is_peak(self, at: datetime | None = None) -> bool:
        """判断给定时刻是否处于峰时。

        **必须按 UTC 判断**：供应商的定价时段是 UTC 定义的，
        用本地时区判断会因部署地不同而错位（在 UTC+8 部署时，
        本地 09:00-12:00 实际是 UTC 01:00-04:00 的峰时）。

        周末全天算谷时（官方文档：峰时仅工作日）。
        """
        moment = at or datetime.now(UTC)
        moment = moment.astimezone(UTC)
        # 周六(5)/周日(6) 全谷时
        if moment.weekday() >= 5:
            return False
        hour = moment.hour
        return any(start <= hour < end for start, end in self.peak_windows_utc)

    def price_at(self, at: datetime | None = None) -> TokenPricing:
        """取给定时刻适用的单价档。"""
        if self.peak is not None and self.is_peak(at):
            return self.peak
        return self.off_peak


class LLMSettings(BaseModel):
    """LLM 供应商配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    provider: str = "deepseek"
    """供应商名。对应 `rag_api.providers` 注册表的键。"""

    api_key: str = ""
    """API Key。**只从环境变量 `RAG_LLM__API_KEY` 读取，不进库、不进日志**。

    空字符串表示未配置——调用时会明确报错而不是静默失败。
    """

    base_url: str = "https://api.deepseek.com/v1"
    """OpenAI 兼容端点。换供应商/自建（vLLM、Ollama）只改这一行。"""

    chat_model: str = "deepseek-v4-pro"
    """生成答案用的强模型。"""

    cheap_model: str = "deepseek-flash"
    """改写/HyDE/压缩等辅助任务用的便宜模型。

    `docs/PROMPT.md` 要求"按任务分级路由（简单任务走便宜小模型）"。
    """

    reasoning_effort: str = Field(
        default="low",
        description=(
            "思考深度（DeepSeek 的 low/high/max）。\n\n"
            "**实测结论（2026-11）**：两个模型默认开启思考模式，"
            "且无法完全关闭。思考 token 按输出价计费，且会吃掉 max_tokens。\n"
            "- `max_tokens=50` 时思考耗尽预算，content 为空、finish_reason=length；\n"
            "- 设 `low` 后简单问题通常 20-30 个思考 token 就能拿到正文。\n"
            "RAG 的多数任务（改写/压缩/抽取）不需要深思考，因此默认 low。"
        ),
    )

    timeout_s: float = Field(default=120.0, gt=0)
    """单次调用超时。LLM 生成长答案可能很慢，比普通 HTTP 超时长。"""

    max_concurrency: int = Field(default=8, ge=1)
    """并发上限。供应商有速率限制，超过会大量 429。"""

    # ---- 定价（USD / 1M tokens）----
    # 数值来源：官方文档 2026-11 核实。调价时改环境变量即可。
    pricing: dict[str, ModelPricing] = Field(
        default_factory=lambda: {
            # DeepSeek-V4.1-Flash
            "deepseek-flash": ModelPricing(
                off_peak=TokenPricing(
                    input=Decimal("0.15"),
                    input_cached=Decimal("0.003"),
                    output=Decimal("0.6"),
                ),
                peak=TokenPricing(
                    input=Decimal("0.30"),
                    input_cached=Decimal("0.006"),
                    output=Decimal("1.2"),
                ),
            ),
            # DeepSeek-V4-Pro
            "deepseek-v4-pro": ModelPricing(
                off_peak=TokenPricing(
                    input=Decimal("0.66"),
                    input_cached=Decimal("0.022"),
                    output=Decimal("1.98"),
                ),
                peak=TokenPricing(
                    input=Decimal("1.32"),
                    input_cached=Decimal("0.044"),
                    output=Decimal("3.96"),
                ),
            ),
        },
        description="模型定价表。未知模型不估算成本（记 0 并告警）",
    )

    def pricing_for(self, model: str) -> ModelPricing | None:
        """取模型定价。未知模型返回 None（不猜）。"""
        return self.pricing.get(model)


def utc_peak_range_description() -> str:
    """峰时段的可读描述（用于启动日志与文档）。"""
    return "UTC 工作日 01:00-04:00, 06:00-10:00"


# 供测试固定时间用
def peak_time() -> datetime:
    """构造一个必定处于峰时的 UTC 时刻（周一 02:00）。"""
    return datetime(2026, 11, 16, 2, 0, tzinfo=UTC)  # 2026-11-16 是周一


def off_peak_time() -> datetime:
    """构造一个必定处于谷时的 UTC 时刻（周一 12:00）。"""
    return datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


__all__ = [
    "LLMSettings",
    "ModelPricing",
    "TokenPricing",
    "off_peak_time",
    "peak_time",
    "time",
    "utc_peak_range_description",
]
