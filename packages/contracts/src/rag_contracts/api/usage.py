"""用量与成本契约（个人中心 + 管理端看板共用）。

## 为什么个人用量与管理看板共用一套模型

`docs/PROMPT.md` 要求 Web 端有"个人中心：配额用量"，管理端有
"按租户-功能-模型三维成本拆解"。两者字段高度重合（都是
call_count / tokens / cost），差别只在**聚合维度**与**可见范围**。

共用一套 `UsageBucket` 的好处：前端图表组件只写一次，
且"个人看到的花费"与"管理端汇总的花费"口径天然一致——
分成两套模型时最容易出现"个人页面显示 3 元、管理端显示 4 元"。

## 金额为什么是 float 而不是 Decimal

契约层输出 `float`：JSON 没有 Decimal，Pydantic 序列化 Decimal 会变成
字符串（`"0.0012"`），前端图表要额外 parseFloat。
**精度损失在契约边界之前已发生**（DB 里是 `Numeric(16,8)`，聚合在 SQL 里做的），
展示用途下 float 的 15 位有效数字足够。

精确对账不走这个接口——`scripts/audit_integrity.py` 直接读 DB。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from rag_contracts.enums.audit import ModelTaskType


class UsageBucket(BaseModel):
    """一组同维度调用的用量与花费。

    `provider` / `model` 在个人视图里可为 None（用户不关心内部模型名，
    只看"这个功能花了我多少额度"）——聚合维度由查询参数决定。
    """

    model_config = ConfigDict(frozen=True)

    task_type: ModelTaskType
    provider: str | None = None
    model: str | None = None
    call_count: int = Field(ge=0, description="调用次数")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_tokens: int = Field(
        ge=0, description="其中命中供应商 prompt cache 的 token 数（按更低单价计费）"
    )
    cost_usd: float = Field(ge=0, description="该组花费（USD）")


class UsageSummary(BaseModel):
    """用量汇总。

    ## `total_cost_usd` 与 buckets 的关系

    它是 buckets 的求和，但仍然单独返回：前端首屏只要一个数字，
    不必等 buckets 渲染完再求和（也不必信任前端求和——浮点累加顺序
    不同会差最后一位，用户会截图质疑"加起来对不上"）。
    """

    model_config = ConfigDict(frozen=True)

    since: datetime = Field(description="统计窗口起点（含）")
    until: datetime | None = Field(
        default=None, description="统计窗口终点（不含），null 表示到现在"
    )
    total_cost_usd: float = Field(ge=0)
    total_calls: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    cache_hit_rate: float = Field(
        ge=0, le=1, description="缓存命中率 = 命中缓存的调用数 / 总调用数"
    )
    buckets: list[UsageBucket] = Field(default_factory=list)


class DailyCostPoint(BaseModel):
    """按天聚合的成本点（管理端趋势图）。"""

    model_config = ConfigDict(frozen=True)

    date: str = Field(description="日期（YYYY-MM-DD，按服务端时区归集）")
    call_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


__all__ = ["DailyCostPoint", "UsageBucket", "UsageSummary"]
