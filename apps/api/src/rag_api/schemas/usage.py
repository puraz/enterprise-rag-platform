"""用量契约的响应实体与映射。

契约形状来自 `rag_contracts.api.usage`；本模块只做 re-export 与
SQL 聚合行 → 响应实体的映射。

## 为什么接收"裸 SQL 行"而不是 ORM 对象

用量是**聚合结果**，没有对应的实体（一行 = 一个分组，不是一条记录）。
所以这里的入参是 `Sequence[Row]`，字段顺序由调用方的 select 决定。

这意味着调用方与映射器之间有隐含契约（第几列是什么）。为了不让它
变成"改 select 顺序就静默算错"，映射器要求入参是 **带命名字段的行**
（SQLAlchemy 的 `.all()` 返回 Row，支持按索引但不支持按名字访问
匿名列）。所以这里用位置索引，但**顺序固定在 docstring 里**，
且所有调用方都用同一份 select 模板（见 `_USAGE_COLUMNS` 说明）。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from rag_contracts.api.usage import DailyCostPoint, UsageBucket, UsageSummary
from rag_contracts.enums import ModelTaskType


def to_usage_summary(
    rows: Sequence[Any],
    *,
    since: datetime,
    until: datetime | None,
    cache_hit_rate: float,
    task_type_factory: type[ModelTaskType] = ModelTaskType,
) -> UsageSummary:
    """聚合行 → 用量汇总。

    每行必须是：

        (task_type, call_count, input_tokens, output_tokens, cached_tokens, cost_usd)

    与 `GET /me/usage` 的 select 保持一致。
    """
    buckets: list[UsageBucket] = []
    total_calls = 0
    total_input = 0
    total_output = 0
    total_cost = 0.0

    for row in rows:
        task_type = task_type_factory(str(row[0]))
        usage = UsageBucket(
            task_type=task_type,
            call_count=int(row[1]),
            input_tokens=int(row[2]),
            output_tokens=int(row[3]),
            cached_tokens=int(row[4]),
            cost_usd=float(row[5]),
        )
        buckets.append(usage)
        total_calls += usage.call_count
        total_input += usage.input_tokens
        total_output += usage.output_tokens
        total_cost += usage.cost_usd

    return UsageSummary(
        since=since,
        until=until,
        total_cost_usd=round(total_cost, 8),
        total_calls=total_calls,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        cache_hit_rate=cache_hit_rate,
        buckets=buckets,
    )


def to_daily_points(rows: Sequence[Any]) -> list[DailyCostPoint]:
    """`ModelCallRepository.daily_costs()` 的行 → 趋势点。

    每行必须是 `(date, call_count, cost_usd, input_tokens, output_tokens)`。
    """
    return [
        DailyCostPoint(
            date=str(row[0]),
            call_count=int(row[1]),
            cost_usd=float(row[2]),
            input_tokens=int(row[3]),
            output_tokens=int(row[4]),
        )
        for row in rows
    ]


__all__ = ["to_daily_points", "to_usage_summary"]
