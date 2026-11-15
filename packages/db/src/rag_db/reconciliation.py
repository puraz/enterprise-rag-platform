"""成本对账：验证成本记录内部一致，且与用量能互相验证。

## 为什么需要对账

"成本账平"不是"总数看起来合理"，而是三条可验证的等式：

### 等式 1：记录的成本 == 按单价快照重算的成本

    cost_usd == (input - cached)/1e6 * unit_price_input
              + cached/1e6 * unit_price_cached
              + output/1e6 * unit_price_output

不成立说明：写入路径算错了，或后来改了记录（表设计是只追加，
理论不该改——查出来就是数据被篡改或有绕过仓储的写入）。

### 等式 2：明细求和 == 聚合结果

`SUM(model_calls.cost_usd)` 与按任务类型/模型聚合的分项之和必须相等。
不成立说明聚合 SQL 有 bug（漏了某个维度、条件写错）。

### 等式 3：token 守恒

`input_tokens >= cached_tokens` —— 缓存命中的 token 是输入的一部分，
不可能超过。
不成立说明：用量字段赋值错了（比如把 cache_read 加到了 input 上，
导致重复计数）。

## 为什么这些等式比"总额看起来对"重要

只有总额时，两类错误会互相抵消：
- 输入单价算高了，输出单价算低了 → 总额看起来正常；
- 某次调用重复记录，另一次漏记 → 总额看起来正常。

等式逐条检查能定位到具体哪条记录、哪一项不成立。

## 精度：全程 Decimal

`Numeric(16,8)` 在库里是精确的，但一旦用 float 比较就会有误差。
允许的偏差是 `Decimal("0.00000001")`（即 1e-8，一个最小单位的量级），
用于吸收数据库往返时的舍入。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_core.logging import get_logger

log = get_logger(__name__)

# 允许的数值偏差：Numeric(16,8) 的最小单位。
# 超过这个说明是真差异，不是舍入。
TOLERANCE = Decimal("0.00000001")


@dataclass
class ReconciliationIssue:
    """一条对账问题。"""

    kind: str
    """问题类型：cost_mismatch / aggregate_mismatch / token_overflow"""

    detail: str
    call_id: str | None = None
    expected: str | None = None
    actual: str | None = None


@dataclass
class ReconciliationReport:
    """对账报告。"""

    tenant_id: uuid.UUID
    since: datetime
    until: datetime
    checked_calls: int = 0
    total_cost: Decimal = Decimal("0")
    issues: list[ReconciliationIssue] = field(default_factory=list)

    @property
    def is_balanced(self) -> bool:
        return not self.issues

    @property
    def issue_count(self) -> int:
        return len(self.issues)


async def reconcile_costs(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    days: int = 30,
    max_detail_rows: int = 5000,
) -> ReconciliationReport:
    """对账。

    Args:
        days: 对账窗口（最近 N 天）。
        max_detail_rows: 最多检查多少条明细。全量逐条核对在百万级数据上
            会很慢，这里设上限（**超限时报告会注明"仅抽样"**，
            不假装全量通过）。
    """
    until = datetime.now(UTC)
    since = until - timedelta(days=days)
    report = ReconciliationReport(tenant_id=tenant_id, since=since, until=until)

    # ---- 等式 2：明细求和 vs 聚合 ----
    total_row = await session.execute(
        text(
            "SELECT coalesce(sum(cost_usd), 0) AS total, count(*) AS cnt "
            "FROM model_calls WHERE tenant_id = :t AND created_at >= :s"
        ),
        {"t": tenant_id, "s": since},
    )
    total, count = total_row.one()
    report.total_cost = Decimal(str(total))
    report.checked_calls = int(count)

    breakdown_row = await session.execute(
        text(
            "SELECT coalesce(sum(sub.total), 0) FROM ("
            "  SELECT sum(cost_usd) AS total FROM model_calls"
            "  WHERE tenant_id = :t AND created_at >= :s"
            "  GROUP BY task_type, provider, model"
            ") sub"
        ),
        {"t": tenant_id, "s": since},
    )
    breakdown_total = Decimal(str(breakdown_row.scalar_one()))
    if abs(breakdown_total - report.total_cost) > TOLERANCE:
        report.issues.append(
            ReconciliationIssue(
                kind="aggregate_mismatch",
                detail="按三维聚合的总和与明细总和不等（聚合 SQL 可能漏了维度或条件写错）",
                expected=str(report.total_cost),
                actual=str(breakdown_total),
            )
        )

    # ---- 等式 1 与 3：逐条核对 ----
    rows = await session.execute(
        text(
            "SELECT id, input_tokens, output_tokens, cached_tokens,"
            "       unit_price_input, unit_price_cached, unit_price_output, cost_usd,"
            "       cache_hit, status "
            "FROM model_calls "
            "WHERE tenant_id = :t AND created_at >= :s "
            "ORDER BY created_at DESC LIMIT :lim"
        ),
        {"t": tenant_id, "s": since, "lim": max_detail_rows},
    )

    checked = 0
    for row in rows:
        checked += 1
        (
            call_id,
            input_tokens,
            output_tokens,
            cached_tokens,
            price_in,
            price_cached,
            price_out,
            cost_usd,
            cache_hit,
            status,
        ) = row

        # ---- 等式 3：token 守恒 ----
        if int(cached_tokens) > int(input_tokens):
            report.issues.append(
                ReconciliationIssue(
                    kind="token_overflow",
                    detail="cached_tokens 大于 input_tokens（缓存 token 是输入的一部分）",
                    call_id=str(call_id),
                    expected=f"<= {input_tokens}",
                    actual=str(cached_tokens),
                )
            )

        # ---- 等式 1：按单价快照重算 ----
        #
        # **缓存命中的调用跳过**：它们成本被强制置 0（我方缓存命中不计费），
        # 但用量字段仍保留（用于统计命中率），所以重算必然不等于 0。
        if bool(cache_hit):
            if Decimal(str(cost_usd)) > TOLERANCE:
                report.issues.append(
                    ReconciliationIssue(
                        kind="cost_mismatch",
                        detail="我方缓存命中的调用不该有成本（应为 0）",
                        call_id=str(call_id),
                        expected="0",
                        actual=str(cost_usd),
                    )
                )
            continue

        # 单价全为空：成本记 0 是允许的（"算不出成本但 Token 仍记录"的设计）
        if price_in is None and price_out is None:
            continue

        expected = Decimal("0")
        if price_in is not None:
            uncached = max(0, int(input_tokens) - int(cached_tokens))
            expected += Decimal(uncached) / Decimal(1_000_000) * Decimal(str(price_in))
        if int(cached_tokens) and price_cached is not None:
            expected += (
                Decimal(int(cached_tokens)) / Decimal(1_000_000) * Decimal(str(price_cached))
            )
        if price_out is not None:
            expected += Decimal(int(output_tokens)) / Decimal(1_000_000) * Decimal(str(price_out))

        actual = Decimal(str(cost_usd))
        if abs(expected - actual) > TOLERANCE:
            report.issues.append(
                ReconciliationIssue(
                    kind="cost_mismatch",
                    detail=(f"按单价快照重算的成本与记录不符（status={status}）"),
                    call_id=str(call_id),
                    expected=f"{expected:.8f}",
                    actual=f"{actual:.8f}",
                )
            )

    if report.checked_calls > checked:
        report.issues.append(
            ReconciliationIssue(
                kind="partial_check",
                detail=(
                    f"仅核对了最近 {checked} 条明细（共 {report.checked_calls} 条）。"
                    f"报告结论不代表全量通过。"
                ),
            )
        )

    log.info(
        "cost_reconciliation",
        tenant_id=str(tenant_id),
        checked_calls=report.checked_calls,
        total_cost=str(report.total_cost),
        issues=report.issue_count,
        balanced=report.is_balanced,
    )
    return report


__all__ = [
    "TOLERANCE",
    "ReconciliationIssue",
    "ReconciliationReport",
    "reconcile_costs",
]
