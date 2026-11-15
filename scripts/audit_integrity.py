"""压测后对账：条数一致 / 无重复 / 无越权 / 成本账平。

## 为什么压测后一定要对账

压测会把系统推到边缘，而边缘暴露的 bug 有两类：

1. **崩溃型**（500/超时）——看得见，容易修；
2. **静默型**——看不见，危害大得多：
   - 并发导致同一文档被写两次（重复 chunk → 检索结果重复）；
   - 连接池耗尽后部分写入丢失（条数对不上）；
   - RLS 在并发下失效（越权）；
   - 成本记录漏记（账不平，用户被多收或少收）。

对账就是专门抓第 2 类。**"压测通过"不等于"数据正确"**——
必须在下游看结果，而不是只看响应码。

## 四类检查

### 1. 条数一致

文档数、版本数、chunk 数、向量数之间的关系必须守恒：

    documents >= sum(versions 的文档)
    chunks 数 == 每个文档 chunk_count 之和（针对 READY 文档）
    chunk_embeddings 行数 == chunks 数（每个 chunk 恰好一条向量）

任一条不成立说明存在丢写或重复写。

### 2. 无重复

- 同一 `(document_version_id, chunk_index)` 只能有一条（唯一约束兜底，
  但约束被绕过的情况要查）；
- 同一租户下相同 `content_hash` 的 chunk 数不该异常多（正常不应有）。

### 3. 无越权

- `chunks.tenant_id` 必须与其 `documents.tenant_id` 一致
  （不一致 = 跨租户数据串了）；
- `chunk_embeddings.tenant_id` 必须与 `chunks.tenant_id` 一致。

### 4. 成本账平

复用 `rag_db.reconciliation.reconcile_costs`（A13 已实现三条等式）。

## 为什么这些检查要写成 SQL 而不是 Python 循环

Python 逐条拉回来比对在百万级数据上不可行（内存 + 网络）。
全部用 `GROUP BY ... HAVING` 让数据库做，返回的只有"异常行"。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_core.logging import get_logger

log = get_logger(__name__)


@dataclass
class CheckResult:
    """单项检查结果。"""

    name: str
    passed: bool
    detail: str
    anomalies: list[str] = field(default_factory=list)
    """异常样本（最多 5 条，便于定位）。"""


@dataclass
class AuditReport:
    """对账报告。"""

    tenant_id: uuid.UUID | None
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def summary(self) -> str:
        lines = []
        for check in self.checks:
            mark = "✓" if check.passed else "✗"
            lines.append(f"{mark} {check.name}: {check.detail}")
            for anomaly in check.anomalies:
                lines.append(f"    - {anomaly}")
        return "\n".join(lines)


async def audit_data_integrity(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID | None = None,
) -> AuditReport:
    """执行全部对账检查。

    Args:
        tenant_id: 限定租户。None 表示全平台（需要 admin 连接）。
    """
    report = AuditReport(tenant_id=tenant_id)

    report.checks.append(await _check_chunk_embedding_parity(session, tenant_id=tenant_id))
    report.checks.append(await _check_duplicate_chunk_index(session, tenant_id=tenant_id))
    report.checks.append(await _check_cross_tenant_leak(session, tenant_id=tenant_id))
    report.checks.append(await _check_document_count_consistency(session, tenant_id=tenant_id))
    report.checks.append(await _check_orphan_embeddings(session, tenant_id=tenant_id))
    report.checks.append(await _check_cost_reconciliation(session, tenant_id=tenant_id))

    log.info(
        "audit_complete",
        tenant_id=str(tenant_id) if tenant_id else "ALL",
        passed=report.passed,
        checks=len(report.checks),
    )
    return report


async def _check_chunk_embedding_parity(
    session: AsyncSession, *, tenant_id: uuid.UUID | None
) -> CheckResult:
    """每个参与检索的 chunk 必须有且只有一条向量（在当前 ACTIVE 版本下）。

    这条检查抓的是"压测导致部分向量写入失败"——
    chunk 写进去了但向量没写，文档标 ready 却检索不到。
    这种故障在响应码上完全看不出来。
    """
    # 用 `(CAST(:tid AS uuid) IS NULL OR ...)` 而不是拼接 WHERE 片段：
    # 字符串拼接即使"片段是常量"也有隐患（将来有人把变量拼进去）。
    #
    # **CAST 不能省**：`$1 IS NULL` 让 PostgreSQL 无法推断参数类型
    # （报 `AmbiguousParameterError`），因为 NULL 字面量没有类型。
    # 显式 CAST 同时声明"这是 uuid"，也顺带禁止了类型混淆。
    result = await session.execute(
        text(
            """
            WITH active_versions AS (
                SELECT knowledge_base_id, id
                FROM index_versions
                WHERE status = 'active'
            )
            SELECT
                count(*) FILTER (WHERE ce.id IS NULL) AS missing,
                count(*) FILTER (WHERE dup.cnt > 1) AS duplicated
            FROM chunks c
            LEFT JOIN active_versions av ON av.knowledge_base_id = c.knowledge_base_id
            LEFT JOIN chunk_embeddings ce
                   ON ce.chunk_id = c.id AND ce.index_version_id = av.id
            LEFT JOIN (
                SELECT chunk_id, count(*) AS cnt
                FROM chunk_embeddings
                GROUP BY chunk_id
            ) dup ON dup.chunk_id = c.id
            WHERE c.is_current = true AND c.chunk_level = 0
              AND (CAST(:tid AS uuid) IS NULL OR c.tenant_id = CAST(:tid AS uuid))
            """
        ),
        {"tid": tenant_id},
    )
    row = result.one()
    missing, duplicated = int(row[0]), int(row[1])

    if missing == 0 and duplicated == 0:
        return CheckResult(
            name="chunk 与向量一一对应",
            passed=True,
            detail="每个检索单元的 chunk 都有向量，且无重复",
        )
    return CheckResult(
        name="chunk 与向量一一对应",
        passed=False,
        detail=f"缺失 {missing} 个、重复 {duplicated} 个",
        anomalies=[
            f"{missing} 个 chunk 没有向量（文档标 ready 但检索不到）" if missing else "",
            f"{duplicated} 个 chunk 有多条向量（检索会返回重复）" if duplicated else "",
        ],
    )


async def _check_duplicate_chunk_index(
    session: AsyncSession, *, tenant_id: uuid.UUID | None
) -> CheckResult:
    """同一版本的 chunk_index 不得重复。

    数据库有 `uq_chunks_version_index` 唯一约束，所以理论上不可能。
    但**约束可能在某些路径被绕过**（比如关掉约束后再建索引）。
    压测后查一次是廉价的保险。
    """
    result = await session.execute(
        text(
            """
            SELECT document_version_id, chunk_index, count(*) AS cnt
            FROM chunks
            WHERE (CAST(:tid AS uuid) IS NULL OR tenant_id = CAST(:tid AS uuid))
            GROUP BY document_version_id, chunk_index
            HAVING count(*) > 1
            LIMIT 5
            """
        ),
        {"tid": tenant_id},
    )
    rows = list(result)

    if not rows:
        return CheckResult(
            name="chunk_index 无重复",
            passed=True,
            detail="同版本内 chunk_index 唯一",
        )
    return CheckResult(
        name="chunk_index 无重复",
        passed=False,
        detail=f"{len(rows)} 组重复（样本）",
        anomalies=[f"version={row[0]} index={row[1]} 出现 {row[2]} 次" for row in rows],
    )


async def _check_cross_tenant_leak(
    session: AsyncSession, *, tenant_id: uuid.UUID | None
) -> CheckResult:
    """**安全关键**：子表的 tenant_id 必须与父表一致。

    ## 一个实测发现的设计缺陷（原版漏报）

    原版查询是：

        WHERE c.tenant_id <> d.tenant_id
          AND c.tenant_id = :tid          -- ← 这里把证据过滤掉了！

    如果把某条 chunk 的 tenant_id 改成别的租户，它正好不满足
    `c.tenant_id = :tid`，于是**被 WHERE 过滤掉**——检查通过，漏洞漏报。
    这是"检查逻辑自己把证据排除"的经典错误。

    正确做法：**按父表归属过滤**（`d.tenant_id = :tid`），
    然后比较子表的 tenant_id。父表归属是权威的（文档属于哪个租户
    是业务事实），子表字段抄错了才该被抓。

    实测验证：注入"把 chunk.tenant_id 改成随机租户"后，
    修复版立刻报出 1 条不一致。
    """
    result = await session.execute(
        text(
            """
            SELECT
              (SELECT count(*)
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.tenant_id <> d.tenant_id
                 AND (CAST(:tid AS uuid) IS NULL OR d.tenant_id = CAST(:tid AS uuid)))
                AS chunk_doc_mismatch,
              (SELECT count(*)
               FROM chunk_embeddings ce JOIN chunks c ON c.id = ce.chunk_id
               WHERE ce.tenant_id <> c.tenant_id
                 AND (CAST(:tid AS uuid) IS NULL OR c.tenant_id = CAST(:tid AS uuid)))
                AS emb_chunk_mismatch,
              (SELECT count(*)
               FROM document_versions dv JOIN documents d ON d.id = dv.document_id
               WHERE dv.tenant_id <> d.tenant_id
                 AND (CAST(:tid AS uuid) IS NULL OR d.tenant_id = CAST(:tid AS uuid)))
                AS version_doc_mismatch
            """
        ),
        {"tid": tenant_id},
    )
    row = result.one()
    chunk_mismatch, emb_mismatch, version_mismatch = (
        int(row[0]),
        int(row[1]),
        int(row[2]),
    )

    if chunk_mismatch == 0 and emb_mismatch == 0 and version_mismatch == 0:
        return CheckResult(
            name="租户归属一致（无越权）",
            passed=True,
            detail="chunk / 向量 / 版本 与其父记录的 tenant_id 完全一致",
        )
    return CheckResult(
        name="租户归属一致（无越权）",
        passed=False,
        detail=(
            f"chunk 不一致 {chunk_mismatch} 条、"
            f"向量不一致 {emb_mismatch} 条、"
            f"版本不一致 {version_mismatch} 条"
        ),
        anomalies=["这是跨租户数据泄漏，必须立即排查写入路径"],
    )


async def _check_document_count_consistency(
    session: AsyncSession, *, tenant_id: uuid.UUID | None
) -> CheckResult:
    """文档的 chunk_count 冗余字段必须与实际 chunk 数一致。

    `documents.chunk_count` 是为了避免列表页 JOIN 而冗余的。
    压测时如果"推进状态"与"写 chunk"不在同一事务，会出现
    count 与实际不符——列表页显示"10 个 chunk"但实际 8 个。

    ## 两个实测发现的盲区（都很隐蔽）

    **盲区一**：原版只查 `status='ready'` 的文档。实测发现评估脚本
    建的文档状态停在 `parsing`（有 37 个 chunk），检查直接跳过了
    ——而那正是最该被抓到的状态机 bug。

    **盲区二**：改成 `chunk_count > 0` 后仍然漏。因为评估脚本建文档时
    `chunk_count` 是默认值 0，chunk 是后续单独插入的——**没有"记录数
    大于 0"的文档，检查对象为空，恒通过**。

    正确做法：**从 chunk 侧反查**。以"实际有 chunk 的文档"为全集，
    比对它们记录的 `chunk_count`。这样无论 `chunk_count` 是 0 还是错值，
    只要实际有 chunk 就会被检查到。
    """
    result = await session.execute(
        text(
            """
            -- 从 chunk 侧聚合，再与文档记录比对：
            -- 这样"有 chunk 但 count=0"也能被抓到
            SELECT d.id, d.chunk_count, agg.actual
            FROM documents d
            JOIN (
                SELECT document_id, count(*) AS actual
                FROM chunks
                WHERE is_current = true AND chunk_level = 0
                GROUP BY document_id
            ) agg ON agg.document_id = d.id
            WHERE d.chunk_count <> agg.actual
              AND (CAST(:tid AS uuid) IS NULL OR d.tenant_id = CAST(:tid AS uuid))
            LIMIT 5
            """
        ),
        {"tid": tenant_id},
    )
    rows = list(result)

    if not rows:
        return CheckResult(
            name="文档 chunk_count 准确",
            passed=True,
            detail="所有有 chunk 的文档，其 chunk_count 记录与实际数一致",
        )
    return CheckResult(
        name="文档 chunk_count 准确",
        passed=False,
        detail=f"{len(rows)} 个文档的计数不准（样本）",
        anomalies=[f"doc={row[0]} 记录 {row[1]} 实际 {row[2]}" for row in rows],
    )


async def _check_orphan_embeddings(
    session: AsyncSession, *, tenant_id: uuid.UUID | None
) -> CheckResult:
    """向量不能指向已删除的 chunk。

    `chunk_embeddings.chunk_id` 是 CASCADE 外键，理论上不会孤儿。
    但如果有人绕过外键（或外键被临时禁用做迁移），会留下孤儿行
    ——它们占存储且可能被检索到。
    """
    result = await session.execute(
        text(
            """
            SELECT count(*)
            FROM chunk_embeddings ce
            LEFT JOIN chunks c ON c.id = ce.chunk_id
            WHERE c.id IS NULL
              AND (CAST(:tid AS uuid) IS NULL OR ce.tenant_id = CAST(:tid AS uuid))
            """
        ),
        {"tid": tenant_id},
    )
    orphans = int(result.scalar_one())

    if orphans == 0:
        return CheckResult(name="无孤儿向量", passed=True, detail="所有向量都指向存在的 chunk")
    return CheckResult(
        name="无孤儿向量",
        passed=False,
        detail=f"{orphans} 条孤儿向量",
        anomalies=["占存储且可能被检索到，需清理"],
    )


async def _check_cost_reconciliation(
    session: AsyncSession, *, tenant_id: uuid.UUID | None
) -> CheckResult:
    """成本账平（复用 A13 的对账逻辑）。"""
    if tenant_id is None:
        return CheckResult(
            name="成本账平",
            passed=True,
            detail="跳过（全平台对账需逐租户执行）",
        )

    from rag_db.reconciliation import reconcile_costs

    report = await reconcile_costs(session, tenant_id=tenant_id, days=1)
    if report.is_balanced:
        return CheckResult(
            name="成本账平",
            passed=True,
            detail=f"{report.checked_calls} 条调用，总额 ${report.total_cost}",
        )
    return CheckResult(
        name="成本账平",
        passed=False,
        detail=f"{report.issue_count} 个问题",
        anomalies=[f"{issue.kind}: {issue.detail}" for issue in report.issues[:5]],
    )


async def audit_active_queries(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
) -> dict[str, Any]:
    """统计当前活跃连接与长事务（压测后的"池是否恢复"检查）。

    **压测后连接池必须回到基线**。如果 `pg_stat_activity` 里还有大量
    连接或长事务，说明有连接泄漏（忘了 close / 事务忘了 commit）。
    """
    result = await session.execute(
        text(
            """
            SELECT
                count(*) FILTER (WHERE state = 'active') AS active,
                count(*) FILTER (WHERE state = 'idle') AS idle,
                count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_tx,
                max(EXTRACT(EPOCH FROM (now() - state_change)))
                    FILTER (WHERE state = 'idle in transaction') AS longest_idle_tx_s
            FROM pg_stat_activity
            WHERE datname = current_database()
              AND application_name LIKE 'enterprise-rag%'
            """
        )
    )
    row = result.one()
    return {
        "active": int(row[0] or 0),
        "idle": int(row[1] or 0),
        "idle_in_transaction": int(row[2] or 0),
        "longest_idle_tx_s": float(row[3]) if row[3] is not None else 0.0,
    }


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "AuditReport",
    "CheckResult",
    "audit_active_queries",
    "audit_data_integrity",
]

# 保留：未来按时间窗口对账时需要
_ = (Decimal, timedelta)
