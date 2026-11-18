"""审计日志与模型调用明细仓储。

## 审计写入的两种事务模式（关键）

**模式 A：与业务同事务**（操作成功类）
    async with db.transaction() as session:
        await doc_repo.soft_delete(...)
        await audit.record(...)          # 同一个 session
    # 一起提交或一起回滚

**模式 B：独立事务**（安全事件类）
    async with db.transaction() as session:
        await doc_repo.soft_delete(...)
        raise PermissionDeniedError()    # 业务回滚
    # 另开事务：
    async with db.transaction() as session:
        await audit.record_security(...) # 这条必须留下来

**为什么必须分开**：如果把"越权被拒"放进业务事务，业务回滚时审计也跟着消失
—— 攻击者只要制造失败就能抹掉痕迹。反过来，把成功审计放独立事务，
业务回滚后会留下"用户删了文档"但文档还在的假记录，同样有害。

## 哈希链的原子推进

    UPDATE audit_chain_heads SET last_seq = last_seq + 1, last_hash = :h
     WHERE tenant_id = :t
    RETURNING last_seq

一条语句完成"读当前序号 + 加一 + 写新 hash"。用 SELECT 再 UPDATE 会有
竞态：两个请求读到同一个 seq，产生两条同 seq 记录（撞唯一约束）。

注意 `last_hash` 依赖新记录的 hash，而新记录的 hash 又依赖 prev_hash（=旧 last_hash）。
所以流程是：
  1. `SELECT last_seq, last_hash FROM audit_chain_heads WHERE tenant_id = :t FOR UPDATE`
     （锁住这一行）
  2. 计算新记录的 hash
  3. `UPDATE ... SET last_seq = last_seq + 1, last_hash = :new_hash`

或者更简洁：用 UPDATE 先占位（last_seq + 1，last_hash 暂填占位符），
拿到新 seq 后计算 hash 再 UPDATE 一次。两次 UPDATE 在同一事务里，
对外原子。

这里用**先锁后算**的方式：`SELECT ... FOR UPDATE` 锁住链头行，
这一行只有一个（per tenant），锁竞争范围极小。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from rag_contracts.enums.audit import AuditAction, AuditOutcome, AuditTargetType, ModelTaskType
from rag_db.models.audit import AuditChainHead, AuditLog
from rag_db.models.model_call import ModelCall

GENESIS_HASH = "0" * 64


def _canonical_json(payload: dict[str, Any]) -> str:
    """规范化 JSON 序列化，用于计算哈希。

    必须保证同样的内容总得到同样的字节：
    - `sort_keys=True`：字典顺序不影响结果；
    - `separators` 去掉多余空格；
    - `ensure_ascii=False`：中文按 UTF-8 编码，体积更小且确定。
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def compute_record_hash(
    *,
    prev_hash: str,
    seq: int,
    tenant_id: uuid.UUID | None,
    action: str,
    outcome: str,
    actor_user_id: uuid.UUID | None,
    target_type: str | None,
    target_id: uuid.UUID | None,
    details: dict[str, Any],
    created_at: datetime,
) -> str:
    """计算审计记录的哈希。

    参与哈希的字段必须**覆盖所有有审计价值的字段**。漏掉的字段可以被
    篡改而不被发现——例如如果 details 不参与，攻击者就能随意修改"访问了
    哪些文档"。

    `created_at` 用 ISO 字符串参与哈希：不同数据库/驱动返回的
    datetime 精度可能不同，转成字符串并固定精度才稳定。
    """
    payload = {
        "prev_hash": prev_hash,
        "seq": seq,
        "tenant_id": str(tenant_id) if tenant_id else None,
        "action": action,
        "outcome": outcome,
        "actor_user_id": str(actor_user_id) if actor_user_id else None,
        "target_type": target_type,
        "target_id": str(target_id) if target_id else None,
        "details": details,
        "created_at": created_at.isoformat(timespec="microseconds"),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class AuditRepository:
    """审计日志数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _ensure_chain_head(self, *, tenant_id: uuid.UUID) -> AuditChainHead:
        """确保链头存在（首次写入时创建）。

        用 ON CONFLICT DO NOTHING 而不是"先查再插"：并发首次写入时
        两个请求都会尝试插入，ON CONFLICT 让第二个静默跳过。
        """
        stmt = (
            pg_insert(AuditChainHead)
            .values(tenant_id=tenant_id, last_seq=0, last_hash=GENESIS_HASH)
            .on_conflict_do_nothing(index_elements=["tenant_id"])
        )
        await self._session.execute(stmt)

        result = await self._session.execute(
            select(AuditChainHead)
            .where(AuditChainHead.tenant_id == tenant_id)
            .with_for_update()  # 锁住链头，后续追加串行
            .limit(1)
        )
        return result.scalar_one()

    async def record(
        self,
        *,
        tenant_id: uuid.UUID | None,
        action: AuditAction,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        actor_user_id: uuid.UUID | None = None,
        actor_email: str | None = None,
        actor_tenant_role: str | None = None,
        actor_platform_role: str | None = None,
        target_type: AuditTargetType | None = None,
        target_id: uuid.UUID | None = None,
        target_name: str | None = None,
        request_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> AuditLog:
        """追加一条审计记录。

        `tenant_id=None` 时不进链（用于登录失败等无法确定租户的事件）：
        这类记录用 NULL tenant_id 存储，只有平台超管可见。
        它们不参与租户链，因为链的意义是"某个租户内的操作不可篡改"。

        **必须由调用方保证在正确的事务里**（见模块 docstring 的两种模式）。
        """
        details_payload = details or {}

        if tenant_id is None:
            # 无租户归属：不参与哈希链
            entry = AuditLog(
                tenant_id=None,
                seq=0,
                prev_hash=GENESIS_HASH,
                record_hash=GENESIS_HASH,
                action=action,
                outcome=outcome,
                actor_user_id=actor_user_id,
                actor_email=actor_email,
                actor_tenant_role=actor_tenant_role,
                actor_platform_role=actor_platform_role,
                target_type=target_type,
                target_id=target_id,
                target_name=target_name,
                request_id=request_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details_payload,
                error_code=error_code,
            )
            self._session.add(entry)
            await self._session.flush()
            return entry

        head = await self._ensure_chain_head(tenant_id=tenant_id)
        new_seq = head.last_seq + 1
        prev_hash = head.last_hash

        # 时间戳由数据库生成，但算哈希需要具体值。
        # 先取数据库时间，保证哈希与落库值一致。
        db_time_result = await self._session.execute(select(func.now()))
        created_at = db_time_result.scalar_one()

        record_hash = compute_record_hash(
            prev_hash=prev_hash,
            seq=new_seq,
            tenant_id=tenant_id,
            action=action.value,
            outcome=outcome.value,
            actor_user_id=actor_user_id,
            target_type=target_type.value if target_type else None,
            target_id=target_id,
            details=details_payload,
            created_at=created_at,
        )

        entry = AuditLog(
            tenant_id=tenant_id,
            seq=new_seq,
            prev_hash=prev_hash,
            record_hash=record_hash,
            created_at=created_at,
            action=action,
            outcome=outcome,
            actor_user_id=actor_user_id,
            actor_email=actor_email,
            actor_tenant_role=actor_tenant_role,
            actor_platform_role=actor_platform_role,
            target_type=target_type,
            target_id=target_id,
            target_name=target_name,
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details_payload,
            error_code=error_code,
        )
        self._session.add(entry)

        # 推进链头（同一事务内）
        head.last_seq = new_seq
        head.last_hash = record_hash
        flag_modified(head, "last_seq")
        flag_modified(head, "last_hash")

        await self._session.flush()
        return entry

    async def verify_chain(
        self, *, tenant_id: uuid.UUID, limit: int = 1000
    ) -> tuple[bool, int | None]:
        """验证审计链完整性。

        返回 `(是否完整, 第一个断点的 seq)`。

        **这是给运维/审计员用的**：定期跑一次，能发现"有人改了历史记录"。
        实际生产应该把链头 hash 定期外发到独立存储，那样即使整条链被
        重算也能发现。

        验证逻辑：从 seq=1 开始，每条记录的 prev_hash 必须等于前一条的
        record_hash，且重算 record_hash 必须与存储值一致。
        """
        result = await self._session.execute(
            select(AuditLog)
            .where(AuditLog.tenant_id == tenant_id)
            .order_by(AuditLog.seq)
            .limit(limit)
        )
        entries = list(result.scalars())

        expected_prev = GENESIS_HASH
        for entry in entries:
            if entry.prev_hash != expected_prev:
                return False, entry.seq

            recomputed = compute_record_hash(
                prev_hash=entry.prev_hash,
                seq=entry.seq,
                tenant_id=entry.tenant_id,
                action=str(entry.action),
                outcome=str(entry.outcome),
                actor_user_id=entry.actor_user_id,
                target_type=str(entry.target_type) if entry.target_type else None,
                target_id=entry.target_id,
                details=entry.details,
                created_at=entry.created_at,
            )
            if recomputed != entry.record_hash:
                return False, entry.seq
            expected_prev = entry.record_hash

        return True, None

    async def list_recent(
        self,
        *,
        tenant_id: uuid.UUID,
        limit: int = 50,
        action: AuditAction | None = None,
        target_type: AuditTargetType | None = None,
        target_id: uuid.UUID | None = None,
        only_denied: bool = False,
    ) -> Sequence[AuditLog]:
        """查询审计记录（管理端）。

        **不用 OFFSET**：审计日志量大，OFFSET 越翻越慢。
        这里用 `(created_at, id)` 的 keyset 由调用方传游标；
        简化起见当前只支持 limit + 过滤条件，A15 管理端接入时再补游标。
        """
        stmt = (
            select(AuditLog)
            .where(AuditLog.tenant_id == tenant_id)
            .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
            .limit(limit)
        )
        if action is not None:
            stmt = stmt.where(AuditLog.action == action)
        if target_type is not None:
            stmt = stmt.where(AuditLog.target_type == target_type)
        if target_id is not None:
            stmt = stmt.where(AuditLog.target_id == target_id)
        if only_denied:
            stmt = stmt.where(AuditLog.outcome != AuditOutcome.SUCCESS)

        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def count_by_action(self, *, tenant_id: uuid.UUID, since: datetime) -> dict[str, int]:
        """按动作统计（看板："最近越权尝试多少次"）。"""
        result = await self._session.execute(
            select(AuditLog.action, func.count())
            .where(AuditLog.tenant_id == tenant_id, AuditLog.created_at >= since)
            .group_by(AuditLog.action)
        )
        return {str(row[0]): int(row[1]) for row in result}


class ModelCallRepository:
    """模型调用明细数据访问。成本账的基础。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        tenant_id: uuid.UUID,
        task_type: ModelTaskType,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        unit_price_input: Decimal | None = None,
        unit_price_output: Decimal | None = None,
        unit_price_cached: Decimal | None = None,
        latency_ms: int | None = None,
        retry_count: int = 0,
        status: str = "success",
        error_code: str | None = None,
        error_message: str | None = None,
        user_id: uuid.UUID | None = None,
        conversation_id: uuid.UUID | None = None,
        message_id: uuid.UUID | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        is_fallback: bool = False,
        cache_hit: bool = False,
        call_metadata: dict[str, Any] | None = None,
    ) -> ModelCall:
        """记录一次模型调用。

        **成本计算公式**（单位：USD/1M tokens）：

            cost = (input_tokens - cached_tokens) / 1e6 * unit_price_input
                 + cached_tokens / 1e6 * unit_price_cached
                 + output_tokens / 1e6 * unit_price_output

        缓存部分单独算：prompt caching 的单价通常是常规输入价的 1/10，
        不区分会让成本估算偏高。

        **单价为 None 时成本记 0**，但记录仍然落库（token 数还在）。
        这比"算不出成本就不记"好：至少能看到用量，事后补上单价还能重算。

        注意用 Decimal 运算，不用 float：10 万条记录累加后 float 的
        误差足以让账目对不上（0.01 量级的钱）。
        """
        cost = Decimal("0")

        if unit_price_input is not None:
            uncached = max(0, input_tokens - cached_tokens)
            cost += Decimal(uncached) / Decimal(1_000_000) * unit_price_input
        if cached_tokens and unit_price_cached is not None:
            cost += Decimal(cached_tokens) / Decimal(1_000_000) * unit_price_cached
        if unit_price_output is not None:
            cost += Decimal(output_tokens) / Decimal(1_000_000) * unit_price_output

        # 缓存命中（我方缓存）：不计费，但记录用量以统计命中率
        if cache_hit:
            cost = Decimal("0")

        record = ModelCall(
            tenant_id=tenant_id,
            task_type=task_type,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            unit_price_input=unit_price_input,
            unit_price_output=unit_price_output,
            unit_price_cached=unit_price_cached,
            cost_usd=cost,
            latency_ms=latency_ms,
            retry_count=retry_count,
            status=status,
            error_code=error_code,
            error_message=error_message,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            request_id=request_id,
            trace_id=trace_id,
            is_fallback=is_fallback,
            cache_hit=cache_hit,
            call_metadata=call_metadata or {},
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def find_trace_id(self, *, tenant_id: uuid.UUID, message_id: uuid.UUID) -> str | None:
        """取某条助手消息对应 LLM 调用的 trace_id（反馈回写 score 用）。

        ## 为什么不给 `messages` 加一列 trace_id

        `model_calls` 已经是"每次模型调用"的权威明细，且与消息是多对一
        （改写/判断调用的 `message_id` 为空，只有真正的答案调用带消息 ID）。
        再加一列意味着两处都要写、两处都可能漏——而对账仍以 model_calls 为准。

        这里按 `created_at DESC` 取最新一条：将来若一条消息有多次重试，
        最新那次才是用户实际看到的那次。
        """
        result = await self._session.execute(
            select(ModelCall.trace_id)
            .where(
                ModelCall.tenant_id == tenant_id,
                ModelCall.message_id == message_id,
                ModelCall.trace_id.is_not(None),
            )
            .order_by(ModelCall.created_at.desc())
            .limit(1)
        )
        trace_id = result.scalar_one_or_none()
        return str(trace_id) if trace_id else None

    async def sum_cost(
        self,
        *,
        tenant_id: uuid.UUID,
        since: datetime,
        until: datetime | None = None,
        task_type: ModelTaskType | None = None,
    ) -> Decimal:
        """汇总时间范围内的成本。

        用 SQL 聚合而不是 Python 累加：数据量大时逐行拉回内存不可行，
        且 SQL 的 NUMERIC 求和是精确的。
        """
        stmt = select(func.coalesce(func.sum(ModelCall.cost_usd), 0)).where(
            ModelCall.tenant_id == tenant_id,
            ModelCall.created_at >= since,
        )
        if until is not None:
            stmt = stmt.where(ModelCall.created_at < until)
        if task_type is not None:
            stmt = stmt.where(ModelCall.task_type == task_type)

        result = await self._session.execute(stmt)
        value = result.scalar_one()
        return value if isinstance(value, Decimal) else Decimal(str(value))

    async def breakdown(
        self,
        *,
        tenant_id: uuid.UUID,
        since: datetime,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """按 (任务类型, 供应商, 模型) 三维拆解成本。

        这是"按租户-功能-模型三维成本拆解"的实现。
        返回每个组合的调用数、token 用量、成本。
        """
        stmt = (
            select(
                ModelCall.task_type,
                ModelCall.provider,
                ModelCall.model,
                func.count().label("call_count"),
                func.coalesce(func.sum(ModelCall.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(ModelCall.output_tokens), 0).label("output_tokens"),
                func.coalesce(func.sum(ModelCall.cached_tokens), 0).label("cached_tokens"),
                func.coalesce(func.sum(ModelCall.cost_usd), 0).label("cost_usd"),
            )
            .where(ModelCall.tenant_id == tenant_id, ModelCall.created_at >= since)
            .group_by(ModelCall.task_type, ModelCall.provider, ModelCall.model)
            .order_by(func.sum(ModelCall.cost_usd).desc())
        )
        if until is not None:
            stmt = stmt.where(ModelCall.created_at < until)

        result = await self._session.execute(stmt)
        return [
            {
                "task_type": str(row[0]),
                "provider": str(row[1]),
                "model": str(row[2]),
                "call_count": int(row[3]),
                "input_tokens": int(row[4]),
                "output_tokens": int(row[5]),
                "cached_tokens": int(row[6]),
                "cost_usd": row[7] if isinstance(row[7], Decimal) else Decimal(str(row[7])),
            }
            for row in result
        ]

    async def cache_hit_rate(self, *, tenant_id: uuid.UUID, since: datetime) -> float:
        """缓存命中率（成本优化的核心指标）。"""
        result = await self._session.execute(
            select(
                func.count().label("total"),
                func.count().filter(ModelCall.cache_hit.is_(True)).label("hits"),
            ).where(ModelCall.tenant_id == tenant_id, ModelCall.created_at >= since)
        )
        total, hits = result.one()
        return float(hits) / float(total) if total else 0.0
