"""审计日志与模型调用明细测试。

两批重点：

1. **哈希链的防篡改能力**：这是审计日志存在的主要理由。
   测试要真的改数据，然后验证 verify_chain 能发现。
2. **成本计算的精度与口径**：单价快照、缓存命中折扣、
   Decimal 精度、失败调用也记录。

如果测试只覆盖"能插入能查出"，那么哈希链和成本公式里的 bug
会一路带到生产 —— 那时账目对不上，而且没有任何记录可回溯。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums.audit import (
    AuditAction,
    AuditOutcome,
    AuditTargetType,
    ModelCallStatus,
    ModelTaskType,
)
from rag_db.repositories import AuditRepository, ModelCallRepository

pytestmark = pytest.mark.integration


async def _make_tenant(session: AsyncSession) -> uuid.UUID:
    tenant_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
            "VALUES (:id, :s, 'x', 'active', now(), now())"
        ),
        {"id": tenant_id, "s": f"t-{tenant_id.hex[:10]}"},
    )
    return tenant_id


async def _make_user(session: AsyncSession) -> uuid.UUID:
    user_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO users (id, email, display_name, status, platform_role, "
            "tokens_valid_after, created_at, updated_at) "
            "VALUES (:id, :e, 'u', 'active', 'none', now(), now(), now())"
        ),
        {"id": user_id, "e": f"u-{user_id.hex[:8]}@example.test"},
    )
    return user_id


# ==================================================== 审计日志基础


async def test_record_audit_entry(db_session: AsyncSession) -> None:
    """审计记录能写入，链从 seq=1 开始。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    repo = AuditRepository(db_session)

    entry = await repo.record(
        tenant_id=tenant_id,
        action=AuditAction.DOC_UPLOAD,
        actor_user_id=user_id,
        actor_email="u@example.test",
        target_type=AuditTargetType.DOCUMENT,
        target_id=uuid.uuid4(),
        target_name="report.pdf",
        details={"size_bytes": 1024},
        request_id="req-1",
    )

    assert entry.seq == 1
    assert entry.prev_hash == "0" * 64
    assert len(entry.record_hash) == 64
    assert entry.outcome == AuditOutcome.SUCCESS.value


async def test_chain_uses_previous_hash(db_session: AsyncSession) -> None:
    """第二条的 prev_hash 必须等于第一条的 record_hash。"""
    tenant_id = await _make_tenant(db_session)
    repo = AuditRepository(db_session)

    first = await repo.record(tenant_id=tenant_id, action=AuditAction.AUTH_LOGIN)
    second = await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_READ)

    assert second.seq == 2
    assert second.prev_hash == first.record_hash


async def test_verify_chain_passes_for_valid_chain(db_session: AsyncSession) -> None:
    """正常写入的链必须通过验证。"""
    tenant_id = await _make_tenant(db_session)
    repo = AuditRepository(db_session)

    for _ in range(5):
        await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_READ)

    ok, bad_seq = await repo.verify_chain(tenant_id=tenant_id)
    assert ok is True
    assert bad_seq is None


async def test_verify_chain_detects_tampered_details(db_session: AsyncSession) -> None:
    """**核心测试：篡改 details 必须被检测出来。**

    这是哈希链的唯一价值所在。如果这条测试靠不住，
    审计日志就只是一张普通表，"不可篡改"是空话。
    """
    tenant_id = await _make_tenant(db_session)
    repo = AuditRepository(db_session)

    first = await repo.record(
        tenant_id=tenant_id,
        action=AuditAction.DOC_READ,
        details={"doc_ids": ["a", "b"]},
    )
    await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_READ)
    await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_READ)

    # 直接改数据库，绕过 ORM（模拟有 DB 权限的攻击者）
    await db_session.execute(
        text("UPDATE audit_logs SET details = :d WHERE id = :i"),
        {"d": '{"doc_ids": ["a"]}', "i": first.id},
    )
    db_session.expire_all()

    ok, bad_seq = await repo.verify_chain(tenant_id=tenant_id)
    assert ok is False, "篡改 details 未被检测到"
    assert bad_seq == 1


async def test_verify_chain_detects_tampered_action(db_session: AsyncSession) -> None:
    """篡改 action 也会被发现（action 参与哈希）。"""
    tenant_id = await _make_tenant(db_session)
    repo = AuditRepository(db_session)

    entry = await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_READ)
    # 把"读"改成"删除"，掩盖一次敏感操作
    await db_session.execute(
        text("UPDATE audit_logs SET action = 'document.delete' WHERE id = :i"),
        {"i": entry.id},
    )
    db_session.expire_all()

    ok, bad_seq = await repo.verify_chain(tenant_id=tenant_id)
    assert ok is False
    assert bad_seq == 1


async def test_verify_chain_detects_deleted_middle_entry(
    db_session: AsyncSession,
) -> None:
    """删除中间一条记录会导致 prev_hash 断链，被发现。"""
    tenant_id = await _make_tenant(db_session)
    repo = AuditRepository(db_session)

    first = await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_READ)
    middle = await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_READ)
    await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_READ)

    await db_session.execute(text("DELETE FROM audit_logs WHERE id = :i"), {"i": middle.id})
    db_session.expire_all()

    ok, bad_seq = await repo.verify_chain(tenant_id=tenant_id)
    # 删除中间条目后，第三条的 prev_hash 指向被删的那条，链断在第 3 条
    assert ok is False
    assert bad_seq == 3
    assert first.seq == 1


async def test_chain_is_per_tenant(db_session: AsyncSession) -> None:
    """每个租户独立的链，互不干扰。"""
    tenant_a = await _make_tenant(db_session)
    tenant_b = await _make_tenant(db_session)
    repo = AuditRepository(db_session)

    a1 = await repo.record(tenant_id=tenant_a, action=AuditAction.DOC_READ)
    b1 = await repo.record(tenant_id=tenant_b, action=AuditAction.DOC_READ)

    # 两个租户的链都从 1 开始，互不影响
    assert a1.seq == 1
    assert b1.seq == 1
    assert a1.prev_hash == b1.prev_hash == "0" * 64

    assert await repo.verify_chain(tenant_id=tenant_a) == (True, None)
    assert await repo.verify_chain(tenant_id=tenant_b) == (True, None)


async def test_global_audit_without_tenant(db_session: AsyncSession) -> None:
    """无租户审计（登录失败）不参与链，seq=0。"""
    repo = AuditRepository(db_session)

    entry = await repo.record(
        tenant_id=None,
        action=AuditAction.AUTH_LOGIN_FAILED,
        outcome=AuditOutcome.DENIED,
        actor_email="attacker@example.test",
        details={"reason": "invalid_password"},
    )

    assert entry.tenant_id is None
    assert entry.seq == 0
    assert entry.outcome == AuditOutcome.DENIED.value


async def test_security_denial_is_recorded(db_session: AsyncSession) -> None:
    """越权拒绝能被记录并查询出来（安全告警的数据源）。"""
    tenant_id = await _make_tenant(db_session)
    repo = AuditRepository(db_session)

    await repo.record(
        tenant_id=tenant_id,
        action=AuditAction.SECURITY_ACCESS_DENIED,
        outcome=AuditOutcome.DENIED,
        error_code="AUTHZ.FORBIDDEN",
        details={"attempted_doc_id": str(uuid.uuid4())},
    )
    await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_READ)

    denied = await repo.list_recent(tenant_id=tenant_id, only_denied=True)
    assert len(denied) == 1
    assert denied[0].action == AuditAction.SECURITY_ACCESS_DENIED.value


async def test_count_by_action(db_session: AsyncSession) -> None:
    """按动作统计（看板用）。"""
    tenant_id = await _make_tenant(db_session)
    repo = AuditRepository(db_session)

    for _ in range(3):
        await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_READ)
    await repo.record(tenant_id=tenant_id, action=AuditAction.DOC_UPLOAD)

    counts = await repo.count_by_action(
        tenant_id=tenant_id, since=datetime.now(UTC) - timedelta(hours=1)
    )
    assert counts.get(AuditAction.DOC_READ.value) == 3
    assert counts.get(AuditAction.DOC_UPLOAD.value) == 1


# ==================================================== 模型调用成本


async def test_record_model_call_computes_cost(db_session: AsyncSession) -> None:
    """成本按 token × 单价正确计算。

    口径（USD / 1M tokens）：
      cost = input/1e6*price_in + output/1e6*price_out
    """
    tenant_id = await _make_tenant(db_session)
    repo = ModelCallRepository(db_session)

    record = await repo.record(
        tenant_id=tenant_id,
        task_type=ModelTaskType.CHAT,
        provider="deepseek",
        model="deepseek-chat",
        input_tokens=1_000_000,
        output_tokens=500_000,
        unit_price_input=Decimal("0.27"),
        unit_price_output=Decimal("1.10"),
    )

    # 1M 输入 * 0.27 + 0.5M 输出 * 1.10 = 0.27 + 0.55 = 0.82
    assert record.cost_usd == Decimal("0.82000000")


async def test_cached_tokens_use_discount_price(db_session: AsyncSession) -> None:
    """缓存命中的 token 按折扣单价计算，不按常规输入价。

    prompt caching 的单价通常是常规价的 1/10。不区分会让成本估算偏高
    —— 在缓存命中率高的场景（长篇系统提示、固定文档）偏差可达 30%+。
    """
    tenant_id = await _make_tenant(db_session)
    repo = ModelCallRepository(db_session)

    record = await repo.record(
        tenant_id=tenant_id,
        task_type=ModelTaskType.CHAT,
        provider="openai",
        model="gpt-4o",
        input_tokens=1_000_000,
        cached_tokens=800_000,
        output_tokens=0,
        unit_price_input=Decimal("2.50"),
        unit_price_cached=Decimal("1.25"),  # 缓存价是常规价的一半
        unit_price_output=Decimal("10.00"),
    )

    # 未缓存 200k * 2.50/1M = 0.5
    # 缓存 800k * 1.25/1M = 1.0
    # 合计 1.5
    assert record.cost_usd == Decimal("1.50000000")


async def test_cost_is_zero_when_price_missing(db_session: AsyncSession) -> None:
    """单价未知时成本记 0，但 token 用量仍然记录。

    比"算不出就不记"好：至少保留用量，事后补上单价还能重算。
    完全丢弃意味着这部分调用从成本账里消失，账永远平不了。
    """
    tenant_id = await _make_tenant(db_session)
    repo = ModelCallRepository(db_session)

    record = await repo.record(
        tenant_id=tenant_id,
        task_type=ModelTaskType.EMBEDDING,
        provider="local",
        model="bge-m3",
        input_tokens=5000,
    )
    assert record.cost_usd == Decimal("0")
    assert record.input_tokens == 5000


async def test_cache_hit_is_free_but_recorded(db_session: AsyncSession) -> None:
    """命中我方缓存时成本为 0，但仍记录一条（统计命中率）。

    不记录的话无法回答"缓存到底省了多少"。
    """
    tenant_id = await _make_tenant(db_session)
    repo = ModelCallRepository(db_session)

    record = await repo.record(
        tenant_id=tenant_id,
        task_type=ModelTaskType.CHAT,
        provider="deepseek",
        model="deepseek-chat",
        input_tokens=1000,
        output_tokens=500,
        unit_price_input=Decimal("0.27"),
        unit_price_output=Decimal("1.10"),
        cache_hit=True,
    )
    assert record.cost_usd == Decimal("0")
    assert record.input_tokens == 1000


async def test_failed_call_is_recorded(db_session: AsyncSession) -> None:
    """失败的调用也要记录 —— 有些供应商失败也计费。"""
    tenant_id = await _make_tenant(db_session)
    repo = ModelCallRepository(db_session)

    record = await repo.record(
        tenant_id=tenant_id,
        task_type=ModelTaskType.CHAT,
        provider="openai",
        model="gpt-4o",
        status=ModelCallStatus.CONTENT_FILTERED.value,
        error_code="PROVIDER.CONTENT_FILTERED",
        input_tokens=800,
    )
    assert record.status == ModelCallStatus.CONTENT_FILTERED.value
    assert record.input_tokens == 800


async def test_sum_cost_is_exact(db_session: AsyncSession) -> None:
    """成本求和必须精确（Decimal 而非 float）。

    用 float 累加 10 万条记录会有可见误差，账目对不上。
    这里用多个会产生无限小数的单价验证精度。
    """
    tenant_id = await _make_tenant(db_session)
    repo = ModelCallRepository(db_session)

    # 0.1 + 0.2 在 float 下 != 0.3
    for _ in range(3):
        await repo.record(
            tenant_id=tenant_id,
            task_type=ModelTaskType.CHAT,
            provider="p",
            model="m",
            input_tokens=1_000_000,
            unit_price_input=Decimal("0.1"),
        )
    for _ in range(3):
        await repo.record(
            tenant_id=tenant_id,
            task_type=ModelTaskType.CHAT,
            provider="p",
            model="m",
            input_tokens=1_000_000,
            unit_price_input=Decimal("0.2"),
        )

    total = await repo.sum_cost(tenant_id=tenant_id, since=datetime.now(UTC) - timedelta(hours=1))
    assert total == Decimal("0.90000000"), f"成本求和精度错误: {total}"


async def test_breakdown_by_task_and_model(db_session: AsyncSession) -> None:
    """三维拆解：任务类型 × 供应商 × 模型。

    这是"按租户-功能-模型三维成本拆解"的实现验证。
    """
    tenant_id = await _make_tenant(db_session)
    repo = ModelCallRepository(db_session)

    # chat 用贵模型
    await repo.record(
        tenant_id=tenant_id,
        task_type=ModelTaskType.CHAT,
        provider="openai",
        model="gpt-4o",
        input_tokens=1_000_000,
        unit_price_input=Decimal("2.50"),
    )
    # rewrite 用便宜模型
    await repo.record(
        tenant_id=tenant_id,
        task_type=ModelTaskType.QUERY_REWRITE,
        provider="deepseek",
        model="deepseek-chat",
        input_tokens=1_000_000,
        unit_price_input=Decimal("0.27"),
    )

    rows = await repo.breakdown(tenant_id=tenant_id, since=datetime.now(UTC) - timedelta(hours=1))
    assert len(rows) == 2
    # 按成本降序，贵模型在前
    assert rows[0]["model"] == "gpt-4o"
    assert rows[0]["cost_usd"] == Decimal("2.50000000")
    assert rows[1]["task_type"] == ModelTaskType.QUERY_REWRITE.value


async def test_cache_hit_rate(db_session: AsyncSession) -> None:
    """缓存命中率统计。"""
    tenant_id = await _make_tenant(db_session)
    repo = ModelCallRepository(db_session)

    for i in range(10):
        await repo.record(
            tenant_id=tenant_id,
            task_type=ModelTaskType.CHAT,
            provider="p",
            model="m",
            cache_hit=i < 3,  # 前 3 条命中
        )

    rate = await repo.cache_hit_rate(
        tenant_id=tenant_id, since=datetime.now(UTC) - timedelta(hours=1)
    )
    assert rate == pytest.approx(0.3)


async def test_cost_isolated_between_tenants(db_session: AsyncSession) -> None:
    """成本统计按租户隔离 —— A 租户的账单不能含 B 租户的调用。"""
    tenant_a = await _make_tenant(db_session)
    tenant_b = await _make_tenant(db_session)
    repo = ModelCallRepository(db_session)

    await repo.record(
        tenant_id=tenant_a,
        task_type=ModelTaskType.CHAT,
        provider="p",
        model="m",
        input_tokens=1_000_000,
        unit_price_input=Decimal("1.0"),
    )
    await repo.record(
        tenant_id=tenant_b,
        task_type=ModelTaskType.CHAT,
        provider="p",
        model="m",
        input_tokens=1_000_000,
        unit_price_input=Decimal("5.0"),
    )

    since = datetime.now(UTC) - timedelta(hours=1)
    assert await repo.sum_cost(tenant_id=tenant_a, since=since) == Decimal("1.00000000")
    assert await repo.sum_cost(tenant_id=tenant_b, since=since) == Decimal("5.00000000")


async def test_fallback_calls_are_distinguishable(db_session: AsyncSession) -> None:
    """降级调用可识别（主模型挂了走备用时不污染主模型成本账）。"""
    tenant_id = await _make_tenant(db_session)
    repo = ModelCallRepository(db_session)

    record = await repo.record(
        tenant_id=tenant_id,
        task_type=ModelTaskType.CHAT,
        provider="deepseek",
        model="deepseek-chat",
        is_fallback=True,
        retry_count=2,
        latency_ms=850,
    )
    assert record.is_fallback is True
    assert record.retry_count == 2
    assert record.latency_ms == 850
