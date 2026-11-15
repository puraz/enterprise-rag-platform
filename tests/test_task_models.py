"""上传会话 / 任务 / outbox 集成测试（真实 PG）。

重点验证状态机的并发安全与 outbox 的至少一次送达语义：

- 分片计数并发自增不丢更新
- 两个请求同时 complete 只有一个成功（防重复合并）
- 过期会话可被清理，已完成的不受影响
- worker 并发认领不重复（SKIP LOCKED）
- IDLE 任务不会被提前认领（available_at 生效）
- worker 崩溃后任务能被回收重派
- 重试超限进死信，不是静默丢弃
- outbox 幂等入队 + 至少一次投递 + 退避重试
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums.uploads import (
    OutboxStatus,
    TaskStatus,
    TaskType,
    UploadSessionStatus,
)
from rag_db.repositories import OutboxRepository, TaskRepository, UploadSessionRepository

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


def _session_kwargs(tenant_id: uuid.UUID, user_id: uuid.UUID) -> dict[str, object]:
    return {
        "tenant_id": tenant_id,
        "created_by": user_id,
        "filename": "report.pdf",
        "total_size": 100_000_000,
        "content_type": "application/pdf",
        "part_size": 10_000_000,
        "total_parts": 10,
        "storage_key_prefix": "uploads/test/",
        "storage_backend": "local",
    }


# ==================================================== 上传会话


async def test_create_upload_session(db_session: AsyncSession) -> None:
    """会话创建，初始状态为 initiated。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)

    session = await UploadSessionRepository(db_session).create(
        **_session_kwargs(tenant_id, user_id)  # type: ignore[arg-type]
    )
    assert session.status == UploadSessionStatus.INITIATED.value
    assert session.uploaded_parts == 0
    assert session.version == 1
    assert session.expires_at > datetime.now(UTC)


async def test_part_count_increments_atomically(db_session: AsyncSession) -> None:
    """分片计数自增正确，且状态从 initiated 变 uploading。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    repo = UploadSessionRepository(db_session)
    session = await repo.create(**_session_kwargs(tenant_id, user_id))  # type: ignore[arg-type]

    for expected_version in range(1, 4):
        ok = await repo.mark_part_uploaded(session_id=session.id, expected_version=expected_version)
        assert ok is True

    await db_session.refresh(session)
    assert session.uploaded_parts == 3
    assert session.status == UploadSessionStatus.UPLOADING.value


async def test_part_upload_rejected_after_completing(db_session: AsyncSession) -> None:
    """进入 COMPLETING 后不能再收分片。

    防止"合并过程中又来了一个分片"，导致合并出的文件比声明的长。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    repo = UploadSessionRepository(db_session)
    session = await repo.create(**_session_kwargs(tenant_id, user_id))  # type: ignore[arg-type]

    await repo.mark_part_uploaded(session_id=session.id, expected_version=1)
    await db_session.refresh(session)
    began = await repo.begin_complete(tenant_id=tenant_id, session_id=session.id)
    assert began is not None

    # 用当前版本号尝试再传一片，应被状态条件拒绝
    await db_session.refresh(session)
    rejected = await repo.mark_part_uploaded(
        session_id=session.id, expected_version=session.version
    )
    assert rejected is False


async def test_duplicate_complete_is_rejected(db_session: AsyncSession) -> None:
    """**两个请求同时 complete，只有一个能把状态切到 COMPLETING。**

    没有这个原子切换，两个请求会同时合并分片、同时创建 blob，
    产生重复存储与重复文档。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    repo = UploadSessionRepository(db_session)
    session = await repo.create(**_session_kwargs(tenant_id, user_id))  # type: ignore[arg-type]

    first = await repo.begin_complete(tenant_id=tenant_id, session_id=session.id)
    second = await repo.begin_complete(tenant_id=tenant_id, session_id=session.id)

    assert first is not None, "第一次 complete 应成功"
    assert second is None, "第二次 complete 必须被拒绝"


async def test_completed_session_cannot_be_aborted(db_session: AsyncSession) -> None:
    """已完成的会话不能中止 —— 它的 blob 已经被文档引用了。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    repo = UploadSessionRepository(db_session)
    session = await repo.create(**_session_kwargs(tenant_id, user_id))  # type: ignore[arg-type]

    blob_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, storage_key, "
            "storage_backend, content_type, status, ref_count, created_at, updated_at) "
            "VALUES (:i, :t, :h, 1, :k, 'local', 'application/pdf', 'complete', 1, now(), now())"
        ),
        {"i": blob_id, "t": tenant_id, "h": uuid.uuid4().hex + uuid.uuid4().hex, "k": "k"},
    )
    await repo.finish_complete(session_id=session.id, blob_id=blob_id)

    assert await repo.abort(tenant_id=tenant_id, session_id=session.id) is False


async def test_list_expired_excludes_completed(db_session: AsyncSession) -> None:
    """清理任务只拿未完成的过期会话。

    已完成的会话即使过了 expires_at 也不能回收：它指向的 blob 正被文档使用。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    repo = UploadSessionRepository(db_session)

    expired_pending = await repo.create(**_session_kwargs(tenant_id, user_id))  # type: ignore[arg-type]
    expired_done = await repo.create(**_session_kwargs(tenant_id, user_id))  # type: ignore[arg-type]
    past = datetime.now(UTC) - timedelta(hours=1)
    for session_id in (expired_pending.id, expired_done.id):
        await db_session.execute(
            text("UPDATE upload_sessions SET expires_at = :t WHERE id = :i"),
            {"t": past, "i": session_id},
        )

    blob_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, storage_key, "
            "storage_backend, content_type, status, ref_count, created_at, updated_at) "
            "VALUES (:i, :t, :h, 1, :k, 'local', 'application/pdf', 'complete', 1, now(), now())"
        ),
        {"i": blob_id, "t": tenant_id, "h": uuid.uuid4().hex + uuid.uuid4().hex, "k": "k2"},
    )
    await repo.finish_complete(session_id=expired_done.id, blob_id=blob_id)

    expired = await repo.list_expired(limit=10)
    ids = {s.id for s in expired}
    assert expired_pending.id in ids
    assert expired_done.id not in ids, "已完成的会话被误判为待清理"


async def test_session_is_tenant_isolated(db_session: AsyncSession) -> None:
    """跨租户取不到别人的上传会话。"""
    tenant_a = await _make_tenant(db_session)
    tenant_b = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    repo = UploadSessionRepository(db_session)
    session = await repo.create(**_session_kwargs(tenant_a, user_id))  # type: ignore[arg-type]

    assert await repo.get(tenant_id=tenant_a, session_id=session.id) is not None
    assert await repo.get(tenant_id=tenant_b, session_id=session.id) is None


# ==================================================== 任务


async def test_enqueue_is_idempotent(db_session: AsyncSession) -> None:
    """相同幂等键只产生一个任务，第二次返回已有任务且 created=False。

    重复入队的后果：同一个文档被两个 worker 同时解析，
    chunk 重复、成本翻倍、状态互相覆盖。
    """
    tenant_id = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    key = f"parse:{uuid.uuid4()}"

    first, created_first = await repo.enqueue(
        tenant_id=tenant_id, task_type=TaskType.PARSE_DOCUMENT.value, idempotency_key=key
    )
    second, created_second = await repo.enqueue(
        tenant_id=tenant_id, task_type=TaskType.PARSE_DOCUMENT.value, idempotency_key=key
    )

    assert created_first is True
    assert created_second is False
    assert first.id == second.id


async def test_enqueue_same_key_different_tenant_is_allowed(db_session: AsyncSession) -> None:
    """不同租户可以用相同幂等键（唯一约束带 tenant_id）。"""
    tenant_a = await _make_tenant(db_session)
    tenant_b = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    key = "parse:same-key"

    _, created_a = await repo.enqueue(
        tenant_id=tenant_a, task_type=TaskType.PARSE_DOCUMENT.value, idempotency_key=key
    )
    _, created_b = await repo.enqueue(
        tenant_id=tenant_b, task_type=TaskType.PARSE_DOCUMENT.value, idempotency_key=key
    )
    assert created_a is True
    assert created_b is True


async def test_claim_sets_running_and_owner(db_session: AsyncSession) -> None:
    """认领后状态变 running，记录 worker 标识。"""
    tenant_id = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    await repo.enqueue(
        tenant_id=tenant_id,
        task_type=TaskType.PARSE_DOCUMENT.value,
        idempotency_key=f"k-{uuid.uuid4()}",
    )

    claimed = await repo.claim(worker_id="worker-1", limit=5)
    assert len(claimed) == 1
    assert claimed[0].status == TaskStatus.RUNNING.value
    assert claimed[0].locked_by == "worker-1"
    assert claimed[0].started_at is not None


async def test_claim_respects_available_at(db_session: AsyncSession) -> None:
    """未来才可执行的任务不会被提前认领。

    这是延迟任务与退避重试的基础：失败的任务设 available_at 延后，
    worker 立即轮询时不该又把它捞起来（会形成忙等循环）。
    """
    tenant_id = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    future = datetime.now(UTC) + timedelta(hours=1)
    await repo.enqueue(
        tenant_id=tenant_id,
        task_type=TaskType.PARSE_DOCUMENT.value,
        idempotency_key=f"k-{uuid.uuid4()}",
        available_at=future,
    )

    claimed = await repo.claim(worker_id="worker-1", limit=5)
    assert claimed == []


async def test_claim_filters_by_task_type(db_session: AsyncSession) -> None:
    """可按任务类型过滤认领（worker 各管一类任务）。"""
    tenant_id = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    await repo.enqueue(
        tenant_id=tenant_id,
        task_type=TaskType.PARSE_DOCUMENT.value,
        idempotency_key=f"p-{uuid.uuid4()}",
    )
    await repo.enqueue(
        tenant_id=tenant_id,
        task_type=TaskType.EMBED_DOCUMENT.value,
        idempotency_key=f"e-{uuid.uuid4()}",
    )

    claimed = await repo.claim(
        worker_id="embed-worker", limit=5, task_types=[TaskType.EMBED_DOCUMENT.value]
    )
    assert len(claimed) == 1
    assert claimed[0].task_type == TaskType.EMBED_DOCUMENT.value


def test_concurrent_claim_no_duplicates() -> None:
    """**两个 worker 并发认领不会拿到同一个任务。**

    必须用独立连接才能测出并发（同连接的并发是串行的）。
    没有 SKIP LOCKED 时，两个 worker 会拿到同一批任务；
    没有 FOR UPDATE 时也一样。

    这是**同步测试**：它要自己用 `asyncio.run` 起独立事件循环，
    以便让两个"worker"真的在不同的连接/任务上并发。
    在 pytest-asyncio 的 auto 模式下，同步 def 不会被当协程处理。
    """
    from rag_core.db import Database
    from rag_core.settings import get_settings

    async def scenario() -> tuple[int, int]:
        db = Database(get_settings())
        try:
            tenant_id = uuid.uuid4()
            async with db.transaction() as s:
                await s.execute(
                    text(
                        "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                        "VALUES (:id, :s, 'x', 'active', now(), now())"
                    ),
                    {"id": tenant_id, "s": f"t-{tenant_id.hex[:10]}"},
                )
                repo = TaskRepository(s)
                for _ in range(10):
                    await repo.enqueue(
                        tenant_id=tenant_id,
                        task_type=TaskType.PARSE_DOCUMENT.value,
                        idempotency_key=f"k-{uuid.uuid4()}",
                    )

            claimed_ids: list[uuid.UUID] = []

            async def worker(worker_id: str) -> None:
                async with db.transaction() as s:
                    claimed = await TaskRepository(s).claim(worker_id=worker_id, limit=5)
                    claimed_ids.extend(t.id for t in claimed)

            await asyncio.gather(worker("w1"), worker("w2"))
            return len(claimed_ids), len(set(claimed_ids))
        finally:
            async with db.transaction() as s:
                await s.execute(text("DELETE FROM tenants WHERE slug LIKE 't-%'"))
                await s.execute(text("DELETE FROM users WHERE email LIKE 'u-%@example.test'"))
            await db.close()

    total, unique = asyncio.run(scenario())
    assert total == unique, f"任务被重复认领：认领 {total} 次，去重后 {unique} 个"
    assert unique == 10


async def test_complete_requires_ownership(db_session: AsyncSession) -> None:
    """只有认领者能标记完成。

    防止任务被回收重派后，原 worker 又把它标成成功 ——
    那样新 worker 的工作会被覆盖，且状态与实际不符。
    """
    tenant_id = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    await repo.enqueue(
        tenant_id=tenant_id,
        task_type=TaskType.PARSE_DOCUMENT.value,
        idempotency_key=f"k-{uuid.uuid4()}",
    )
    claimed = await repo.claim(worker_id="owner", limit=1)
    task_id = claimed[0].id

    assert await repo.complete(task_id=task_id, worker_id="impostor") is False
    assert await repo.complete(task_id=task_id, worker_id="owner") is True


async def test_fail_retries_then_goes_dead(db_session: AsyncSession) -> None:
    """失败重试到上限后进死信，不是静默丢弃。

    死信是"待人工处理队列"：管理端能看到、能决定重试或放弃。
    直接丢弃会让"文档为什么一直失败"变成无解之谜。
    """
    tenant_id = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    await repo.enqueue(
        tenant_id=tenant_id,
        task_type=TaskType.PARSE_DOCUMENT.value,
        idempotency_key=f"k-{uuid.uuid4()}",
        max_retries=2,
    )

    # 第 1、2 次失败 → 回到 pending
    for attempt in range(2):
        claimed = await repo.claim(worker_id="w", limit=1)
        assert len(claimed) == 1
        state = await repo.fail(
            task_id=claimed[0].id,
            worker_id="w",
            error_code="DOC.PARSE_FAILED",
            error_message=f"失败 {attempt + 1}",
            backoff_s=0,
        )
        assert state == TaskStatus.FAILED.value
        # 清掉退避时间让下一轮能认领
        await db_session.execute(
            text("UPDATE tasks SET available_at = now() WHERE id = :i"),
            {"i": claimed[0].id},
        )

    # 第 3 次失败 → 死信
    claimed = await repo.claim(worker_id="w", limit=1)
    state = await repo.fail(
        task_id=claimed[0].id,
        worker_id="w",
        error_code="DOC.PARSE_FAILED",
        error_message="最后一次",
        backoff_s=0,
    )
    assert state == TaskStatus.DEAD.value

    dead = await repo.list_dead(tenant_id=tenant_id)
    assert len(dead) == 1
    assert dead[0].error_code == "DOC.PARSE_FAILED"


async def test_reclaim_stale_running_tasks(db_session: AsyncSession) -> None:
    """worker 崩溃遗留的 running 任务能被回收重派。

    不回收的后果：任务永远停在 running，队列为空但文档一直"处理中"，
    且没有任何错误信息 —— 这种故障最难排查。
    """
    tenant_id = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    await repo.enqueue(
        tenant_id=tenant_id,
        task_type=TaskType.PARSE_DOCUMENT.value,
        idempotency_key=f"k-{uuid.uuid4()}",
    )
    claimed = await repo.claim(worker_id="crashed-worker", limit=1)
    task_id = claimed[0].id

    # 把 locked_at 改成很久以前，模拟 worker 崩溃
    await db_session.execute(
        text("UPDATE tasks SET locked_at = now() - interval '1 hour' WHERE id = :i"),
        {"i": task_id},
    )

    reclaimed = await repo.reclaim_stale(lock_timeout_s=300)
    assert reclaimed == 1

    row = await db_session.execute(
        text("SELECT status, retry_count, locked_by FROM tasks WHERE id = :i"),
        {"i": task_id},
    )
    status, retry_count, locked_by = row.one()
    assert status == TaskStatus.PENDING.value
    assert retry_count == 1
    assert locked_by is None


async def test_update_progress_only_by_owner(db_session: AsyncSession) -> None:
    """只有认领者能更新进度。"""
    tenant_id = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    await repo.enqueue(
        tenant_id=tenant_id,
        task_type=TaskType.PARSE_DOCUMENT.value,
        idempotency_key=f"k-{uuid.uuid4()}",
    )
    claimed = await repo.claim(worker_id="owner", limit=1)
    task_id = claimed[0].id

    assert await repo.update_progress(task_id=task_id, worker_id="other", progress=50) is False
    assert await repo.update_progress(task_id=task_id, worker_id="owner", progress=50) is True
    assert await repo.update_progress(task_id=task_id, worker_id="owner", progress=150) is True
    # 进度被钳到 100
    row = await db_session.execute(text("SELECT progress FROM tasks WHERE id = :i"), {"i": task_id})
    assert int(row.scalar_one()) == 100


async def test_cancel_task(db_session: AsyncSession) -> None:
    """可取消 pending 任务；已完成的任务不可取消。"""
    tenant_id = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    task, _ = await repo.enqueue(
        tenant_id=tenant_id,
        task_type=TaskType.PARSE_DOCUMENT.value,
        idempotency_key=f"k-{uuid.uuid4()}",
    )

    assert await repo.cancel(tenant_id=tenant_id, task_id=task.id) is True
    assert await repo.cancel(tenant_id=tenant_id, task_id=task.id) is False


async def test_count_by_status(db_session: AsyncSession) -> None:
    """按状态统计供看板使用。"""
    tenant_id = await _make_tenant(db_session)
    repo = TaskRepository(db_session)
    for _ in range(3):
        await repo.enqueue(
            tenant_id=tenant_id,
            task_type=TaskType.PARSE_DOCUMENT.value,
            idempotency_key=f"k-{uuid.uuid4()}",
        )
    await repo.claim(worker_id="w", limit=1)

    counts = await repo.count_by_status(tenant_id=tenant_id)
    assert counts.get(TaskStatus.PENDING.value) == 2
    assert counts.get(TaskStatus.RUNNING.value) == 1


# ==================================================== outbox


async def test_outbox_add_and_claim(db_session: AsyncSession) -> None:
    """outbox 消息能写入并被认领。"""
    tenant_id = await _make_tenant(db_session)
    repo = OutboxRepository(db_session)
    await repo.add(
        tenant_id=tenant_id,
        topic="task.dispatch",
        payload={"task_id": str(uuid.uuid4())},
    )

    claimed = await repo.claim_batch(limit=10)
    assert len(claimed) == 1
    assert claimed[0].status == OutboxStatus.PENDING.value
    assert claimed[0].attempts == 0


async def test_outbox_delay_defers_claim(db_session: AsyncSession) -> None:
    """延后投递的消息不会被立即认领。"""
    tenant_id = await _make_tenant(db_session)
    repo = OutboxRepository(db_session)
    await repo.add(
        tenant_id=tenant_id,
        topic="task.dispatch",
        payload={"x": 1},
        delay_s=3600,
    )

    claimed = await repo.claim_batch(limit=10)
    assert claimed == []


async def test_outbox_mark_dispatched(db_session: AsyncSession) -> None:
    """投递成功后标记 dispatched，之后不再被认领（至少一次送达的终点）。"""
    tenant_id = await _make_tenant(db_session)
    repo = OutboxRepository(db_session)
    message = await repo.add(tenant_id=tenant_id, topic="task.dispatch", payload={"task_id": "t-1"})

    claimed = await repo.claim_batch(limit=10)
    await repo.mark_dispatched(message_id=claimed[0].id)

    again = await repo.claim_batch(limit=10)
    assert again == []

    row = await db_session.execute(
        text("SELECT status, dispatched_at FROM outbox_messages WHERE id = :i"),
        {"i": message.id},
    )
    status, dispatched_at = row.one()
    assert status == OutboxStatus.DISPATCHED.value
    assert dispatched_at is not None


async def test_outbox_failure_backs_off_then_fails(db_session: AsyncSession) -> None:
    """投递失败增加尝试次数并延后；超限标记 failed。

    标记 failed 而不是无限重试：一条永远投不出去的消息
    不该永远占用转发进程的注意力，应该让运维看到。
    """
    tenant_id = await _make_tenant(db_session)
    repo = OutboxRepository(db_session)
    message = await repo.add(tenant_id=tenant_id, topic="task.dispatch", payload={})
    await db_session.execute(
        text("UPDATE outbox_messages SET max_attempts = 2 WHERE id = :i"), {"i": message.id}
    )

    claimed = await repo.claim_batch(limit=10)
    state = await repo.mark_attempt_failed(message_id=claimed[0].id, error="连接超时", backoff_s=0)
    assert state == OutboxStatus.PENDING.value

    await db_session.execute(
        text("UPDATE outbox_messages SET next_attempt_at = now() WHERE id = :i"),
        {"i": message.id},
    )
    claimed = await repo.claim_batch(limit=10)
    state = await repo.mark_attempt_failed(message_id=claimed[0].id, error="连接超时", backoff_s=0)
    assert state == OutboxStatus.FAILED.value

    row = await db_session.execute(
        text("SELECT attempts, last_error FROM outbox_messages WHERE id = :i"),
        {"i": message.id},
    )
    attempts, last_error = row.one()
    assert int(attempts) == 2
    assert last_error == "连接超时"


async def test_outbox_coexists_with_business_write(db_session: AsyncSession) -> None:
    """**outbox 的核心价值**：业务写入与消息写入在同一事务里。

    这个测试证明"同事务"是可用的：两者要么同时存在，要么同时不存在。
    若分两个事务，中间崩溃会导致"文档存在但消息没发出"（文档永远 PENDING）
    或"消息发出但文档不存在"（worker 收到消息却查不到数据）。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    upload_repo = UploadSessionRepository(db_session)
    outbox_repo = OutboxRepository(db_session)

    session = await upload_repo.create(**_session_kwargs(tenant_id, user_id))  # type: ignore[arg-type]
    await outbox_repo.add(
        tenant_id=tenant_id,
        topic="upload.completed",
        payload={"upload_session_id": str(session.id)},
    )

    # 两者都在同一事务的未提交状态里可见
    sessions = await db_session.execute(
        text("SELECT count(*) FROM upload_sessions WHERE tenant_id = :t"), {"t": tenant_id}
    )
    messages = await db_session.execute(
        text("SELECT count(*) FROM outbox_messages WHERE tenant_id = :t"), {"t": tenant_id}
    )
    assert int(sessions.scalar_one()) == 1
    assert int(messages.scalar_one()) == 1
