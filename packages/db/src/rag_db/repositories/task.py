"""上传会话 / 任务 / outbox 仓储。

这三个仓储的共同点：**都是并发敏感的状态机**。
- 上传会话：多个分片并发上传、complete 只能成功一次；
- 任务：多个 worker 并发认领不能重复；
- outbox：多个转发进程并发投递不能重复标记。

所有状态流转都用「带条件的原子 UPDATE + RETURNING」，
不用「读出来判断再写回」。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums.uploads import OutboxStatus, TaskStatus, UploadSessionStatus
from rag_db.models.task import OutboxMessage, Task
from rag_db.models.upload import UploadSession


class UploadSessionRepository:
    """分片上传会话数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        created_by: uuid.UUID,
        filename: str,
        total_size: int,
        content_type: str,
        part_size: int,
        total_parts: int,
        storage_key_prefix: str,
        storage_backend: str,
        content_hash: str | None = None,
        knowledge_base_id: uuid.UUID | None = None,
        ttl: timedelta = timedelta(hours=24),
    ) -> UploadSession:
        # 用 timezone-aware UTC 时间：库里存的是 timezone=True 的时间戳，
        # 混用 naive datetime 会让 PG 按会话时区解释，产生偏移错误。
        session = UploadSession(
            tenant_id=tenant_id,
            created_by=created_by,
            filename=filename,
            total_size=total_size,
            content_type=content_type,
            content_hash=content_hash,
            part_size=part_size,
            total_parts=total_parts,
            storage_key_prefix=storage_key_prefix,
            storage_backend=storage_backend,
            knowledge_base_id=knowledge_base_id,
            expires_at=datetime.now(UTC) + ttl,
        )
        self._session.add(session)
        await self._session.flush()
        return session

    async def get(self, *, tenant_id: uuid.UUID, session_id: uuid.UUID) -> UploadSession | None:
        """取会话。必须带 tenant_id，防止跨租户访问别人的上传会话。"""
        result = await self._session.execute(
            select(UploadSession)
            .where(
                UploadSession.id == session_id,
                UploadSession.tenant_id == tenant_id,
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def mark_part_uploaded(self, *, session_id: uuid.UUID, expected_version: int) -> bool:
        """记录一个分片已完成（原子自增）。

        用 `uploaded_parts = uploaded_parts + 1` 而不是 Python 侧 +1：
        并发上传多个分片时，读-改-写会丢更新，导致计数偏小，
        最终判定"分片不齐"而拒绝合并 —— 用户明明传完了却合并不了。

        带乐观锁版本号：如果会话已被别的操作改成 COMPLETING/ABORTED，
        更新会失败，调用方据此拒绝这次分片。
        """
        result = await self._session.execute(
            update(UploadSession)
            .where(
                UploadSession.id == session_id,
                UploadSession.version == expected_version,
                UploadSession.status.in_(
                    [UploadSessionStatus.INITIATED.value, UploadSessionStatus.UPLOADING.value]
                ),
            )
            .values(
                uploaded_parts=UploadSession.uploaded_parts + 1,
                version=UploadSession.version + 1,
                status=UploadSessionStatus.UPLOADING.value,
            )
            .returning(UploadSession.version)
        )
        return result.scalar_one_or_none() is not None

    async def begin_complete(
        self, *, tenant_id: uuid.UUID, session_id: uuid.UUID
    ) -> UploadSession | None:
        """把会话从 UPLOADING 原子切到 COMPLETING。

        返回 None 表示状态不允许（已完成/已中止/正在合并）。
        **这是防重复合并的关键**：两个请求同时 complete 时，
        只有一个能把状态从 UPLOADING 改成 COMPLETING，另一个立刻失败。
        没有这个原子切换，两个请求会同时合并分片、同时创建 blob，
        产生重复数据。
        """
        result = await self._session.execute(
            update(UploadSession)
            .where(
                UploadSession.id == session_id,
                UploadSession.tenant_id == tenant_id,
                UploadSession.status.in_(
                    [UploadSessionStatus.INITIATED.value, UploadSessionStatus.UPLOADING.value]
                ),
            )
            .values(
                status=UploadSessionStatus.COMPLETING.value,
                version=UploadSession.version + 1,
            )
            .returning(UploadSession)
        )
        return result.scalar_one_or_none()

    async def finish_complete(self, *, session_id: uuid.UUID, blob_id: uuid.UUID) -> None:
        """合并成功，标记会话完成。"""
        await self._session.execute(
            update(UploadSession)
            .where(UploadSession.id == session_id)
            .values(
                status=UploadSessionStatus.COMPLETED.value,
                blob_id=blob_id,
                completed_at=func.now(),
                version=UploadSession.version + 1,
            )
        )

    async def mark_failed(
        self, *, session_id: uuid.UUID, error_code: str, error_message: str
    ) -> None:
        await self._session.execute(
            update(UploadSession)
            .where(UploadSession.id == session_id)
            .values(
                status=UploadSessionStatus.FAILED.value,
                error_code=error_code,
                error_message=error_message,
                version=UploadSession.version + 1,
            )
        )

    async def abort(self, *, tenant_id: uuid.UUID, session_id: uuid.UUID) -> bool:
        """中止会话。已完成的会话不能中止。"""
        result = await self._session.execute(
            update(UploadSession)
            .where(
                UploadSession.id == session_id,
                UploadSession.tenant_id == tenant_id,
                UploadSession.status.not_in(
                    [UploadSessionStatus.COMPLETED.value, UploadSessionStatus.ABORTED.value]
                ),
            )
            .values(
                status=UploadSessionStatus.ABORTED.value,
                aborted_at=func.now(),
                version=UploadSession.version + 1,
            )
            .returning(UploadSession.id)
        )
        return result.scalar_one_or_none() is not None

    async def list_expired(self, *, limit: int = 100) -> list[UploadSession]:
        """列出过期但未清理的会话（清理任务用）。

        只挑"未完成"的：已完成的会话过 Expires 也不该回收，
        它们指向的 blob 已经被文档引用了。
        """
        result = await self._session.execute(
            select(UploadSession)
            .where(
                UploadSession.expires_at < func.now(),
                UploadSession.status.in_(
                    [
                        UploadSessionStatus.INITIATED.value,
                        UploadSessionStatus.UPLOADING.value,
                        UploadSessionStatus.COMPLETING.value,
                    ]
                ),
            )
            .order_by(UploadSession.expires_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return list(result.scalars())


class TaskRepository:
    """异步任务数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def enqueue(
        self,
        *,
        tenant_id: uuid.UUID,
        task_type: str,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        document_id: uuid.UUID | None = None,
        document_version_id: uuid.UUID | None = None,
        knowledge_base_id: uuid.UUID | None = None,
        max_retries: int = 3,
        available_at: datetime | None = None,
    ) -> tuple[Task, bool]:
        """入队任务。返回 (task, created)。

        `created=False` 表示幂等键已存在，复用了已有任务。
        用 `INSERT ... ON CONFLICT DO NOTHING` 而不是"先查再插"：
        并发提交同一个幂等键时，先查再插会撞唯一约束报错，
        ON CONFLICT 让数据库原子地处理"已存在"这种情况。

        **注意：`created=False` 时应返回已有任务而不是报错**——
        对调用方来说"任务已在队列里"就是成功，不是失败。
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = (
            pg_insert(Task)
            .values(
                tenant_id=tenant_id,
                task_type=task_type,
                idempotency_key=idempotency_key,
                payload=payload or {},
                document_id=document_id,
                document_version_id=document_version_id,
                knowledge_base_id=knowledge_base_id,
                max_retries=max_retries,
                status=TaskStatus.PENDING.value,
                available_at=available_at or func.now(),
            )
            .on_conflict_do_nothing(constraint="uq_tasks_tenant_idempotency")
            .returning(Task)
        )
        result = await self._session.execute(stmt)
        created_task = result.scalar_one_or_none()
        if created_task is not None:
            return created_task, True

        existing = await self._session.execute(
            select(Task)
            .where(
                Task.tenant_id == tenant_id,
                Task.idempotency_key == idempotency_key,
            )
            .limit(1)
        )
        return existing.scalar_one(), False

    async def claim(
        self,
        *,
        worker_id: str,
        limit: int = 1,
        task_types: Sequence[str] | None = None,
        lock_timeout_s: int = 300,
    ) -> list[Task]:
        """原子认领待执行任务。

        **这是 worker 并发安全的核心。**

        `FOR UPDATE SKIP LOCKED` 的三个作用：
        1. `FOR UPDATE`：锁住候选行，其它事务不能同时改；
        2. `SKIP LOCKED`：别的 worker 跳过已锁定的行而不是阻塞等待
           —— 没有它，N 个 worker 会串行排队，吞吐量退化成单 worker；
        3. 配合 `available_at <= now()`：延迟任务不会被提前拿走。

        `lock_timeout_s` 用于回收崩溃 worker 的任务：
        认领超过这个时间还没结束的任务，会被视为 worker 挂了，
        重新放回 PENDING（由 `reclaim_stale` 处理）。
        """
        conditions = [
            Task.status == TaskStatus.PENDING.value,
            Task.available_at <= func.now(),
        ]
        if task_types:
            conditions.append(Task.task_type.in_(list(task_types)))

        subq = (
            select(Task.id)
            .where(and_(*conditions))
            .order_by(Task.available_at, Task.created_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )

        result = await self._session.execute(
            update(Task)
            .where(Task.id.in_(subq))
            .values(
                status=TaskStatus.RUNNING.value,
                locked_by=worker_id,
                locked_at=func.now(),
                started_at=func.now(),
                version=Task.version + 1,
            )
            .returning(Task)
        )
        return list(result.scalars())

    async def reclaim_stale(self, *, lock_timeout_s: int = 300) -> int:
        """回收 worker 崩溃后遗留的 RUNNING 任务。

        判据：status=running 且 locked_at 早于 now() - lock_timeout。

        **这一步不能省**：worker 被 kill、容器被驱逐、进程 OOM 时，
        任务会永远停在 running，再也不会被任何人处理。
        表现为"文档一直显示处理中"，而队列是空的。

        回收时 `retry_count + 1`：避免一个让 worker 崩溃的任务被无限重派。
        """
        cutoff = func.now() - text(f"interval '{int(lock_timeout_s)} seconds'")
        result = await self._session.execute(
            update(Task)
            .where(
                Task.status == TaskStatus.RUNNING.value,
                Task.locked_at < cutoff,
            )
            .values(
                status=TaskStatus.PENDING.value,
                locked_by=None,
                locked_at=None,
                retry_count=Task.retry_count + 1,
                available_at=func.now(),
                version=Task.version + 1,
            )
            .returning(Task.id)
        )
        return len(list(result.scalars()))

    async def complete(
        self,
        *,
        task_id: uuid.UUID,
        worker_id: str,
    ) -> bool:
        """标记任务成功。

        带上 `locked_by` 条件：只有认领它的 worker 才能标记完成。
        防止任务被回收重派给另一个 worker 后，原 worker 又把它标成成功。
        """
        result = await self._session.execute(
            update(Task)
            .where(
                Task.id == task_id,
                Task.status == TaskStatus.RUNNING.value,
                Task.locked_by == worker_id,
            )
            .values(
                status=TaskStatus.SUCCEEDED.value,
                progress=100,
                finished_at=func.now(),
                version=Task.version + 1,
            )
            .returning(Task.id)
        )
        return result.scalar_one_or_none() is not None

    async def fail(
        self,
        *,
        task_id: uuid.UUID,
        worker_id: str,
        error_code: str,
        error_message: str,
        retry: bool = True,
        backoff_s: int = 30,
    ) -> str:
        """标记任务失败，返回新状态（failed / dead）。

        有重试次数时回到 PENDING 并设置 `available_at` 延后
        （指数退避由调用方计算传入）；超过上限则进死信（DEAD）。

        死信不是终点而是**待人工处理队列**：管理端能看到，人工决定重试或放弃。
        直接丢弃会让"文档为什么一直失败"变成无解之谜。
        """
        # 先读当前重试次数决定下一步（这一读必须在同一事务里）
        current = await self._session.execute(
            select(Task.retry_count, Task.max_retries)
            .where(Task.id == task_id, Task.locked_by == worker_id)
            .limit(1)
        )
        row = current.one_or_none()
        if row is None:
            return "not_owned"

        retry_count, max_retries = int(row[0]), int(row[1])
        exhausted = not retry or retry_count >= max_retries

        if exhausted:
            await self._session.execute(
                update(Task)
                .where(Task.id == task_id, Task.locked_by == worker_id)
                .values(
                    status=TaskStatus.DEAD.value,
                    error_code=error_code,
                    error_message=error_message,
                    finished_at=func.now(),
                    locked_by=None,
                    locked_at=None,
                    version=Task.version + 1,
                )
            )
            return TaskStatus.DEAD.value

        await self._session.execute(
            update(Task)
            .where(Task.id == task_id, Task.locked_by == worker_id)
            .values(
                status=TaskStatus.PENDING.value,
                retry_count=Task.retry_count + 1,
                error_code=error_code,
                error_message=error_message,
                available_at=func.now() + text(f"interval '{int(backoff_s)} seconds'"),
                locked_by=None,
                locked_at=None,
                version=Task.version + 1,
            )
        )
        return TaskStatus.FAILED.value

    async def update_progress(self, *, task_id: uuid.UUID, worker_id: str, progress: int) -> bool:
        """回写进度，供前端轮询。只允许认领者更新。"""
        result = await self._session.execute(
            update(Task)
            .where(
                Task.id == task_id,
                Task.locked_by == worker_id,
                Task.status == TaskStatus.RUNNING.value,
            )
            .values(progress=max(0, min(100, progress)))
            .returning(Task.id)
        )
        return result.scalar_one_or_none() is not None

    async def touch_lock(self, *, task_id: uuid.UUID, worker_id: str) -> bool:
        """刷新认领时间（长任务的续锁）。

        ## 为什么必须有这个方法

        `reclaim_stale` 的判据是 `locked_at < now() - lock_timeout`。
        引擎只在认领时写一次 `locked_at`。索引重建这类任务要跑几十分钟，
        远超 `lock_timeout`（默认 600 秒），于是会在跑到一半时被
        另一个 worker 抢走重跑——两个进程同时调 embedding API（花双倍钱）、
        同时写旁表（互相阻塞）。

        所以长任务 handler 每处理完一批就调一次本方法。

        ## 为什么带 `worker_id` 条件

        任务可能已被回收并派给别的 worker。此时本进程续锁必须失败，
        不能把 `locked_at` 改成"一个已经不属于我"的任务的锁——
        那会让原 worker 误以为还持有锁而继续跑。

        返回 False 时调用方应**主动停止**（说明锁已被抢走）。
        """
        result = await self._session.execute(
            update(Task)
            .where(
                Task.id == task_id,
                Task.locked_by == worker_id,
                Task.status == TaskStatus.RUNNING.value,
            )
            .values(locked_at=func.now())
            .returning(Task.id)
        )
        return result.scalar_one_or_none() is not None

    async def cancel(self, *, tenant_id: uuid.UUID, task_id: uuid.UUID) -> bool:
        """取消任务。已结束的任务不能取消。"""
        result = await self._session.execute(
            update(Task)
            .where(
                Task.id == task_id,
                Task.tenant_id == tenant_id,
                Task.status.in_([TaskStatus.PENDING.value, TaskStatus.RUNNING.value]),
            )
            .values(
                status=TaskStatus.CANCELLED.value,
                finished_at=func.now(),
                version=Task.version + 1,
            )
            .returning(Task.id)
        )
        return result.scalar_one_or_none() is not None

    async def list_dead(self, *, tenant_id: uuid.UUID, limit: int = 50) -> list[Task]:
        """列出死信任务（管理端）。"""
        result = await self._session.execute(
            select(Task)
            .where(Task.tenant_id == tenant_id, Task.status == TaskStatus.DEAD.value)
            .order_by(Task.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars())

    async def count_by_status(self, *, tenant_id: uuid.UUID) -> dict[str, int]:
        """按状态统计（看板：积压多少、死了多少）。"""
        result = await self._session.execute(
            select(Task.status, func.count())
            .where(Task.tenant_id == tenant_id)
            .group_by(Task.status)
        )
        return {str(row[0]): int(row[1]) for row in result}

    async def get_for_tenant(self, *, tenant_id: uuid.UUID, task_id: uuid.UUID) -> Task | None:
        """按 ID 取任务（用户面轮询进度用）。

        **必须带 tenant_id**：任务 ID 是 UUID（不可枚举），但只靠它查询
        就等于"知道 ID 就能看"，而任务行里有 document_id / 错误详情。
        租户条件让跨租户探测返回 None（上层转 404）。

        与 `claim()` 的区别：那个是跨租户认领（必须看到全部租户），
        本方法是租户内读取，语义完全不同，不能合并。
        """
        result = await self._session.execute(
            select(Task).where(Task.id == task_id, Task.tenant_id == tenant_id).limit(1)
        )
        return result.scalar_one_or_none()

    async def list_filtered(
        self,
        *,
        tenant_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        task_type: str | None = None,
        document_id: uuid.UUID | None = None,
        include_total: bool = False,
    ) -> tuple[list[Task], int | None]:
        """管理端任务台账：按状态/类型/文档过滤。

        ## 为什么状态过滤在 SQL 里而不是查出来再筛

        任务表在活跃系统上是增长最快的表之一（每个文档 4+ 个任务，
        每次重试一条新行）。查出来再筛会让 limit 20 变成"扫 20 条但
        留下 2 条"，前端看到的分页全是空洞。

        排序用 `created_at DESC`：排障时最近的任务最有价值。
        """
        conditions = [Task.tenant_id == tenant_id]
        if status is not None:
            conditions.append(Task.status == status)
        if task_type is not None:
            conditions.append(Task.task_type == task_type)
        if document_id is not None:
            conditions.append(Task.document_id == document_id)

        stmt = (
            select(Task)
            .where(*conditions)
            .order_by(Task.created_at.desc(), Task.id.desc())
            .limit(limit)
        )
        if offset > 0:
            stmt = stmt.offset(offset)

        rows = list((await self._session.execute(stmt)).scalars())

        total: int | None = None
        if include_total:
            count_stmt = select(func.count()).select_from(Task).where(*conditions)
            total = int((await self._session.execute(count_stmt)).scalar_one())

        return rows, total

    async def list_dead_paginated(
        self, *, tenant_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> tuple[list[Task], int | None]:
        """死信列表（带分页）。

        `list_dead` 是内部脚本用的（固定 limit、无分页）；
        管理端需要翻页与总数，所以单独一个方法——把两者合并会让
        "脚本只想取最近 50 条"也触发 COUNT(*)。
        """
        return await self.list_filtered(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
            status=TaskStatus.DEAD.value,
            include_total=True,
        )

    async def count_dead(self, *, tenant_id: uuid.UUID) -> int:
        """死信数量（看板红点）。

        独立方法而不是复用 `count_by_status()[dead]`：看板首屏只要这一个
        数字时不值得 GROUP BY 全表。
        """
        result = await self._session.execute(
            select(func.count())
            .select_from(Task)
            .where(Task.tenant_id == tenant_id, Task.status == TaskStatus.DEAD.value)
        )
        return int(result.scalar_one())


class OutboxRepository:
    """发件箱数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        *,
        tenant_id: uuid.UUID,
        topic: str,
        payload: dict[str, Any],
        delay_s: int = 0,
    ) -> OutboxMessage:
        """写入一条待发消息。

        **必须在业务事务里调用**（与业务写入同一个 session/transaction），
        否则又变成"两个独立写操作"，outbox 就失去意义了。

        payload 不得含敏感数据：它会进队列、可能被日志记录。
        只放 ID 和必要参数，让消费者自己去查数据。
        """
        if delay_s > 0:
            next_at: Any = func.now() + text(f"interval '{int(delay_s)} seconds'")
        else:
            next_at = func.now()
        message = OutboxMessage(
            tenant_id=tenant_id,
            topic=topic,
            payload=payload,
            next_attempt_at=next_at,
        )
        self._session.add(message)
        await self._session.flush()
        return message

    async def claim_batch(self, *, limit: int = 100) -> list[OutboxMessage]:
        """认领一批待投递消息。

        同 TaskRepository.claim：`FOR UPDATE SKIP LOCKED` 让多个转发进程
        并行工作而不重复认领同一条消息。
        """
        subq = (
            select(OutboxMessage.id)
            .where(
                OutboxMessage.status == OutboxStatus.PENDING.value,
                OutboxMessage.next_attempt_at <= func.now(),
            )
            .order_by(OutboxMessage.next_attempt_at, OutboxMessage.created_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(
            select(OutboxMessage).where(OutboxMessage.id.in_(subq))
        )
        return list(result.scalars())

    async def mark_dispatched(self, *, message_id: uuid.UUID) -> None:
        await self._session.execute(
            update(OutboxMessage)
            .where(OutboxMessage.id == message_id)
            .values(
                status=OutboxStatus.DISPATCHED.value,
                dispatched_at=func.now(),
            )
        )

    async def mark_attempt_failed(
        self,
        *,
        message_id: uuid.UUID,
        error: str,
        backoff_s: int = 30,
    ) -> str:
        """投递失败：增加尝试次数并延后重试，超限则标记 failed。

        返回新状态（pending / failed）。
        """
        current = await self._session.execute(
            select(OutboxMessage.attempts, OutboxMessage.max_attempts)
            .where(OutboxMessage.id == message_id)
            .limit(1)
        )
        row = current.one_or_none()
        if row is None:
            return "missing"

        attempts, max_attempts = int(row[0]), int(row[1])
        if attempts + 1 >= max_attempts:
            await self._session.execute(
                update(OutboxMessage)
                .where(OutboxMessage.id == message_id)
                .values(
                    status=OutboxStatus.FAILED.value,
                    attempts=OutboxMessage.attempts + 1,
                    last_error=error,
                )
            )
            return OutboxStatus.FAILED.value

        await self._session.execute(
            update(OutboxMessage)
            .where(OutboxMessage.id == message_id)
            .values(
                attempts=OutboxMessage.attempts + 1,
                last_error=error,
                next_attempt_at=func.now() + text(f"interval '{int(backoff_s)} seconds'"),
            )
        )
        return OutboxStatus.PENDING.value

    async def purge_dispatched(self, *, older_than_s: int = 86400) -> int:
        """物理删除已投递且超过保留期的消息。

        不清理的话 outbox 表会无限增长。保留一天是为了出问题时
        还能查到"这条消息确实发过、什么时候发的"。
        """
        from sqlalchemy import delete

        result = await self._session.execute(
            delete(OutboxMessage)
            .where(
                OutboxMessage.status == OutboxStatus.DISPATCHED.value,
                OutboxMessage.dispatched_at
                < func.now() - text(f"interval '{int(older_than_s)} seconds'"),
            )
            .returning(OutboxMessage.id)
        )
        return len(list(result.scalars()))
