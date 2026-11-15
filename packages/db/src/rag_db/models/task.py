"""异步任务与 outbox 模型。

## 任务表与文档状态表的分工

`documents.status` 是**业务状态**（这份文档最终能用了吗）；
`tasks.status` 是**执行记录**（第 3 次重试跑到哪了）。

两者分离的理由：
- 一次文档处理会产生多个任务（解析/切分/向量化/索引各一个）；
- 每个任务可能重试多次，每次重试是新的执行记录；
- 管理端要回答"重试了 3 次、前 2 次为什么失败"，这只能靠任务历史。

如果合成一张表，你要么丢失重试历史，要么在文档表上塞一堆"上次错误、
上次耗时、重试次数"字段，然后随任务类型增多而失控。

## outbox 为什么必要

跨存储一致性：API 写 PG（文档记录）+ 发队列消息（通知 worker）。
两者不可能在一个事务里完成（队列不是 PG）。

不用 outbox 的两种失败：
- 写库成功、发消息失败 → 文档永远 PENDING，用户看到"处理中"再也不动；
- 发消息成功、写库失败 → worker 收到消息但查不到文档，反复重试后进死信。

outbox 的做法：**业务写入与消息写入在同一个 PG 事务里**，
后台任务轮询 outbox 表转发到队列。这保证"业务数据存在 ⇒ 消息最终会被发出"。

代价是**至少一次送达**（消息可能重复投递），因此**消费者必须幂等**。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from rag_contracts.enums.uploads import OutboxStatus, TaskStatus, TaskType
from rag_db.base import (
    Base,
    OptimisticLockMixin,
    TenantScopedMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class Task(Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin, OptimisticLockMixin):
    """一次异步任务执行。

    与文档是多对一：一份文档可能有解析/切分/向量化/索引多个任务。
    """

    __tablename__ = "tasks"

    task_type: Mapped[TaskType] = mapped_column(
        String(64),
        nullable=False,
        comment="任务类型，决定 worker 用哪个 handler",
    )
    status: Mapped[TaskStatus] = mapped_column(
        String(32),
        nullable=False,
        default=TaskStatus.PENDING,
        server_default=TaskStatus.PENDING.value,
        comment="任务状态：pending/running/succeeded/failed/dead/cancelled",
    )

    # ---- 幂等键 ----
    # 同一个业务操作只应产生一个任务。例如"解析文档 D 的版本 V"，
    # 重复提交应复用同一个任务，而不是产生两个任务同时跑（浪费算力、
    # 且两个任务会互相覆盖状态）。
    idempotency_key: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="幂等键（如 parse:{document_id}:{version_id}），唯一约束防重复入队",
    )

    # ---- 关联业务对象（可空，因为不同任务类型关联的对象不同）----
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,
        comment="关联文档",
    )
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=True,
        comment="关联文档版本",
    )
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=True,
        comment="关联知识库",
    )

    # ---- 执行信息 ----
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="任务参数（handler 自定义）",
    )
    progress: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="进度百分比 0-100（长任务回写给前端轮询）",
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="已重试次数",
    )
    max_retries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
        server_default="3",
        comment="最大重试次数，超过进死信",
    )

    # ---- 认领信息 ----
    # 记录谁在跑：便于排查"任务卡住了，是哪个 worker 挂了吗"
    locked_by: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="认领该任务的 worker 标识",
    )
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="认领时间。超时未完成的会被回收重派",
    )

    # ---- 结果 ----
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="失败错误码",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="失败详情",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="开始执行时间",
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="结束时间（成功或失败）",
    )

    # ---- 调度 ----
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
        comment="最早可执行时间（延迟任务/退避重试用）",
    )

    __table_args__ = (
        # 幂等键唯一：同一业务操作不会重复入队
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_tasks_tenant_idempotency"),
        # **worker 认领的主索引**：
        # WHERE status='pending' AND available_at <= now() ORDER BY available_at
        # 部分索引只包含待执行任务，索引体积远小于全表（已完成任务是绝大多数）
        Index(
            "ix_tasks_claimable",
            "available_at",
            postgresql_where=text("status = 'pending'"),
        ),
        # 按文档查任务历史（管理端）
        Index("ix_tasks_document", "document_id", "created_at"),
        # 死信队列查看
        Index(
            "ix_tasks_dead",
            "tenant_id",
            "created_at",
            postgresql_where=text("status = 'dead'"),
        ),
        # 卡住的任务：running 但 locked_at 很久以前（worker 崩溃）
        Index(
            "ix_tasks_stale_running",
            "locked_at",
            postgresql_where=text("status = 'running'"),
        ),
        CheckConstraint("progress >= 0 AND progress <= 100", name="ck_tasks_progress_range"),
        CheckConstraint("retry_count >= 0", name="ck_tasks_retry_non_negative"),
        CheckConstraint("max_retries >= 0", name="ck_tasks_max_retries_non_negative"),
    )

    def __repr__(self) -> str:
        return f"<Task {self.task_type} {self.idempotency_key} ({self.status})>"


class OutboxMessage(Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin):
    """发件箱消息：保证业务写入与消息投递的最终一致。

    **必须在业务事务里插入**，不能单独提交——否则又回到"两个写操作"的问题。

    转发流程（worker 的定时任务）：
      1. 认领待发消息（FOR UPDATE SKIP LOCKED）；
      2. 投递到队列；
      3. 成功则标记 dispatched，失败则增加 attempts 并延后重试；
      4. attempts 超限 → failed，需人工介入。

    消费者会收到重复消息（至少一次），所以**每个 handler 必须幂等**。
    幂等键就是 `tasks.idempotency_key`：收到重复消息时，插入任务会撞唯一约束，
    handler 捕获后直接返回成功即可。
    """

    __tablename__ = "outbox_messages"

    topic: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="队列主题（如 task.dispatch），决定消费者",
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="消息体。**不得包含敏感数据**：它会被写进队列与日志",
    )
    status: Mapped[OutboxStatus] = mapped_column(
        String(32),
        nullable=False,
        default=OutboxStatus.PENDING,
        server_default=OutboxStatus.PENDING.value,
        comment="pending/dispatched/failed",
    )
    attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="已尝试投递次数",
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10,
        server_default="10",
        comment="最大尝试次数，超过标记 failed",
    )
    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="最后一次投递失败原因",
    )
    # 延迟投递：定期重发的消息可以设未来时间，避免立刻重试
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
        comment="下次尝试时间（退避用）",
    )
    dispatched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="成功投递时间",
    )

    __table_args__ = (
        # 转发任务扫描：WHERE status='pending' AND next_attempt_at <= now()
        Index(
            "ix_outbox_dispatchable",
            "next_attempt_at",
            postgresql_where=text("status = 'pending'"),
        ),
        # 按主题统计积压（告警用）
        Index(
            "ix_outbox_topic_status",
            "topic",
            "status",
        ),
        # 清理已完成消息：WHERE status='dispatched' AND dispatched_at < cutoff
        Index(
            "ix_outbox_dispatched_at",
            "dispatched_at",
            postgresql_where=text("status = 'dispatched'"),
        ),
        CheckConstraint("attempts >= 0", name="ck_outbox_attempts_non_negative"),
    )

    def __repr__(self) -> str:
        return f"<OutboxMessage {self.topic} ({self.status})>"
