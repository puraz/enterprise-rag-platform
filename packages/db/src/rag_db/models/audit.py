"""审计日志与哈希链。

## 为什么需要哈希链

合规要求"审计日志不可篡改"。仅靠权限控制不够——拥有数据库超级权限的人
（或拿到 dump 的攻击者）可以直接 UPDATE 历史记录，事后无法发现。

哈希链的做法：每条记录包含 `prev_hash`，并计算

    hash = SHA256(prev_hash || seq || tenant_id || action || ... || created_at)

篡改任意一条记录，它的 hash 就会变，导致下一条的 prev_hash 对不上，
整条链在该点之后全部失效。验证时从头遍历或从最近的锚点回溯即可发现。

这**不是绝对防篡改**（有 DB 权限的人可以重算整条链），但：

- 让"悄悄改一条"变成"必须重算全部后续记录"，成本与风险大幅上升；
- 定期把 `chain_head` 的 hash 外发（写入对象存储的 WORM 桶、或发给独立审计服务），
  就能把"重算整条链"也挡住——外部的锚点改不了。

## 写入串行化

追加必须知道前一条的 hash，天然串行。用 `audit_chain_heads` 表：

    UPDATE audit_chain_heads
       SET last_seq = last_seq + 1, last_hash = :new_hash
     WHERE tenant_id = :t
    RETURNING last_seq

一条语句原子推进。为什么不直接 `SELECT ... FROM audit_logs ORDER BY seq DESC LIMIT 1
FOR UPDATE`：那要扫日志表（即使有索引也要维护锁），而链头表只有一行，
且不随日志增长。**代价是同一租户的审计写入串行**——这是哈希链的固有代价，
接受它：审计写入本来就不是高频操作（相比检索/问答）。

## 分区

按 `created_at` 月分区，便于：
- 按时间范围清理（保留 180 天）；
- 查询时分区裁剪（查"上个月谁删了文档"只扫一个分区）。
分区由 A3-6 的迁移创建，这里只保证模型字段可用于分区键。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import INET, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from rag_contracts.enums.audit import AuditAction, AuditOutcome, AuditTargetType
from rag_contracts.enums.tenancy import PlatformRole, TenantRole
from rag_db.base import (
    Base,
    TenantScopedMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class AuditChainHead(Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin):
    """每个租户的审计链头。

    只有一行/租户，记录当前链的最后一个序号与哈希。
    追加审计时用它原子推进（见模块 docstring）。

    单独成表而不是"查最后一条日志"的原因：链头表恒为一行，
    UPDATE 命中单行、锁竞争范围小；查日志表需要索引扫描 + 行锁。
    """

    __tablename__ = "audit_chain_heads"

    last_seq: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default="0",
        comment="链上最后一条记录的序号（从 1 开始）",
    )
    last_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="0" * 64,
        server_default="0" * 64,
        comment="链上最后一条记录的哈希。创世值全 0",
    )

    __table_args__ = (
        # 一个租户一条链头
        UniqueConstraint("tenant_id", name="uq_audit_chain_heads_tenant"),
    )


class AuditLog(Base, UUIDPrimaryKeyMixin):
    """审计日志：只追加，永不修改或删除（除按保留期批量清理）。

    **注意没有 TimestampMixin 的 updated_at**：审计记录不存在"更新"语义。
    只有 created_at，且由数据库生成以保证单调。

    **也不直接继承 TenantScopedMixin**：虽然需要 tenant_id，
    但审计允许 tenant_id 为 NULL —— 登录失败、跨租户探测这类事件发生时，
    我们可能还不知道（或不该信任）租户。这类记录只有平台超管可见，
    且不参与租户的哈希链（它们不属于任何租户的链）。
    """

    __tablename__ = "audit_logs"

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tenants.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="租户 ID。NULL 表示全局事件（登录失败/跨租户探测），仅平台超管可见",
    )

    # ---- 链 ----
    seq: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="该租户审计链内的序号，从 1 开始严格递增",
    )
    prev_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="前一条记录的 hash（链式引用）",
    )
    record_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="本条记录的 hash = SHA256(prev_hash + 规范化内容)",
    )

    # ---- 谁 ----
    # 不设外键：用户可能被删除，但审计记录必须保留（这正是审计的意义）。
    # 用户名冗余存储：用户改名或被删后，仍能看出"当时的操作者是谁"。
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        comment="操作者用户 ID（无 FK：用户删除后审计仍需保留）",
    )
    actor_email: Mapped[str | None] = mapped_column(
        String(320),
        nullable=True,
        comment="操作者邮箱快照（用户改名/删除后仍可追溯）",
    )
    actor_tenant_role: Mapped[TenantRole | None] = mapped_column(
        String(32),
        nullable=True,
        comment="操作时的租户角色快照（角色后来变了也不影响历史判定）",
    )
    actor_platform_role: Mapped[PlatformRole | None] = mapped_column(
        String(32),
        nullable=True,
        comment="操作时的平台角色快照",
    )

    # ---- 何时 / 从哪 ----
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        nullable=False,
        comment="事件时间（数据库侧生成）",
    )
    request_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="关联的请求 ID，可回查完整链路日志",
    )
    ip_address: Mapped[str | None] = mapped_column(
        INET,
        nullable=True,
        comment="客户端 IP（INET 类型支持网段查询）",
    )
    user_agent: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="客户端 UA",
    )

    # ---- 做了什么 ----
    action: Mapped[AuditAction] = mapped_column(
        String(64),
        nullable=False,
        comment="动作类型，见 AuditAction",
    )
    outcome: Mapped[AuditOutcome] = mapped_column(
        String(32),
        nullable=False,
        default=AuditOutcome.SUCCESS,
        server_default=AuditOutcome.SUCCESS.value,
        comment="结果：success/denied/failed",
    )

    # ---- 对什么做的 ----
    target_type: Mapped[AuditTargetType | None] = mapped_column(
        String(32),
        nullable=True,
        comment="目标对象类型",
    )
    target_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        comment="目标对象 ID（无 FK：目标可能已被删除）",
    )
    target_name: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="目标名称快照（对象删除后仍可读）",
    )

    # ---- 详情 ----
    # **绝不放用户文档内容、密钥、完整 prompt**。
    # 只放 ID、计数、状态变化这类可公开的结构化信息。
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="结构化详情（禁止含文档正文/密钥/PII）",
    )
    # 被拒/失败时的原因（错误码），便于统计"哪类越权最多"
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="失败或拒绝时的错误码",
    )

    __table_args__ = (
        # 链内序号唯一，保证链完整不跳号
        UniqueConstraint("tenant_id", "seq", name="uq_audit_logs_tenant_seq"),
        # 主查询：某租户最近的操作
        Index("ix_audit_logs_tenant_created", "tenant_id", "created_at"),
        # 按动作筛选（"最近所有越权尝试"）
        Index("ix_audit_logs_tenant_action", "tenant_id", "action", "created_at"),
        # 按对象查（"这个文档被谁看过"）
        Index("ix_audit_logs_target", "target_type", "target_id", "created_at"),
        # 安全事件告警：只看被拒/跨租户这类
        Index(
            "ix_audit_logs_security",
            "tenant_id",
            "created_at",
            postgresql_where=text("outcome != 'success'"),
        ),
        # 按操作者查（"这个用户最近做了什么"）
        Index("ix_audit_logs_actor", "actor_user_id", "created_at"),
    )
