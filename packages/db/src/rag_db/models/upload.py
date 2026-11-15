"""分片上传会话模型。

## 为什么会话元数据在 PG、分片进度在 Redis

会话是**业务数据**：它决定了"这个上传到哪一步了、谁发起的、最终要落成哪个文档"。
丢了它，用户上传的 GB 级分片全部作废。所以必须持久化。

分片进度是**运行时状态**：高频写入（每个分片一次）、可重建
（从对象存储列出实际存在的分片即可恢复）。放 Redis 位图，快且能自动过期。

**权威来源是对象存储**，Redis 只是缓存。会话恢复时以对象存储的
分片列表为准，Redis 里的记录用于快速判断"这个分片是不是已经传过了"。
这个设计让 Redis 丢失不会导致数据错误，只影响性能。

## 断点续传的判定

客户端请求"这个会话已上传哪些分片"时：
1. 先查 Redis 的集合（快）；
2. 若 Redis 无记录或要求强一致，列出对象存储里的分片对象（准）；
3. 返回缺失的分片号列表，客户端只补传这些。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from rag_contracts.enums.uploads import UploadSessionStatus
from rag_db.base import (
    Base,
    OptimisticLockMixin,
    TenantScopedMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class UploadSession(
    Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin, OptimisticLockMixin
):
    """一次分片上传会话。

    生命周期：initiate → upload parts → complete/abort。
    会话本身不存分片数据，只存元数据；分片在对象存储里。
    """

    __tablename__ = "upload_sessions"

    # 发起人。删用户时 RESTRICT：有进行中的上传说明这个用户还活跃，
    # 而且删除用户不该静默丢弃他的上传会话（会话本身也需审计）
    created_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        comment="发起上传的用户",
    )
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=True,
        comment="目标知识库（initiate 时可能还未确定，complete 时必须非空）",
    )

    status: Mapped[UploadSessionStatus] = mapped_column(
        String(32),
        nullable=False,
        default=UploadSessionStatus.INITIATED,
        server_default=UploadSessionStatus.INITIATED.value,
        comment="会话状态机，见 UploadSessionStatus",
    )

    # ---- 文件元信息（客户端在 initiate 时声明，complete 时校验）----
    filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="原始文件名（先净化再存，防止路径穿越）",
    )
    total_size: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="声明的总字节数",
    )
    content_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="声明的 MIME 类型（complete 时以魔数校验结果为准）",
    )
    # 客户端预先算好的整体 hash：用于秒传判断与完成后校验
    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="客户端声明的整体 SHA-256；为空则允许（服务端合并后计算）",
    )

    # ---- 分片规划（initiate 时确定，之后不可变）----
    part_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="每片字节数（除最后一片）。S3 要求 ≥5MB，本地后端可更小",
    )
    total_parts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="分片总数 = ceil(total_size / part_size)",
    )
    uploaded_parts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="已完成分片数（缓存值，权威来源是对象存储）",
    )

    # ---- 后端相关标识 ----
    storage_backend: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="存储后端（local/s3），complete 时按这个选实现",
    )
    # S3 multipart 的 UploadId；本地后端为自定义 ID
    storage_upload_id: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="对象存储的 multipart upload id",
    )
    # 分片对象的键前缀，如 "uploads/{tenant}/{session}/"
    storage_key_prefix: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="分片对象键前缀",
    )

    # ---- 结果 ----
    blob_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("blobs.id", ondelete="SET NULL"),
        nullable=True,
        comment="完成后指向的 blob（失败/中止时为空）",
    )
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

    # ---- 生命周期时间 ----
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="会话过期时间。过期后分片由清理任务回收",
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="完成时间",
    )
    aborted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="中止时间",
    )

    __table_args__ = (
        # 清理任务扫描：WHERE status IN (...) AND expires_at < now()
        Index("ix_upload_sessions_status_expires", "status", "expires_at"),
        # 用户查看自己的上传列表
        Index("ix_upload_sessions_tenant_user", "tenant_id", "created_by"),
        # 按 hash 查是否已有同名上传（秒传前置判断）
        Index(
            "ix_upload_sessions_tenant_hash",
            "tenant_id",
            "content_hash",
            postgresql_where=text("content_hash IS NOT NULL"),
        ),
        CheckConstraint("total_size > 0", name="ck_upload_sessions_size_positive"),
        CheckConstraint("part_size > 0", name="ck_upload_sessions_part_size_positive"),
        CheckConstraint("total_parts > 0", name="ck_upload_sessions_parts_positive"),
        CheckConstraint(
            "uploaded_parts >= 0 AND uploaded_parts <= total_parts",
            name="ck_upload_sessions_uploaded_in_range",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<UploadSession {self.filename} "
            f"{self.uploaded_parts}/{self.total_parts} ({self.status})>"
        )
