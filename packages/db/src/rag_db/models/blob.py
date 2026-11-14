"""物理文件（blob）模型：内容寻址存储。

为什么单独一层而不是把文件属性挂在 document_versions 上：
- 秒传需要"按内容找是否已存在"，这是全局（租户内）查询，不是版本级；
- 同一个文件上传到两个知识库时，物理数据只存一份；
- 版本回滚只需要改指向，不必复制数据。

内容寻址（content-addressed）：主键语义上由 `(tenant_id, content_hash)` 决定，
但技术上是 UUID 主键 + 该组合的唯一索引——这样外键更好写，
且未来若要"跨租户共享同一物理文件"（合规允许时），只需放宽唯一约束。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from rag_contracts.enums.documents import BlobStatus
from rag_db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Blob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """物理文件记录。

    这里**只存元数据**，字节留在对象存储（LocalStorage / S3）。
    数据库存大字段会让备份变大几十倍、且 PG 的 TOAST 机制在大对象上
    读写效率远不如专门的对象存储。
    """

    __tablename__ = "blobs"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        comment="租户 ID",
    )
    content_hash: Mapped[str] = mapped_column(
        # SHA-256 十六进制固定 64 字符。用定长 CHAR 还是 VARCHAR 都行，
        # 这里用 VARCHAR(64) 便于后续换算法（如 BLAKE3 长度不同）
        String(64),
        nullable=False,
        comment="内容 SHA-256（小写十六进制），秒传与完整性校验的依据",
    )
    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="字节数（BigInteger：支持 >2GB 文件）",
    )
    # 对象存储里的键。设计成不含原始文件名：
    # 原始文件名可能含用户隐私或恶意字符，且重命名不该影响存储路径。
    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="对象存储键（不含原始文件名，按租户/hash 分片）",
    )
    storage_backend: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="后端标识（local/s3），回滚或迁移后仍知道去哪儿取",
    )
    content_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="MIME 类型（以魔数校验结果为准，不信客户端声明）",
    )
    status: Mapped[BlobStatus] = mapped_column(
        String(32),
        nullable=False,
        default=BlobStatus.PENDING,
        server_default=BlobStatus.PENDING.value,
        index=True,
        comment="状态：pending/complete/missing/deleted",
    )
    # 引用计数：多个文档版本指向同一 blob 时，删除需先减到 0。
    # 用计数而不是每次 count(*) 查询，是因为删除路径要高并发安全；
    # 定期对账任务会校正这个值（防止程序 bug 导致永久泄漏）。
    ref_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="引用计数（被多少文档版本引用），0 时可物理删除",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="失败原因（status=missing 时说明对账发现的情况）",
    )
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后一次完整性校验通过的时间",
    )

    __table_args__ = (
        # 秒传的核心：同租户下同一内容只有一条记录。
        # 用唯一索引而不是"先查再插"——并发上传同一文件时有竞态窗口。
        UniqueConstraint("tenant_id", "content_hash", name="uq_blobs_tenant_hash"),
        # 按存储键查（下载/删除时需要定位对象）
        Index("ix_blobs_storage_key", "storage_key"),
        # 对账任务扫孤儿：status=complete 但 ref_count=0 且创建已久
        Index("ix_blobs_status_ref", "status", "ref_count"),
    )

    def __repr__(self) -> str:
        return f"<Blob {self.content_hash[:12]} ({self.size_bytes}B, refs={self.ref_count})>"
