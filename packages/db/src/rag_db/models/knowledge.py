"""知识库模型。

知识库是文档的**归属容器**，也是权限继承的起点：
库级可见性决定谁能看到这个库里的文档，文档可以在库的基础上再收紧。
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_contracts.enums.documents import (
    ConfidentialityLevel,
    KnowledgeBaseStatus,
    KnowledgeBaseVisibility,
)
from rag_db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from rag_db.models.document import Document


class KnowledgeBase(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """知识库。

    与文档是 1:N：**一个文档只属于一个知识库**。
    理由见 `document.py` 的模块 docstring——多归属会让 ACL、chunk 归属、
    索引重建全部产生歧义。
    """

    __tablename__ = "knowledge_bases"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        comment="租户 ID",
    )
    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="知识库名称（租户内唯一）",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="描述",
    )
    status: Mapped[KnowledgeBaseStatus] = mapped_column(
        String(32),
        nullable=False,
        default=KnowledgeBaseStatus.ACTIVE,
        server_default=KnowledgeBaseStatus.ACTIVE.value,
        comment="状态：active/archived/deleted",
    )
    visibility: Mapped[KnowledgeBaseVisibility] = mapped_column(
        String(32),
        nullable=False,
        default=KnowledgeBaseVisibility.TENANT,
        server_default=KnowledgeBaseVisibility.TENANT.value,
        comment="可见范围：private/tenant/department",
    )
    confidentiality: Mapped[ConfidentialityLevel] = mapped_column(
        String(32),
        nullable=False,
        default=ConfidentialityLevel.INTERNAL,
        server_default=ConfidentialityLevel.INTERNAL.value,
        comment="库级密级，文档密级不得低于此级别",
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        comment="创建者。RESTRICT：有库时不能删用户，必须先转移或清理",
    )

    # 默认切分策略：新建文档若未指定则继承库的设置。
    # 允许文档级覆盖（不同文档类型适合不同策略，例如表格 vs 正文）。
    default_chunking_strategy: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="recursive",
        server_default="recursive",
        comment="默认切分策略名，对应 chunkers 注册表里的实现",
    )
    default_chunk_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=800,
        server_default="800",
        comment="默认 chunk 目标长度（字符数，按策略解释）",
    )
    default_chunk_overlap: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=100,
        server_default="100",
        comment="默认 chunk 重叠长度",
    )

    document_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="文档数缓存（列表页展示用，定期对账修正）",
    )

    documents: Mapped[list[Document]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        # 租户内库名唯一（同名库会让用户选错）
        UniqueConstraint("tenant_id", "name", name="uq_knowledge_bases_tenant_name"),
        # 列表页主查询：WHERE tenant_id = ? AND status = 'active' ORDER BY updated_at
        Index("ix_knowledge_bases_tenant_status", "tenant_id", "status"),
        # 切分参数必须合理，脏配置会导致 chunk 无限循环或全空
        CheckConstraint("default_chunk_size > 0", name="ck_knowledge_bases_chunk_size_positive"),
        CheckConstraint(
            "default_chunk_overlap >= 0 AND default_chunk_overlap < default_chunk_size",
            name="ck_knowledge_bases_overlap_less_than_size",
        ),
    )

    def __repr__(self) -> str:
        return f"<KnowledgeBase {self.name} ({self.status})>"
