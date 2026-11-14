"""文档与文档版本模型：知识库的核心。

## 三层结构（为什么不是一个表）

    blobs              物理字节流，按 (tenant_id, content_hash) 唯一 → 秒传
      ▲
      │ 引用
    document_versions  某文档的某个版本，快照了处理配置 → 可回滚、可审计
      ▲
      │ 1:N
    documents          业务上的"一份文件"，有 ACL、有名称、有当前版本指针

三层解耦解决三个不同问题：
- **秒传**：同一字节流只存一份（blobs 唯一约束）；
- **版本**：同一文档的历史可追溯（versions 1:N）；
- **权限**：ACL 挂在 document 上，与物理存储无关。

## 为什么文档只属于一个知识库

多归属（M:N）在四个地方制造歧义：
1. "从库 A 删除"是删文档还是删关联？另一个库还能用吗？
2. 库 A 是 tenant 可见、库 B 是 private，ACL 取交集还是并集？
3. chunk 的 `kb_id` 填哪个？混合检索按库过滤时以谁为准？
4. 重建索引是针对库的，多归属文档会被两个库各建一次吗？

企业里"一份制度要出现在两个库"的真实需求，正确解法是**上传两次**：
两个库的切分策略、密级、ACL 可能都不同，独立处理才是对的。
存储成本由 blob 去重吸收（同一文件只存一份字节）。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    Boolean,
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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_contracts.enums.documents import (
    ConfidentialityLevel,
    DocumentFormat,
    DocumentStatus,
    DocumentVersionStatus,
    SourceType,
)
from rag_db.base import (
    Base,
    OptimisticLockMixin,
    TenantScopedMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from rag_db.models.blob import Blob
    from rag_db.models.knowledge import KnowledgeBase


class Document(Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin, OptimisticLockMixin):
    """逻辑文档：用户眼中的"一份文件"。

    **不含 content_hash**：hash 属于版本级别（不同版本内容当然不同）。
    秒传判断通过 `document_versions.blob_id → blobs.content_hash` 间接完成。
    """

    __tablename__ = "documents"

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属知识库（单归属，见模块 docstring）",
    )
    # 用户可见的文件名。与 blob.storage_key 无关——
    # 重命名文档不应影响存储，存储里也不该出现用户文件名（隐私与安全）。
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="文档标题（用户可见，可重命名）",
    )
    # 原始文件名单独存：审计需要知道"用户上传时叫什么"
    original_filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="上传时的原始文件名（审计与下载时的默认名）",
    )
    format: Mapped[DocumentFormat] = mapped_column(
        String(32),
        nullable=False,
        comment="文档格式，决定用哪个 parser",
    )
    source_type: Mapped[SourceType] = mapped_column(
        String(32),
        nullable=False,
        default=SourceType.UPLOAD,
        server_default=SourceType.UPLOAD.value,
        comment="来源类型：upload/url/connector/api",
    )
    # 文档级密级：可以比库级更严，不能更松（service 层校验）
    confidentiality: Mapped[ConfidentialityLevel] = mapped_column(
        String(32),
        nullable=False,
        default=ConfidentialityLevel.INTERNAL,
        server_default=ConfidentialityLevel.INTERNAL.value,
        comment="文档密级，不得低于所属库的密级",
    )
    status: Mapped[DocumentStatus] = mapped_column(
        String(32),
        nullable=False,
        default=DocumentStatus.PENDING,
        server_default=DocumentStatus.PENDING.value,
        comment="处理状态机，见 DocumentStatus 的流转说明",
    )
    # 指向当前生效版本。允许 NULL：刚创建、还没有任何版本时
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        # 用 SET NULL 而不是 CASCADE：删版本不该删文档
        ForeignKey("document_versions.id", ondelete="SET NULL", use_alter=True),
        nullable=True,
        comment="当前生效版本 ID（NULL 表示尚无版本）",
    )
    current_version_no: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="当前版本序号（0 表示尚无版本），冗余字段避免每次 JOIN",
    )
    latest_version_no: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="已创建的版本总数，新版本号 = latest_version_no + 1",
    )

    # 失败信息：允许重试而不必重新上传
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="失败错误码（rag_contracts.errors.ErrorCode 的值）",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="失败详情（仅内部可见，不含用户文档内容）",
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="已重试次数（防止无限重试）",
    )

    # 统计信息（列表页展示）
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="当前版本的 chunk 数",
    )
    page_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="页数（解析后填充，非分页格式为 NULL）",
    )
    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="当前版本总 token 数（成本估算用）",
    )
    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default="0",
        comment="当前版本文件大小（冗余，列表页避免 JOIN blob）",
    )

    created_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        comment="上传者",
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="软删除时间（NULL 表示未删除）。索引与 chunk 由异步任务清理",
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")
    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        # 显式指定外键：document 与 version 之间有两个关联
        # （version.document_id 和 document.current_version_id）
        foreign_keys="DocumentVersion.document_id",
    )

    __table_args__ = (
        # 同库内标题唯一：同名的两份文档会让用户无法分辨该点哪个。
        # 部分索引：软删除的文档不占用名字，允许重建同名文档。
        Index(
            "uq_documents_kb_title_active",
            "knowledge_base_id",
            "title",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
        # 列表页主查询：WHERE tenant_id=? AND knowledge_base_id=? AND deleted_at IS NULL
        # ORDER BY updated_at DESC
        Index(
            "ix_documents_tenant_kb_updated",
            "tenant_id",
            "knowledge_base_id",
            "updated_at",
        ),
        # 状态看板：WHERE tenant_id=? AND status=? （查"有多少文档卡在处理中"）
        Index("ix_documents_tenant_status", "tenant_id", "status"),
        # 失败重试扫描：WHERE status='failed' AND retry_count < N
        Index("ix_documents_status_retry", "status", "retry_count"),
        # 检索过滤：只有 ready 且未删除的文档参与检索
        Index(
            "ix_documents_tenant_ready",
            "tenant_id",
            "status",
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    def __repr__(self) -> str:
        return f"<Document {self.title} v{self.current_version_no} ({self.status})>"


class DocumentVersion(Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin):
    """文档的某个版本。

    每次内容变更（上传新文件、编辑）都新建一条，**不原地覆盖**。
    这样回滚只是把 `documents.current_version_id` 指回去，
    不需要重新解析、重新切分、重新向量化。

    ## 为什么快照处理配置

    `chunking_strategy` / `chunking_config` / `parser_name` / `embedding_model`
    都存到版本上，而不是只读知识库的当前配置。

    不存的话：库的默认切分策略从"递归"改成"语义"后，回滚到 v1 时
    重新生成的 chunk 与当初完全不同——**"回滚"名不副实**。
    而且出了检索质量问题，无法回答"这批 chunk 当时是用什么配置生成的"。
    """

    __tablename__ = "document_versions"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属文档",
    )
    version_no: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="版本序号，从 1 递增。唯一约束见 __table_args__",
    )
    blob_id: Mapped[uuid.UUID] = mapped_column(
        # RESTRICT：blob 被引用时不能删。删除走"引用计数减到 0"的路径
        ForeignKey("blobs.id", ondelete="RESTRICT"),
        nullable=False,
        comment="物理文件 ID",
    )
    status: Mapped[DocumentVersionStatus] = mapped_column(
        String(32),
        nullable=False,
        default=DocumentVersionStatus.PROCESSING,
        server_default=DocumentVersionStatus.PROCESSING.value,
        comment="版本状态：processing/ready/failed/superseded",
    )

    # ---- 处理配置快照（见模块 docstring）----
    parser_name: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="解析器实现名（如 pdf_pymupdf），失败时可能为 NULL",
    )
    chunking_strategy: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="recursive",
        server_default="recursive",
        comment="切分策略名（快照，不随库配置变化）",
    )
    chunking_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="切分参数快照（size/overlap/separators 等）",
    )
    embedding_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="向量化模型标识（换模型后旧版本仍可追溯）",
    )
    embedding_dim: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="向量维度。与 embedding_model 绑定，用于校验索引一致性",
    )

    # ---- 处理结果统计 ----
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="本版本生成的 chunk 数",
    )
    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="本版本 token 总数",
    )
    page_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="页数",
    )

    # 解析产物的结构化结果（章节树、表格位置等），供引用定位使用。
    # 用 JSONB 而非单独表：结构随格式变化大，且只做整体读取。
    parse_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="解析元数据（章节树/页眉页脚识别结果等），供引用定位",
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
    created_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        comment="产生此版本的操作者",
    )
    # 该版本被取代/归档的时间（不删除，保留审计）
    superseded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="被新版本取代的时间（NULL 表示未被取代）",
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="是否为当前生效版本（冗余标记，便于按版本过滤 chunk）",
    )

    document: Mapped[Document] = relationship(
        back_populates="versions",
        foreign_keys=[document_id],
    )
    blob: Mapped[Blob] = relationship()

    __table_args__ = (
        # 同一文档的版本号唯一。并发上传新版本时靠这个约束防重复。
        UniqueConstraint("document_id", "version_no", name="uq_document_versions_doc_no"),
        # 版本历史列表：WHERE document_id=? ORDER BY version_no DESC
        Index("ix_document_versions_doc_no", "document_id", "version_no"),
        # 处理中的版本（worker 扫描补偿任务用）
        Index("ix_document_versions_status", "status"),
        # 找当前版本：每个文档最多一条 is_current=true
        Index(
            "uq_document_versions_current",
            "document_id",
            unique=True,
            postgresql_where=text("is_current = true"),
        ),
        # chunk 表按 (document_version_id) 过滤，需要能快速按版本统计
        Index("ix_document_versions_blob", "blob_id"),
    )

    def __repr__(self) -> str:
        return f"<DocumentVersion doc={self.document_id} v{self.version_no} ({self.status})>"
