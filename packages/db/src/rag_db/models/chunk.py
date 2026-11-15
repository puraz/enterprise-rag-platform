"""分块（chunk）模型：检索的最小单元。

## 设计要点

### 1. 单表 + HNSW，不做租户分区

分区（HASH by tenant_id）能让每个分区有独立 HNSW，过滤租户后召回准确。
但代价是：分区数固定后增加分区要 ATTACH/重建，Alembic 迁移复杂，
A10 的零停机重建要对每个分区各做一遍。

单表的隐患是"向量索引先扫描、后过滤 tenant_id"：
若某租户只占总量 1%，要扫 1000 个候选才能凑够 10 条结果。

缓解手段：PostgreSQL 侧启用 `hnsw.iterative_scan = relaxed_order`
（pgvector 0.8.0+，本项目服务端 0.8.7 已支持），
让查询在结果不足时自动扩大扫描，而不是直接返回空。

**适用边界**：单租户 chunk 数 ≲ 50 万、租户数 ≲ 50。
超过这个量级应改为 HASH 分区——A10 压测时需要验证这一点，
用真实数据量测"过滤后召回率"而不是凭感觉。

### 2. 位置信息必须精确到可高亮

`page_no` / `bbox` / `char_start` / `char_end` / `section_path` 全部保留。
前端"点击引用跳到原文高亮"依赖这些字段——只存文本无法定位。

### 3. 父子关系支持 small-to-big

`parent_chunk_id` 自引用：父子文档策略下，小块用于检索匹配精度，
命中后回填父块拿到完整上下文。parent_child 策略必需。

### 4. 向量列与索引版本

`embedding` 列是当前生效版本的向量。A10 做模型升级或索引重建时，
用 expand-contract：新增 `embedding_v2` 列 → 双写 → 回填 → 切索引 → 删旧列。
**不要 ALTER COLUMN TYPE**——那会锁表并重写全表。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_contracts.enums.retrieval import ChunkType
from rag_db.base import (
    Base,
    TenantScopedMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from rag_db.models.document import Document, DocumentVersion

# 向量维度。
#
# 为什么是 1024：
# - bge-m3 / bge-large-zh-v1.5 等主流中文本地模型都是 1024 维；
# - 本机无 GPU，embedding 优先本地方案（docs/PROMPT.md 的成本要求）；
# - 1024 维 × 4 字节 = 4KB/chunk，百万 chunk 约 4GB，单机可承受。
#
# 改维度的正确方式（expand-contract）：
#   1. 新增 embedding_v2 vector(1536) 列
#   2. 双写：新 chunk 两列都写，旧 chunk 后台回填 v2
#   3. 建 v2 的 HNSW 索引，切查询到 v2
#   4. 删旧列与旧索引
# HNSW 索引的维度必须与列一致，且**不能在不重建索引的情况下改维度**。
EMBEDDING_DIM = 1024


class Chunk(Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin):
    """文本分块。

    与 document_version 是多对一：一份文档版本切出 N 个 chunk。
    检索时按 `tenant_id + knowledge_base_id` 过滤，按向量距离排序。
    """

    __tablename__ = "chunks"

    # ---- 归属链 ----
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属知识库（冗余：检索按库过滤时避免 JOIN）",
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属文档（冗余：引用展示与删除清理需要）",
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属文档版本。只有当前版本的 chunk 参与检索",
    )

    # ---- 次序与结构 ----
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="在所属版本内的序号（从 0 开始），用于原文顺序还原",
    )
    chunk_type: Mapped[ChunkType] = mapped_column(
        String(32),
        nullable=False,
        default=ChunkType.TEXT,
        server_default=ChunkType.TEXT.value,
        comment="内容类型：text/title/table/code/formula 等",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="chunk 文本内容",
    )
    # 送去 embedding 的实际文本。与 content 可能不同：
    # 结构化策略会在前面加章节路径前缀（"第三章 > 3.2 报销流程 > ..."），
    # 提升向量对上下文的感知；但 content 保持原样供引用展示与高亮。
    embedding_input: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="实际送去向量化的文本（含上下文前缀），NULL 表示与 content 相同",
    )

    # ---- 父子关系（small-to-big）----
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        # SET NULL：父块被删时子块保留（子块自身仍可检索），
        # 只在回填时退化为使用自身内容
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
        comment="父块 ID（parent_child 策略用），命中子块后可回填父块上下文",
    )
    chunk_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="层级：0=叶子块，>0=父块。检索默认只用 level=0",
    )

    # ---- 位置信息（引用高亮必需）----
    page_no: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="页码（从 1 开始），非分页格式为 NULL",
    )
    char_start: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="在原文档纯文本中的起始字符偏移（含），供高亮定位",
    )
    char_end: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="在原文档纯文本中的结束字符偏移（不含）",
    )
    bbox: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="版面坐标 {x0,y0,x1,y1}（PDF 用），供前端在原图上框选",
    )
    section_path: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
        comment="章节路径（如 ['第三章','3.2 报销流程']），供面包屑与引用",
    )

    # ---- 检索用统计 ----
    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="token 数（上下文裁剪与成本估算用）",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="chunk 内容的 SHA-256，用于去重与缓存 key",
    )

    # ---- 向量 ----
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=True,
        comment=f"向量（维度 {EMBEDDING_DIM}）。NULL 表示尚未向量化",
    )
    embedding_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="生成该向量的模型标识（换模型后可识别哪些需要重算）",
    )

    # ---- 元数据 ----
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="扩展元数据（表格结构、代码语言等）",
    )
    is_current: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        server_default="true",
        comment="是否属于文档当前版本。旧版本的 chunk 置 false，检索时过滤",
    )

    document: Mapped[Document] = relationship()
    document_version: Mapped[DocumentVersion] = relationship()

    __table_args__ = (
        # 同一版本内序号唯一：重复的 chunk_index 说明切分逻辑有 bug
        # （会产生重复 chunk，检索结果里出现两条一样的引用）
        UniqueConstraint("document_version_id", "chunk_index", name="uq_chunks_version_index"),
        # ---- 检索主查询的索引 ----
        # 命中路径：WHERE tenant_id=? AND knowledge_base_id=? AND is_current=true
        # 按向量排序。tenant_id 必须在最左，匹配 RLS 的强制过滤。
        Index(
            "ix_chunks_tenant_kb_current",
            "tenant_id",
            "knowledge_base_id",
            "is_current",
        ),
        # 按版本统计/清理：WHERE document_version_id=?
        Index("ix_chunks_version_index", "document_version_id", "chunk_index"),
        # 按文档清理：软删文档后异步删 chunk
        Index("ix_chunks_document", "document_id"),
        # 父块回填：WHERE parent_chunk_id = ?
        Index("ix_chunks_parent", "parent_chunk_id"),
        # 内容去重
        Index("ix_chunks_content_hash", "tenant_id", "content_hash"),
        # 未向量化的 chunk（补偿任务扫描：embedding IS NULL）
        Index(
            "ix_chunks_pending_embedding",
            "tenant_id",
            postgresql_where=text("embedding IS NULL"),
        ),
        # 位置信息必须成对出现，否则高亮定位会算出错误的区间
        CheckConstraint(
            "(char_start IS NULL AND char_end IS NULL) "
            "OR (char_start IS NOT NULL AND char_end IS NOT NULL AND char_end > char_start)",
            name="ck_chunks_char_range_consistent",
        ),
        CheckConstraint("chunk_index >= 0", name="ck_chunks_index_non_negative"),
        CheckConstraint("chunk_level >= 0", name="ck_chunks_level_non_negative"),
        CheckConstraint(
            "token_count IS NULL OR token_count >= 0", name="ck_chunks_token_non_negative"
        ),
        # ---- HNSW 向量索引 ----
        #
        # **必须在模型里声明，不能用裸 SQL 建。**
        # 裸 SQL 建的索引不在 SQLAlchemy metadata 里，Alembic autogenerate
        # 会认为它是"数据库里多余的索引"并生成 DROP —— 下次
        # `make migrate-new` 就会把索引删掉，且没有任何报错提示。
        # （实际踩到的坑：A3-3 用手写 SQL 建索引，A3-4 生成迁移时出现了
        #   `Detected removed index 'ix_chunks_embedding_hnsw'`。）
        #
        # SQLAlchemy 完全能表达 HNSW，编译结果与手写 SQL 等价：
        #   CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)
        #   WITH (m=16, ef_construction=64) WHERE embedding IS NOT NULL AND ...
        #
        # 参数依据：
        # - m=16：每节点最大连接数，官方默认。内存开销约为向量总量 20%；
        #   调到 32 会翻倍到 40%，百万 chunk（约 4GB 向量）时本机扛不住。
        # - ef_construction=64：构建时候选列表，官方默认，越大质量越好但更慢。
        # - vector_cosine_ops：与 bge 系列训练目标一致。度量选错不报错，
        #   只会让召回悄悄变差 —— 最难发现的检索问题之一。
        # - WHERE 部分索引：未向量化与旧版本的 chunk 不占索引空间。
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_where=text("embedding IS NOT NULL AND is_current = true"),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Chunk v={self.document_version_id} #{self.chunk_index} ({len(self.content)} chars)>"
        )


class IndexVersion(Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin):
    """向量索引版本（A10 零停机重建用）。

    为什么需要这张表而不是直接建/删索引：
    "重建索引"在生产里不能停服。正确做法是：
      1. 建新版本（BUILDING），写入新 chunk 时**双写**两个版本；
      2. 后台回填旧 chunk 到新版本；
      3. 校验条数与抽样查询一致后，把 ACTIVE 指向新版本（秒级切换）；
      4. 旧版本标 RETIRED，观察一段时间后回收。

    这张表记录每个版本的元信息，`alias` 概念由 status=ACTIVE 表达
    （同一 (tenant_id, knowledge_base_id) 最多一条 ACTIVE）。

    切换只是一条 UPDATE，因此在途请求不受影响：
    读操作按切换前的 ACTIVE 版本完成，写操作双写两个版本。
    """

    __tablename__ = "index_versions"

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属知识库",
    )
    version_tag: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="版本标识（如 v1 / 20261115-embed-bge-m3），人可读",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="building",
        server_default="building",
        comment="building/ready/active/retired/failed",
    )
    embedding_model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="该版本使用的 embedding 模型（切模型必然换版本）",
    )
    embedding_dim: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="向量维度，用于校验与 chunks.embedding 一致",
    )
    index_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="hnsw",
        server_default="hnsw",
        comment="hnsw/ivfflat/none",
    )
    distance_metric: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="cosine",
        server_default="cosine",
        comment="cosine/l2/inner_product",
    )
    # HNSW 参数记录在这里：重建时用同样参数才能保证查询特性一致
    index_params: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="索引参数快照（m/ef_construction 等）",
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="已写入该版本的 chunk 数（回填进度用）",
    )
    expected_chunk_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="预期 chunk 总数，回填完成判定依据",
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="切换为 ACTIVE 的时间",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="构建失败原因",
    )

    __table_args__ = (
        UniqueConstraint("knowledge_base_id", "version_tag", name="uq_index_versions_kb_tag"),
        # 同一知识库最多一个 ACTIVE —— 这是 "alias" 的实现
        Index(
            "uq_index_versions_active",
            "knowledge_base_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
        Index("ix_index_versions_kb_status", "knowledge_base_id", "status"),
    )

    def __repr__(self) -> str:
        return f"<IndexVersion {self.version_tag} ({self.status})>"
