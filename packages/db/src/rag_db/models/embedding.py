"""向量表：一个 chunk 可以有多个版本的向量。

## 为什么向量不能放在 chunks 表上（A10 的关键设计修正）

最初的设计是 `chunks.embedding` 单列 + `chunks.index_version_id`。
用这个设计做"零停机重建索引"时，演练（`scripts/drill_index_rebuild.py`）
暴露了本质缺陷：

    回填时把 chunk 的 index_version_id 从 V1 改成 V2
      → 这个 chunk 立刻从 V1 的查询结果里消失
      → 回填完成时 V1 一个 chunk 都没有了
      → "零停机"不成立，且旧版本无法回滚（向量已被覆盖）

根因：**一行只能存一个向量**，所以"两份向量并存"在物理上做不到。

## 旁表的解法

    chunks              内容（与向量无关，一份）
      └── chunk_embeddings   向量（每个索引版本一份）

    回填 V2 时：为每个 chunk **新增一行** chunk_embeddings，
    旧行（V1）不动 → V1 的查询完全不受影响 → 真零停机
    切换：ACTIVE 指向 V2（一条 UPDATE）
    回滚：ACTIVE 指回 V1（V1 的行还在，秒级）
    回收：确认新版本稳定后，DELETE FROM chunk_embeddings WHERE index_version_id = V1

代价：
- 回填期间向量存储翻倍（100 万 chunk × 4KB × 2 = 8GB，可接受）；
- 检索多一次 JOIN（在 HNSW 索引命中后 JOIN，成本很低）。

## 为什么 HNSW 索引建在旁表上

向量在旁表上，索引自然也在旁表上：
`(embedding vector_cosine_ops)` 的部分索引，WHERE index_version_id IS NOT NULL。

**注意**：HNSW 索引是"全局一个索引"，按 index_version_id 过滤时
pgvector 会用 `iterative_scan` 扩大扫描（见 chunk 仓储的说明）。

## 与 `chunks.embedding` 的关系

`chunks.embedding` 这一列**保留但不再使用**——A10 之前的数据在它上面。
迁移策略（expand-contract）：
1. 建旁表；
2. 回填：把每行的 chunks.embedding 复制到 chunk_embeddings（归入默认版本）；
3. 切代码到旁表；
4. **下个版本**再删 chunks.embedding（本步骤不删，保证可回滚）。

这是标准的 expand-contract：先加、再双写、后删。删列留到确认稳定之后。
"""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from rag_db.base import Base, TenantScopedMixin, TimestampMixin, UUIDPrimaryKeyMixin
from rag_db.models.chunk import EMBEDDING_DIM


class ChunkEmbedding(Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin):
    """一个 chunk 在某个索引版本下的向量。

    生命周期：
    - 文档向量化时，为当前 ACTIVE 版本创建一行；
    - 重建索引时，为新版本再创建一行（旧行保留，供回滚）；
    - 旧版本回收时删除。

    **不是 1:1**：一个 chunk 通常有 1 行（稳定态），
    重建期间有 2 行（新旧并存）。
    """

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属 chunk。CASCADE：删 chunk 时向量一起删（它们没有独立意义）",
    )
    index_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("index_versions.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属索引版本。CASCADE：回收版本时清掉它的全部向量",
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=False,
        comment=f"向量（维度 {EMBEDDING_DIM}）",
    )
    embedding_model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="生成此向量的模型（换模型必然换版本，这里是双保险）",
    )
    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="该 chunk 的 token 数（成本估算；从 chunk 冗余过来便于按版本聚合）",
    )

    __table_args__ = (
        # 一个 chunk 在一个版本下只有一条向量。
        # 重复会让检索返回同一 chunk 两次（引用重复）。
        #
        # 这个唯一约束也是**回填的幂等保证**：重跑回填时用
        # ON CONFLICT DO UPDATE，不会产生重复行。
        UniqueConstraint("chunk_id", "index_version_id", name="uq_chunk_embeddings_chunk_version"),
        # ---- 检索主路径 ----
        # WHERE index_version_id = <active> ORDER BY embedding <=> ?
        # 版本过滤必须在最左：它区分"这套向量属于哪个版本"，
        # 是所有重建场景的第一条件。
        Index(
            "ix_chunk_embeddings_version",
            "index_version_id",
            "chunk_id",
        ),
        # ---- HNSW 向量索引 ----
        #
        # 建在旁表上，且**不带版本过滤**：HNSW 是全局索引，
        # 部分索引（WHERE index_version_id = 某个具体值）会因为
        # 版本 ID 是动态的而无法预建——每建一个新版本就要建一个新索引，
        # 那正是"重建索引"要做的事，会循环依赖。
        #
        # 所以：一个全局 HNSW 索引，查询时按 index_version_id 过滤，
        # 靠 pgvector 0.8+ 的 `hnsw.iterative_scan = relaxed_order`
        # 在结果不足时自动扩大扫描（见 `ChunkRepository.search_vector`）。
        #
        # ## 为什么**不能**用部分索引（A10 实测结论）
        #
        # 直觉上 `WHERE embedding IS NOT NULL` 的部分索引能省空间
        # （`embedding` 列本身是 NOT NULL，所以这个谓词恒真，本该无害）。
        # 但实测（pgvector 0.8.7，20000 行 × 1024 维）：
        #
        #   完整 HNSW 索引        → Index Scan using hnsw，4.2ms
        #   部分 HNSW 索引        → Seq Scan + Sort，131ms（**完全不被选中**）
        #   部分索引 + 显式谓词   → 仍然 Seq Scan，131ms
        #   部分索引 + 强制       → 仍然 Seq Scan（enable_seqscan=off 也无效）
        #
        # 即"部分索引"这个形式本身让 HNSW 无法用于 ORDER BY 向量排序。
        # 代价是 30 倍延迟，而且**不报任何错**——只有 EXPLAIN 能发现。
        #
        # 因此这里建**完整索引**，不加 postgresql_where。
        Index(
            "ix_chunk_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="token_non_negative",
        ),
    )

    def __repr__(self) -> str:
        return f"<ChunkEmbedding chunk={self.chunk_id} version={self.index_version_id}>"


__all__ = ["ChunkEmbedding"]
