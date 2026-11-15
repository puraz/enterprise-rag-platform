"""分块与检索仓储。

## 关键设计

### 向量检索的过滤顺序问题

朴素写法：

    SELECT * FROM chunks
    WHERE tenant_id = ? AND knowledge_base_id = ?
    ORDER BY embedding <=> ? LIMIT 10

PostgreSQL 会**先用 HNSW 索引扫描**再过滤。若目标租户只占总量的 1%，
要扫约 1000 个候选才能凑够 10 条 —— 或更糟，`hnsw.ef_search=40`
时扫 40 个候选中一个都不匹配，**直接返回空结果**。

缓解手段（本项目已启用）：
1. `SET LOCAL hnsw.iterative_scan = relaxed_order` —— pgvector 0.8.0+
   让查询自动扩大扫描直到凑够 limit 条；
2. 索引上的部分条件 `WHERE is_current = true` 减少无效数据。

**注意** `hnsw.iterative_scan` 是扩展级 GUC，必须先用过 vector 扩展
（会话里存在 vector 类型）才能 SET，否则报 parameter not found。
这也是为什么要在每个检索事务里显式设置、而不是在连接初始化时一次性设。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums.retrieval import ChunkType
from rag_db.models.chunk import Chunk, IndexVersion


class ChunkRepository:
    """分块数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_create(
        self,
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        document_id: uuid.UUID,
        document_version_id: uuid.UUID,
        chunks: Sequence[dict[str, Any]],
    ) -> int:
        """批量插入 chunk，返回插入条数。

        用 `add_all` + 一次 flush 而不是逐条 add/flush：
        一份文档可能有几百个 chunk，逐条往返会让切分步骤的耗时主要花在
        网络往返上（实测几百 chunk 从数秒降到百毫秒级）。

        调用方负责提供完整字段（含 content_hash），
        这里不做业务校验 —— 校验在 chunker 与 service 层。
        """
        rows = [
            Chunk(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                document_version_id=document_version_id,
                chunk_index=item["chunk_index"],
                content=item["content"],
                content_hash=item["content_hash"],
                chunk_type=item.get("chunk_type", ChunkType.TEXT.value),
                embedding_input=item.get("embedding_input"),
                parent_chunk_id=item.get("parent_chunk_id"),
                chunk_level=item.get("chunk_level", 0),
                page_no=item.get("page_no"),
                char_start=item.get("char_start"),
                char_end=item.get("char_end"),
                bbox=item.get("bbox"),
                section_path=item.get("section_path", []),
                token_count=item.get("token_count"),
                chunk_metadata=item.get("chunk_metadata", {}),
                # 向量与模型标识：切分阶段可能还没有（先入库后向量化），
                # 也可能一步到位（切分与 embedding 在同一流水线阶段完成）
                embedding=item.get("embedding"),
                embedding_model=item.get("embedding_model"),
                is_current=item.get("is_current", True),
            )
            for item in chunks
        ]
        self._session.add_all(rows)
        await self._session.flush()
        return len(rows)

    async def list_by_version(
        self, *, document_version_id: uuid.UUID, limit: int | None = None
    ) -> list[Chunk]:
        stmt = (
            select(Chunk)
            .where(Chunk.document_version_id == document_version_id)
            .order_by(Chunk.chunk_index)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def list_pending_embedding(
        self, *, tenant_id: uuid.UUID, limit: int = 100
    ) -> list[Chunk]:
        """扫描尚未向量化的 chunk（补偿任务用）。

        向量化可能因为供应商限流而部分失败，需要能续跑而不是整批重来。
        """
        result = await self._session.execute(
            select(Chunk)
            .where(Chunk.tenant_id == tenant_id, Chunk.embedding.is_(None))
            .order_by(Chunk.created_at)
            .limit(limit)
        )
        return list(result.scalars())

    async def set_embeddings(
        self,
        *,
        updates: Sequence[tuple[uuid.UUID, list[float], str]],
    ) -> int:
        """批量写入向量。updates 为 (chunk_id, embedding, model) 列表。

        逐条 UPDATE 是有意的：不同 chunk 的向量长度相同但值不同，
        无法用单条 UPDATE 表达。用 executemany（SQLAlchemy 的
        `execute` 传 list）让驱动做批量绑定，避免 N 次网络往返。
        """
        if not updates:
            return 0
        await self._session.execute(
            update(Chunk),
            [
                {"id": chunk_id, "embedding": embedding, "embedding_model": model}
                for chunk_id, embedding, model in updates
            ],
        )
        return len(updates)

    async def mark_version_chunks_stale(self, *, document_version_id: uuid.UUID) -> int:
        """把某版本的 chunk 标记为非当前（切到新版本时调用）。

        不直接删：回滚到旧版本时还要用这些 chunk。
        """
        result = await self._session.execute(
            update(Chunk)
            .where(Chunk.document_version_id == document_version_id, Chunk.is_current.is_(True))
            .values(is_current=False)
            .returning(Chunk.id)
        )
        return len(list(result.scalars()))

    async def delete_by_version(self, *, document_version_id: uuid.UUID) -> int:
        """物理删除某版本的全部 chunk（版本被永久清理时）。

        用 `RETURNING id` 统计删除条数，而不是 `result.rowcount`：
        SQLAlchemy 的 `execute()` 在类型标注上返回 `Result`，而 `rowcount`
        只存在于 `CursorResult`；用 RETURNING 既类型安全，也让"删了多少"
        由数据库直接告知，不依赖驱动的隐式行为。
        """
        result = await self._session.execute(
            delete(Chunk)
            .where(Chunk.document_version_id == document_version_id)
            .returning(Chunk.id)
        )
        return len(list(result.scalars()))

    async def search_vector(
        self,
        *,
        tenant_id: uuid.UUID,
        query_embedding: list[float],
        knowledge_base_ids: Sequence[uuid.UUID] | None = None,
        limit: int = 10,
        max_distance: float | None = None,
        ef_search: int | None = None,
        iterative_scan: str = "relaxed_order",
        access_filter: Any = None,
    ) -> list[tuple[Chunk, float]]:
        """向量检索。返回 (chunk, cosine_distance) 列表，距离升序。

        参数说明：
        - `knowledge_base_ids`：限定检索范围。None 表示租户内全部。
        - `max_distance`：距离上限，用于过滤明显不相关的结果
          （余弦距离 > 0.4 通常已经无关，具体阈值要在 A11 用黄金集调）。
        - `ef_search`：查询时候选列表大小。越大召回越高、越慢。
          默认 40 是 pgvector 默认值；A11 压测时按召回率调。
        - `iterative_scan`：`relaxed_order` / `strict_order` / `off`。
          见模块 docstring —— 多租户下过滤后召回的关键开关。

        **必须在同一事务内 SET LOCAL**：`SET LOCAL` 只影响当前事务，
        避免污染连接池里的其它请求（用 `SET` 会泄漏到下一个借用者）。
        """
        # 扩展级 GUC：先确保 vector 类型已加载（SELECT 一次即可）
        if ef_search is not None:
            await self._session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
        if iterative_scan != "off":
            # 白名单校验，防 SQL 注入（这个值不能参数化，只能字符串拼接）
            if iterative_scan not in ("relaxed_order", "strict_order"):
                raise ValueError(f"非法 iterative_scan 值: {iterative_scan}")
            await self._session.execute(text(f"SET LOCAL hnsw.iterative_scan = {iterative_scan}"))

        # 余弦距离运算符 <=>：值越小越相似
        distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
        conditions = [
            Chunk.tenant_id == tenant_id,
            Chunk.is_current.is_(True),
            Chunk.embedding.is_not(None),
        ]
        # **ACL 过滤必须进 WHERE**（不能查完再在 Python 里过滤）：
        # - 查完再过滤会让分页错乱、越权内容进应用内存、浪费 I/O；
        # - 检索返回的是文档内容本身，泄漏发生在答案里，比元数据泄漏更严重。
        if access_filter is not None:
            conditions.append(access_filter)

        stmt = select(Chunk, distance).where(*conditions).order_by(distance).limit(limit)
        if knowledge_base_ids:
            stmt = stmt.where(Chunk.knowledge_base_id.in_(knowledge_base_ids))
        if max_distance is not None:
            stmt = stmt.where(distance <= max_distance)

        result = await self._session.execute(stmt)
        return [(row[0], float(row[1])) for row in result]

    async def search_keyword(
        self,
        *,
        tenant_id: uuid.UUID,
        keywords: Sequence[str],
        knowledge_base_ids: Sequence[uuid.UUID] | None = None,
        limit: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """关键词检索（BM25 的简化替代：ts_rank 全文检索）。

        为什么先用 PG 全文检索而不是引入 OpenSearch：
        `docs/PROMPT.md` 允许"PG 全文检索 + 中文分词"作为 SparseIndex 实现。
        少一个中间件；中文分词精度不如专业 BM25，但起步够用。
        `SparseIndex` 抽象层保证后续可换（A10）。

        注意：默认的 `simple` 分词器不认识中文，会把整个句子当一个词。
        A10 需要装 zhparser 或改用 pg_bigm / pg_trgm。
        这里先用 `to_tsvector('simple', ...)` 让整条链路可跑通并写清限制。
        """
        if not keywords:
            return []

        # 用 OR 连接关键词：任一命中即可，rank 排序
        query_text = " | ".join(keyword.replace("'", "''") for keyword in keywords)
        rank = func.ts_rank(
            func.to_tsvector("simple", Chunk.content),
            func.to_tsquery("simple", query_text),
        ).label("rank")

        stmt = (
            select(Chunk, rank)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.is_current.is_(True),
                func.to_tsvector("simple", Chunk.content).op("@@")(
                    func.to_tsquery("simple", query_text)
                ),
            )
            .order_by(rank.desc())
            .limit(limit)
        )
        if knowledge_base_ids:
            stmt = stmt.where(Chunk.knowledge_base_id.in_(knowledge_base_ids))

        result = await self._session.execute(stmt)
        # 关键词检索的 rank 越大越相关，统一转成"距离"语义（越小越好）取负
        return [(row[0], -float(row[1])) for row in result]

    async def count_by_version(self, *, document_version_id: uuid.UUID) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_version_id == document_version_id)
        )
        return int(result.scalar_one())

    async def expanded_context(
        self, *, chunk_ids: Sequence[uuid.UUID], include_parent: bool = True
    ) -> list[Chunk]:
        """父块回填：给定命中的子块，取它们的父块。

        small-to-big 策略的核心：小块检索精度高，但上下文不足；
        命中后用父块补全上下文再送给 LLM。
        """
        if not chunk_ids:
            return []

        if not include_parent:
            result = await self._session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))
            return list(result.scalars())

        # 一次查出：命中的块自身 + 它们的父块
        subq = select(Chunk.parent_chunk_id).where(
            Chunk.id.in_(chunk_ids), Chunk.parent_chunk_id.is_not(None)
        )
        result = await self._session.execute(
            select(Chunk).where(Chunk.id.in_(subq)).order_by(Chunk.chunk_index)
        )
        return list(result.scalars())


class IndexVersionRepository:
    """索引版本数据访问（A10 零停机重建的基础）。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_active(self, *, knowledge_base_id: uuid.UUID) -> IndexVersion | None:
        """取当前生效的索引版本。部分唯一索引保证最多一条。"""
        result = await self._session.execute(
            select(IndexVersion)
            .where(
                IndexVersion.knowledge_base_id == knowledge_base_id,
                IndexVersion.status == "active",
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        version_tag: str,
        embedding_model: str,
        embedding_dim: int,
        index_type: str = "hnsw",
        distance_metric: str = "cosine",
        index_params: dict[str, Any] | None = None,
        expected_chunk_count: int | None = None,
    ) -> IndexVersion:
        version = IndexVersion(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            version_tag=version_tag,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            index_type=index_type,
            distance_metric=distance_metric,
            index_params=index_params or {},
            expected_chunk_count=expected_chunk_count,
            status="building",
        )
        self._session.add(version)
        await self._session.flush()
        return version

    async def activate(self, *, knowledge_base_id: uuid.UUID, version_id: uuid.UUID) -> None:
        """切换 ACTIVE 版本（alias 切换）。

        顺序关键：先把旧的置 retired，再把新的置 active。
        反过来的话会短暂出现两条 active，撞部分唯一索引。

        这个操作是**秒级**的：只改一行状态，不碰 chunk 数据。
        因此在途查询继续用旧版本完成，新请求用新版本 —— 零停机。
        """
        await self._session.execute(
            update(IndexVersion)
            .where(
                IndexVersion.knowledge_base_id == knowledge_base_id,
                IndexVersion.status == "active",
            )
            .values(status="retired")
        )
        await self._session.execute(
            update(IndexVersion)
            .where(IndexVersion.id == version_id)
            .values(status="active", activated_at=func.now())
        )

    async def mark_ready(self, *, version_id: uuid.UUID, chunk_count: int) -> None:
        await self._session.execute(
            update(IndexVersion)
            .where(IndexVersion.id == version_id)
            .values(status="ready", chunk_count=chunk_count)
        )
