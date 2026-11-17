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
from rag_db.models.embedding import ChunkEmbedding


def _quote_ts_token(token: str) -> str:
    """把词项包成 tsquery 的『引号字面量』。

    不包裹时，`E1024` 里的符号（`.`、`-`、`:`）会被当成 tsquery 操作符，
    一个错误的关键词会让整条 SQL 报语法错误。包裹后所有字符都是词项本身，
    内部单引号按 SQL 字面量规则双写。
    """
    return "'" + token.replace("'", "''") + "'"


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
        index_version_id: uuid.UUID | None = None,
    ) -> int:
        """批量插入 chunk，返回插入条数。

        用 `add_all` + 一次 flush 而不是逐条 add/flush：
        一份文档可能有几百个 chunk，逐条往返会让切分步骤的耗时主要花在
        网络往返上（实测几百 chunk 从数秒降到百毫秒级）。

        ## 向量写到旁表（A10）

        `chunks` 表中**没有** embedding 列（A10 之后向量在
        `chunk_embeddings` 旁表上）。但为了兼容"切分与向量化一步完成"
        的调用方（以及测试），本方法接受块里的 `embedding` 字段，
        并在 `index_version_id` 非空时写入旁表。

        若块带 `embedding` 但没传 `index_version_id`，会**丢弃向量**
        并告警——向量必须归属某个索引版本，否则检索永远查不到
        （`chunk_embeddings.index_version_id` 是 NOT NULL）。
        静默丢弃是刻意选择：切分 handler 的正常路径就是不带向量，
        带向量只是"顺手一步到位"的便利路径。

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
                # 关键词检索的预分词结果。调用方（worker）负责用
                # `rag_adapters.retrieval.keyword.tokenize_for_index` 生成；
                # 直接插入的测试/历史数据留空，FTS 表达式会回退到 content。
                search_text=item.get("search_text") or "",
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
                is_current=item.get("is_current", True),
            )
            for item in chunks
        ]
        self._session.add_all(rows)
        await self._session.flush()

        # 旁表向量（仅当调用方提供了向量 + 索引版本）
        if index_version_id is not None:
            embedding_rows = [
                ChunkEmbedding(
                    tenant_id=tenant_id,
                    chunk_id=row.id,
                    index_version_id=index_version_id,
                    embedding=item["embedding"],
                    embedding_model=item.get("embedding_model") or "unknown",
                )
                for row, item in zip(rows, chunks, strict=True)
                if item.get("embedding") is not None
            ]
            if embedding_rows:
                self._session.add_all(embedding_rows)
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
        self,
        *,
        tenant_id: uuid.UUID,
        index_version_id: uuid.UUID,
        limit: int = 100,
    ) -> list[Chunk]:
        """扫描在某索引版本下**还没有向量**的 chunk。

        向量化可能因为限流而部分失败，需要能续跑而不是整批重来。

        **判据是"旁表里没有该版本的行"**，不是"chunks.embedding IS NULL"：
        旁表方案下 `chunks.embedding` 是历史遗留列，不再更新；
        真正的"是否已向量化"只由旁表决定。
        """
        embedded_exists = (
            select(ChunkEmbedding.chunk_id)
            .where(
                ChunkEmbedding.chunk_id == Chunk.id,
                ChunkEmbedding.index_version_id == index_version_id,
            )
            .exists()
        )
        result = await self._session.execute(
            select(Chunk)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.is_current.is_(True),
                Chunk.chunk_level == 0,
                ~embedded_exists,
            )
            .order_by(Chunk.document_id, Chunk.chunk_index)
            .limit(limit)
        )
        return list(result.scalars())

    async def set_embeddings(
        self,
        *,
        updates: Sequence[tuple[uuid.UUID, list[float], str]],
        index_version_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> int:
        """把向量写入**旁表**（某个索引版本下）。

        用 `ON CONFLICT (chunk_id, index_version_id) DO UPDATE` 而不是纯 INSERT：

        重派任务会对同一 chunk 同一版本再写一次。纯 INSERT 会撞唯一约束
        （报错信息是"duplicate key"，但根因是"没做幂等"）。
        DO UPDATE 让重跑安全，且不会产生重复行（重复行会让检索返回
        同一 chunk 两次——引用重复）。

        必须带 `tenant_id`：旁表是 RLS 表，写入时 RLS 会校验它。
        """
        if not updates:
            return 0

        from sqlalchemy.dialects.postgresql import insert as pg_insert

        rows = [
            {
                "tenant_id": tenant_id,
                "chunk_id": chunk_id,
                "index_version_id": index_version_id,
                "embedding": embedding,
                "embedding_model": model,
            }
            for chunk_id, embedding, model in updates
        ]
        stmt = pg_insert(ChunkEmbedding).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_chunk_embeddings_chunk_version",
            set_={
                "embedding": stmt.excluded.embedding,
                "embedding_model": stmt.excluded.embedding_model,
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)
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
        index_version_ids: Sequence[uuid.UUID] | None = None,
        candidate_multiplier: int = 4,
    ) -> list[tuple[Chunk, float]]:
        """向量检索（两段式）。返回 (chunk, cosine_distance)，距离升序。

        ## 为什么必须两段式（A10 实测结论）

        直觉写法是一条 JOIN：

            SELECT c.* FROM chunks c JOIN chunk_embeddings ce ON ...
            WHERE c.tenant_id = ? AND ...
            ORDER BY ce.embedding <=> ? LIMIT 10

        **这个写法用不上 HNSW 索引**。实测（20000 行 × 1024 维）：

            JOIN 写法          → Hash Join + Seq Scan + Sort，282ms
            纯旁表 HNSW        → Index Scan using hnsw，7.6ms（快 37 倍）

        原因：`ORDER BY` 的距离项来自 JOIN 的右表，planner 无法把它
        下推成"先走索引取 top-N 再 JOIN"，只能全表扫完再排序。
        `enable_seqscan=off` 也救不回来（它只能禁用顺序扫描，
        无法让 JOIN 变成嵌套循环 + 索引扫描）。

        所以改成两段：
          1. **只在旁表上**走 HNSW，取 `limit × candidate_multiplier` 个候选 ID；
          2. 再按 ID 取 chunk 内容，并在此处应用 ACL/知识库过滤。

        ## candidate_multiplier 为什么存在

        第一段在旁表上无法应用 chunk 级的过滤（ACL、知识库、is_current），
        只有拿到 chunk 行才知道。所以第一段必须**多取一些**，
        第二段过滤后才有可能凑够 limit。

        取 4 倍是经验值：ACL 过滤通常只排除少数文档；若过滤极严
        （如只允许 1% 文档），4 倍也可能不够，此时应把 ACL 条件
        下推到第一段（需要 chunk 表与旁表 JOIN，代价是失去索引——
        取舍见 ADR 0002）。

        ## 其他参数

        - `knowledge_base_ids`：限定检索范围。None 表示租户内全部。
        - `max_distance`：距离上限（余弦距离 > 0.4 通常已无关）。
        - `ef_search`：候选列表大小，越大召回越高、越慢。
        - `iterative_scan`：`relaxed_order` / `strict_order` / `off`。
        - `index_version_ids`：只查这些索引版本的向量。**零停机重建必须带**：
          重建期间库里有两套向量，不过滤会返回重复内容且距离不可比。

        **SET LOCAL 必须在本事务内**：避免污染连接池里的其它请求。
        """
        if ef_search is not None:
            await self._session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
        if iterative_scan != "off":
            # 白名单校验：这个值不能参数化，只能字符串拼接
            if iterative_scan not in ("relaxed_order", "strict_order"):
                raise ValueError(f"非法 iterative_scan 值: {iterative_scan}")
            await self._session.execute(text(f"SET LOCAL hnsw.iterative_scan = {iterative_scan}"))

        # ---- 第一段：旁表上走 HNSW ----
        distance = ChunkEmbedding.embedding.cosine_distance(query_embedding).label("distance")
        candidate_limit = max(limit * candidate_multiplier, limit)

        stage_one = (
            select(ChunkEmbedding.chunk_id, distance)
            .where(ChunkEmbedding.tenant_id == tenant_id)
            .order_by(distance)
            .limit(candidate_limit)
        )
        if index_version_ids:
            stage_one = stage_one.where(
                ChunkEmbedding.index_version_id.in_(list(index_version_ids))
            )
        if max_distance is not None:
            stage_one = stage_one.where(distance <= max_distance)

        rows = list((await self._session.execute(stage_one)).all())
        if not rows:
            return []

        # 保持第一段的相似度顺序（字典序会被打乱，那会让最终排序错）
        ordered_ids = [row[0] for row in rows]
        distance_by_id = {row[0]: float(row[1]) for row in rows}

        # ---- 第二段：取内容并过滤 ----
        #
        # **ACL 过滤在这一段**：它是 chunk 级条件（依赖文档密级/ACL 表），
        # 第一段在旁表上无法表达。这也是 candidate_multiplier 存在的原因。
        conditions = [
            Chunk.id.in_(ordered_ids),
            Chunk.tenant_id == tenant_id,
            Chunk.is_current.is_(True),
            # 父块不参与检索（small-to-big：父块只用于回填上下文）
            Chunk.chunk_level == 0,
        ]
        if access_filter is not None:
            conditions.append(access_filter)
        if knowledge_base_ids:
            conditions.append(Chunk.knowledge_base_id.in_(list(knowledge_base_ids)))

        stage_two = select(Chunk).where(*conditions)
        chunks = list((await self._session.execute(stage_two)).scalars())
        by_id = {chunk.id: chunk for chunk in chunks}

        # 按第一段的距离顺序组装，并截断到 limit
        results: list[tuple[Chunk, float]] = []
        for chunk_id in ordered_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue  # 被第二段过滤掉（ACL / 知识库 / 非当前版本）
            results.append((chunk, distance_by_id[chunk_id]))
            if len(results) >= limit:
                break
        return results

    async def search_keyword(
        self,
        *,
        tenant_id: uuid.UUID,
        keywords: Sequence[str],
        knowledge_base_ids: Sequence[uuid.UUID] | None = None,
        limit: int = 10,
        access_filter: Any = None,
    ) -> list[tuple[Chunk, float]]:
        """关键词检索（jieba 预分词 + PostgreSQL 内置 FTS + `ts_rank`）。

        ## 为什么不直接用 `to_tsvector('simple', content)`

        `simple` 分词器**不认中文**：一整句中文是一个 token，
        查询 `二线` 永远匹配不到「二线城市住宿标准」。ADR 0003 的评估
        因此得到"关键词路零命中、混合检索零收益"的错误结论。

        写入侧已经用 jieba 预分词（`chunks.search_text`），所以这里用
        **同一表达式**（生成列 `search_vector`）建索引与查询：

            to_tsvector('simple', coalesce(nullif(search_text, ''), content))

        空 `search_text` 回退到 content：直接 SQL 插入的历史/测试数据
        仍能按英文词项检索。中文历史数据需要跑
        `scripts/backfill_search_text.py`。

        ## 为什么用 `ts_rank` 而不是子串计数

        `ts_rank` 计入词频与文档长度归一化；子串计数把"出现 5 次的
        核心词"与"出现 1 次的偶然词"当成一样。没有真正的 IDF，
        但已经远好于 ILIKE 的"包含即 1 分"。

        ## `access_filter` 必须传

        双路检索做 RRF 融合。只给向量路加 ACL 而关键词路没加，
        越权内容会因"排名靠前"被融合抬到更前面——**漏一处等于全漏**
        （`tests/test_acl_retrieval.py` 盯着这一点）。
        """
        cleaned = [keyword.strip() for keyword in keywords if keyword.strip()]
        if not cleaned:
            return []

        # 上限保护：OR 链过长会让 SQL 解析与执行都变慢（ADR 0003）。
        cleaned = cleaned[:32]
        query_text = " | ".join(_quote_ts_token(keyword) for keyword in cleaned)

        # 直接查生成列 `search_vector`（索引就建在它上面）。
        #
        # 不要在这里重写 `to_tsvector(...)` 表达式：即使 SQL 文本相同，
        # planner 也不保证把表达式匹配回生成列，最坏情况是**全表扫描**
        # 而不报错。生成列已包含"空 search_text 回退到 content"的逻辑。
        search_vector = Chunk.search_vector
        tsquery = func.to_tsquery("simple", query_text)
        rank = func.ts_rank(search_vector, tsquery).label("rank")

        conditions = [
            Chunk.tenant_id == tenant_id,
            Chunk.is_current.is_(True),
            # 父块不参与检索（与 search_vector 保持一致——否则父块与子块
            # 同时出现，同一内容占两个结果位）
            Chunk.chunk_level == 0,
            search_vector.op("@@")(tsquery),
        ]
        if knowledge_base_ids:
            conditions.append(Chunk.knowledge_base_id.in_(list(knowledge_base_ids)))
        if access_filter is not None:
            conditions.append(access_filter)

        stmt = (
            select(Chunk, rank)
            .where(*conditions)
            # 同分时短 chunk 优先（信息密度高），最后用 id 保证稳定排序
            # （不稳定排序会让评估结果不可复现）。
            .order_by(rank.desc(), func.length(Chunk.content).asc(), Chunk.id)
            .limit(limit)
        )

        result = await self._session.execute(stmt)
        # 借用"距离语义"（越小越好）让调用方不必区分两路的方向：
        # 取负后，向量路与关键词路都是"升序 = 更相关"。
        return [(row[0], -float(row[1])) for row in result]

    async def get_current_by_ids(
        self,
        *,
        tenant_id: uuid.UUID,
        chunk_ids: Sequence[uuid.UUID],
        access_filter: Any = None,
    ) -> list[Chunk]:
        """按 ID 取当前 chunk（用于把检索结果回填成完整材料）。

        **必须带 tenant_id 与 is_current**：检索图只传 ID，回填时
        再校验一次归属与时效，避免"检索后被删/被换成旧版本"导致
        把过期内容拼进 prompt。`access_filter` 是第二道防线
        （ID 来自已过滤的检索，但多传一次成本极低）。

        返回顺序由调用方决定（按检索排名组装，不能按数据库返回顺序）。
        """
        if not chunk_ids:
            return []

        conditions = [
            Chunk.id.in_(list(chunk_ids)),
            Chunk.tenant_id == tenant_id,
            Chunk.is_current.is_(True),
        ]
        if access_filter is not None:
            conditions.append(access_filter)

        result = await self._session.execute(select(Chunk).where(*conditions))
        return list(result.scalars())

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
