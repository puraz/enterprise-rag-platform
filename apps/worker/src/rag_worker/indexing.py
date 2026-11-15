"""索引版本重建：零停机切换（旁表方案）。

## 为什么"重建索引"是个难题

朴素做法是停机重建：

    STOP 查询 → 清空旧向量 → 重新算全部向量 → 建索引 → START

一个知识库可能有几万 chunk（GPU 上几十分钟）。这期间检索完全不可用。

## 零停机的关键：向量存在**旁表**里

    chunks              内容（与向量无关，一份）
      └── chunk_embeddings   向量（每个索引版本一份）

回填 V2 时**新增** chunk_embeddings 行，旧行（V1）**完全不动**：

    回填中：  chunk A → [V1 向量] [V2 向量]
    查询时：  WHERE index_version_id = <ACTIVE=V1>  → 仍走 V1，不受影响
    切换：    UPDATE 把 ACTIVE 指向 V2（一条语句，秒级）
    回滚：    UPDATE 指回 V1（V1 的行还在，秒级）
    回收：    DELETE FROM chunk_embeddings WHERE index_version_id = V1

## 曾经走过的弯路（记录在此，避免重犯）

最初的设计是 `chunks.embedding` 单列 + `chunks.index_version_id`，
回填时就地覆盖向量并改版本号。演练（scripts/drill_index_rebuild.py）
证明它是错的：

    把 chunk 的 index_version_id 改成 V2 的那一刻
      → 该 chunk 从 V1 的查询结果里消失
      → 回填完成时 V1 空了
      → 既有停机窗口，又无法回滚（旧向量被覆盖）

根因是"一行只能存一个向量"。**这个限制不是靠更小心地写代码能绕过的**，
必须改物理结构。所以改用旁表。

## 代价（诚实记录）

- 回填期间向量存储翻倍（100 万 chunk × 4KB × 2 ≈ 8GB）；
- 检索多一次 JOIN（HNSW 命中后按主键 JOIN，成本很低）；
- 回收旧版本前要观察一段时间（否则失去回滚能力）。

这个代价是"零停机"的必需品，不是可以优化掉的浪费。

## 中断与回滚

- 回填中途崩溃：已写入的 V2 行保留，重启后从"还缺哪些 chunk"继续。
  判据是"旁表里没有该版本的行"，不是计数器。
- 切换后发现问题：`rollback_to(旧版本)` 把 ACTIVE 改回旧版本，
  旧行还在，秒级生效。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from rag_adapters.providers.embedding import EmbeddingProvider, EmbeddingTask
from rag_core.logging import get_logger
from rag_db.models.chunk import Chunk, IndexVersion
from rag_db.models.embedding import ChunkEmbedding
from rag_db.repositories.chunk import ChunkRepository, IndexVersionRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RebuildProgress:
    """一次回填的进度快照。"""

    index_version_id: uuid.UUID
    total: int
    embedded: int

    @property
    def is_complete(self) -> bool:
        return self.total > 0 and self.embedded >= self.total


class IndexRebuilder:
    """索引版本重建编排。

    与 handler 的分工：handler 只处理"单个文档的向量化"，
    这里处理"整个知识库的版本切换"。后者要保证跨文档的一致性
    （切换时所有文档都在同一版本上）。

    ## 事务契约：**本类自己 commit**

    与 `TaskHandler` 不同（handler 绝不 commit，事务归引擎），
    本类的方法**会主动 commit**。原因是它要做的是"分批推进的长流程"：

    - `backfill()` 每批提交一次 → 崩溃后从断点继续，不必从头再来；
    - `activate()` / `rollback()` 各自是一条原子操作，必须立即生效。

    调用方应传入 `db.session()`（自动提交的会话）或自己管理事务，
    **不要传 `db.transaction()` 的会话**——那会重现
    "Can't operate on closed transaction" 错误。
    """

    def __init__(self, *, provider: EmbeddingProvider) -> None:
        self._provider = provider

    async def start(
        self,
        *,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        version_tag: str,
        index_params: dict[str, Any] | None = None,
    ) -> IndexVersion:
        """创建新的 building 版本。

        **不切换 ACTIVE**：这时查询仍走旧版本，新版本还不可见。
        允许同时存在多个 building 版本吗？技术上可以（version_tag 唯一），
        但生产上不该并行跑多个——它们会争抢 GPU。调用方应保证串行。
        """
        repo = IndexVersionRepository(session)

        # 预期 chunk 数：决定"回填完成"的判定依据
        total = await self._count_chunks(session, tenant_id=tenant_id, kb_id=knowledge_base_id)

        version = await repo.create(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            version_tag=version_tag,
            embedding_model=self._provider.name,
            embedding_dim=self._provider.capabilities.dimension,
            index_params=index_params or {"m": 16, "ef_construction": 64},
            expected_chunk_count=total,
        )
        log.info(
            "index_version_created",
            index_version_id=str(version.id),
            knowledge_base_id=str(knowledge_base_id),
            version_tag=version_tag,
            expected_chunk_count=total,
            model=self._provider.name,
        )
        return version

    async def backfill(
        self,
        *,
        session: AsyncSession,
        index_version: IndexVersion,
        batch_size: int = 32,
        max_batches: int | None = None,
    ) -> RebuildProgress:
        """把属于其它版本（或未归属）的 chunk 重算到目标版本。

        每批提交：崩溃后从断点继续，不必从头再来。

        Args:
            max_batches: 单次调用最多处理多少批（测试用，避免一次跑完）。

        Returns:
            进度快照。`is_complete` 为 True 表示可以切换。
        """
        total = await self._count_chunks(
            session,
            tenant_id=index_version.tenant_id,
            kb_id=index_version.knowledge_base_id,
        )
        embedded = 0
        batches = 0

        while True:
            if max_batches is not None and batches >= max_batches:
                break

            pending = await self._claim_batch(
                session,
                index_version=index_version,
                batch_size=batch_size,
            )
            if not pending:
                break

            texts = [chunk.embedding_input or chunk.content for chunk in pending]
            result = await self._provider.embed_batch(texts, task=EmbeddingTask.DOCUMENT)

            # 顺序必须严格对应：错位会让向量与内容不匹配且不报错
            pairs = list(zip(pending, result.vectors, strict=True))

            # **新增旁表行，不改 chunk**
            #
            # 这是零停机的核心：旧版本（V1）的向量行完全不动，
            # 所以在途查询继续走 V1 时结果完全不受影响。
            # 如果改 chunk 的归属，那个 chunk 会立刻从 V1 查询里消失
            # —— 这正是旧方案失败的原因。
            await ChunkRepository(session).set_embeddings(
                updates=[(chunk.id, vector, self._provider.name) for chunk, vector in pairs],
                index_version_id=index_version.id,
                tenant_id=index_version.tenant_id,
            )
            await session.commit()

            embedded += len(pairs)
            batches += 1

            log.info(
                "index_backfill_batch",
                index_version_id=str(index_version.id),
                batch=batches,
                embedded=embedded,
                total=total,
            )

            if batches >= 1 and embedded >= total:
                break

        # 更新版本的 chunk_count（供切换前校验）
        actual = await self._count_version_chunks(session, index_version_id=index_version.id)
        await IndexVersionRepository(session).mark_ready(
            version_id=index_version.id, chunk_count=actual
        )
        await session.commit()

        return RebuildProgress(index_version_id=index_version.id, total=total, embedded=embedded)

    async def activate(
        self,
        *,
        session: AsyncSession,
        index_version: IndexVersion,
    ) -> None:
        """切换到新版本（alias 切换）。

        **切换前硬校验**：新版本的 chunk 数必须等于知识库的当前 chunk 总数。

        为什么必须校验：本方案是"就地重算"，回填期间新算出的 chunk
        属于 building 版本，对查询不可见。如果没回填完就切换，
        查询只能看到已重算的那部分 → **静默丢失结果**。
        这比"切换失败"严重得多：用户会得到"知识库里查不到这份文档"，
        而实际上文档在库里，只是向量没算完。
        """
        expected = index_version.expected_chunk_count
        actual = await self._count_version_chunks(session, index_version_id=index_version.id)

        if expected is not None and actual < expected:
            raise _RebuildError(
                f"回填未完成，拒绝切换：预期 {expected} 个 chunk，实际 {actual}。"
                f"未完成就切换会导致检索静默丢失内容。"
            )

        await IndexVersionRepository(session).activate(
            knowledge_base_id=index_version.knowledge_base_id,
            version_id=index_version.id,
        )
        await session.commit()

        log.info(
            "index_version_activated",
            index_version_id=str(index_version.id),
            knowledge_base_id=str(index_version.knowledge_base_id),
            chunk_count=actual,
            note="切换是一条 UPDATE，在途查询不受影响",
        )

    async def rollback(
        self,
        *,
        session: AsyncSession,
        knowledge_base_id: uuid.UUID,
        to_version_id: uuid.UUID,
    ) -> None:
        """回滚到指定版本。

        **旧版本的向量在切换时没有删**，所以回滚只是把 ACTIVE 改回去——
        秒级生效。这也是为什么"回收旧版本要延迟观察一段时间"：
        立刻回收会让回滚失去可能。
        """
        repo = IndexVersionRepository(session)
        target = await session.get(IndexVersion, to_version_id)
        if target is None:
            raise _RebuildError(f"目标版本不存在：{to_version_id}")
        if target.knowledge_base_id != knowledge_base_id:
            raise _RebuildError("目标版本不属于该知识库")

        await repo.activate(knowledge_base_id=knowledge_base_id, version_id=to_version_id)
        await session.commit()

        log.warning(
            "index_version_rolled_back",
            to_version_id=str(to_version_id),
            knowledge_base_id=str(knowledge_base_id),
            note="已回滚到旧版本，请排查新版本的问题",
        )

    # ------------------------------------------------------------ 内部

    async def _count_chunks(
        self, session: AsyncSession, *, tenant_id: uuid.UUID, kb_id: uuid.UUID
    ) -> int:
        """知识库当前参与检索的 chunk 总数。"""
        result = await session.execute(
            select(func.count())
            .select_from(Chunk)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.knowledge_base_id == kb_id,
                Chunk.is_current.is_(True),
                Chunk.chunk_level == 0,  # 父块不参与检索，不算进"待回填"
            )
        )
        return int(result.scalar_one())

    async def _count_version_chunks(
        self, session: AsyncSession, *, index_version_id: uuid.UUID
    ) -> int:
        """某个索引版本下有多少条向量（只看参与检索的叶子 chunk）。

        基于旁表 COUNT：这是"回填完成度"的唯一权威来源。
        不用 `IndexVersion.chunk_count` 计数器——它可能滞后或错。
        """
        result = await session.execute(
            select(func.count())
            .select_from(ChunkEmbedding)
            .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
            .where(
                ChunkEmbedding.index_version_id == index_version_id,
                Chunk.is_current.is_(True),
                Chunk.chunk_level == 0,
            )
        )
        return int(result.scalar_one())

    async def _claim_batch(
        self,
        session: AsyncSession,
        *,
        index_version: IndexVersion,
        batch_size: int,
    ) -> list[Chunk]:
        """取一批"还没归入目标版本"的 chunk。

        **不加 FOR UPDATE SKIP LOCKED**：这里只有一个重建进程在跑
        （多副本并行重建会争抢 GPU 且难以协调进度）。
        如果未来要支持并行回填，应加锁并分区（按 document_id 取模）。

        排序用 `chunk_index` 而不是随机：进度可预测、日志可读，
        且失败后用同样的顺序扫描能保证"这次处理过的不会再被选中"
        （因为它们已被改成目标版本）。
        """
        # 判据：**旁表里还没有这个版本的行**
        #
        # 不能用"chunk.index_version_id != 目标"——旁表方案下 chunk 没有
        # 版本归属字段（一个 chunk 同时属于多个版本）。
        exists_for_version = (
            select(ChunkEmbedding.chunk_id)
            .where(
                ChunkEmbedding.chunk_id == Chunk.id,
                ChunkEmbedding.index_version_id == index_version.id,
            )
            .exists()
        )
        result = await session.execute(
            select(Chunk)
            .where(
                Chunk.tenant_id == index_version.tenant_id,
                Chunk.knowledge_base_id == index_version.knowledge_base_id,
                Chunk.is_current.is_(True),
                Chunk.chunk_level == 0,
                ~exists_for_version,
            )
            .order_by(Chunk.document_id, Chunk.chunk_index)
            .limit(batch_size)
            # SKIP LOCKED：防止两个重建进程重复处理同一批
            .with_for_update(skip_locked=True)
        )
        return list(result.scalars())


class _RebuildError(Exception):
    """重建/切换失败。不可重试：需要人介入判断是补做回填还是回滚。"""


__all__ = ["IndexRebuilder", "RebuildProgress"]
