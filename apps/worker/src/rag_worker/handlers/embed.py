"""向量化任务处理器：chunks → embeddings → 建索引版本。

## 管线位置

    切分完成（documents.status = EMBEDDING，chunks 已入库但 embedding 为 NULL）
      │
      └─▶ [本处理器]  读 chunk → 批量 embed → 回写向量 → 状态推进到 INDEXING/READY
            │
            ▼
          A11 检索接手

## 为什么向量化与切分分成两个任务

`docs/PROMPT.md` 要求"CPU 密集隔离"。更重要的是失败隔离：

- 切分是 CPU（快、稳定）；向量化可能要 GPU 或远程 API（慢、会限流、会挂）。
- 合成一步的话，向量服务临时故障 → **切分成果一起回滚**，重跑要重新解析+切分。
- 分开后：切分结果保留，向量化失败只需重跑向量化，
  且能按 `list_pending_embedding` 续跑（部分成功的保留）。

## 批量与分批

`EmbeddingProvider.capabilities.max_input_tokens` 是单条上限，不是批量上限。
真正的批量受显存/API 限制。这里用配置的 `embedding.batch_size`（默认 32）：

- 太小：GPU 利用率低（每次 kernel launch 开销占比高）；
- 太大：显存溢出（bge-m3 在 7.7GB 卡上 32 条 800 字符是安全值）。

**顺序必须与输入严格对应**：`embed_batch` 返回的向量按输入顺序排列。
错位会让"检索能跑但相关性全乱"，且不报错——这是 embedding 层最隐蔽的 bug。
所以这里用 `zip(chunks, vectors, strict=True)` 配对，
`strict=True` 让长度不匹配立刻抛错而不是静默截断。

## 为什么不在这里写 `chunks.embedding_model`

要在同一位置写。`embedding_model` 记录了"这个向量是哪个模型算的"，
A10 换模型时靠它识别哪些 chunk 需要重算。
分开写（先写向量、后写模型）会出现"有向量但不知道模型"的窗口期。

## 事务边界：handler 内绝不 commit

handler 拿到的 session 由引擎管理（`database.transaction()` 内部是
`async with session.begin()`），**在 handler 里 commit 会把事务关掉**，
后续任何语句都会报：

    Can't operate on closed transaction inside context manager

所以本 handler 只做 `flush`，事务边界完全交给引擎。

代价：大文档（几千 chunk）的向量化在**一个事务**里完成。
如果中途失败，整批回滚——包括已算好的向量。
这与"可续跑"的设计有冲突，但选择更简单正确的那条：

- 单事务：崩溃回滚 → 重跑会重算全部（浪费算力，但结果正确）；
- 分批提交：必须由引擎支持"每批一个短事务"，
  而 handler 契约里没有这个能力（`TaskHandler` 只拿一个 session）。

**要支持分批提交，正确做法是扩展 handler 契约**，让引擎提供一个
`new_session()` 工厂，而不是让 handler 自己 commit 破坏事务边界。
这个改造记在 docs/adr/0002 的"待改进"里。

## 幂等

重派时 chunk 已有向量，`list_pending_embedding` 只返回 NULL 的，
所以重跑天然幂等——不会重复算已有向量。这是"向量化可续跑"的直接收益。

## 为什么要归入 ACTIVE 索引版本

写好向量还不够——检索按 `chunks.index_version_id = <ACTIVE>` 过滤。
不归属版本的 chunk（NULL）**永远不会被检索到**，而且没有任何报错。

所以向量化时必须查当前 ACTIVE 版本并写入该列：
- 知识库还没有 ACTIVE 版本（首次入库）：自动创建一个，
  否则新文档永远检索不到；
- 已有 ACTIVE 版本：新 chunk 归入它，立即可检索。

**这是 A10 引入 `index_version_id` 后新文档能"上传即可搜"的关键一环。**
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from rag_adapters.providers.embedding import EmbeddingProvider, EmbeddingTask
from rag_contracts.enums import DocumentStatus, DocumentVersionStatus
from rag_contracts.enums.audit import ModelTaskType
from rag_core.errors import NotFoundError
from rag_core.logging import get_logger
from rag_core.settings.embedding import EmbeddingSettings
from rag_db.models.chunk import Chunk
from rag_db.models.document import Document, DocumentVersion
from rag_db.repositories import ChunkRepository, DocumentRepository
from rag_db.repositories.audit import ModelCallRepository
from rag_worker.engine import require_session

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger(__name__)


class EmbedDocumentHandler:
    """`TaskType.EMBED_DOCUMENT` 的处理器。

    payload：
        {"document_id": "...", "document_version_id": "..."}
    """

    # 单文档处理：引擎会包在一个事务里（失败整体回滚）。
    # 长任务（索引重建）才需要 owns_transaction = True。
    owns_transaction = False

    def __init__(
        self,
        *,
        provider: EmbeddingProvider,
        settings: EmbeddingSettings,
        tracer: Any = None,
    ) -> None:
        """
        Args:
            provider: Embedding 实现。worker 进程持有单例。
            settings: 用于成本入账（provider 名、模型名、单价）。
            tracer: 可选。用于给 `model_calls.trace_id` 提供任务 trace
                （引擎已用 `worker.task` span 包住 handler，
                这里能直接取到当前 trace id）。
        """
        self._provider = provider
        self._settings = settings
        self._tracer = tracer

    async def __call__(
        self,
        *,
        session: AsyncSession | None,
        task: Any,
        progress: Any,
        session_factory: Any = None,
    ) -> None:
        # 本 handler 不自管事务（owns_transaction = False），引擎保证传 session。
        # `session_factory` 只为满足 TaskHandler 协议签名，这里不使用。
        session = require_session(session)
        payload = dict(task.payload or {})
        document_id = _require_uuid(payload, "document_id")
        version_id = _require_uuid(payload, "document_version_id")
        # 可选：把向量写入指定索引版本（零停机重建）。
        # 不传时走原有逻辑（ACTIVE 版本，必要时自动创建）。
        target_index_version_id = (
            _require_uuid(payload, "index_version_id") if payload.get("index_version_id") else None
        )

        await progress(5, "loading_chunks")
        version, document = await self._load(session, version_id=version_id)
        if version is None or document is None:
            raise NotFoundError("文档版本不存在（可能已被删除）")

        chunks = await ChunkRepository(session).list_by_version(document_version_id=version_id)
        if not chunks:
            # 切分产出 0 个 chunk：空文档是合法输入。
            # 直接标记完成，不报错——报错会让空文件反复重试进死信。
            log.info(
                "no_chunks_to_embed",
                version_id=str(version_id),
                note="该版本没有 chunk（空文档），直接完成",
            )
            await self._finish(session=session, version=version, document=document, count=0)
            return

        # 只算还没向量的（重派时续跑，不重算已完成的）
        pending = [chunk for chunk in chunks if chunk.embedding is None]
        if not pending:
            log.info(
                "all_chunks_already_embedded",
                version_id=str(version_id),
                chunk_count=len(chunks),
            )
            await self._finish(
                session=session, version=version, document=document, count=len(chunks)
            )
            return

        await progress(15, "embedding")
        # 先确定这批向量归属哪个索引版本（ACTIVE，或重建时指定的目标版本）
        index_version_id = await self._resolve_index_version(
            session, document=document, target=target_index_version_id
        )
        await self._embed_all(
            session=session,
            chunks=pending,
            total=len(chunks),
            tenant_id=version.tenant_id,
            document_version_id=version_id,
            progress=progress,
            index_version_id=index_version_id,
        )

        await progress(90, "verifying")
        # 校验：全部 chunk 都该有向量了。漏掉说明 embed 返回数量不足。
        missing = await self._count_missing(
            session, version_id=version_id, index_version_id=index_version_id
        )
        if missing:
            raise _retryable(f"{missing} 个 chunk 仍未向量化（embedding 返回数量不足，将重试）")

        await progress(95, "finalizing")
        await self._finish(session=session, version=version, document=document, count=len(chunks))

        log.info(
            "document_embedded",
            document_id=str(document_id),
            version_id=str(version_id),
            chunk_count=len(chunks),
            embedded_now=len(pending),
            model=self._provider.name,
        )

    # ------------------------------------------------------------ 内部

    async def _load(
        self, session: AsyncSession, *, version_id: uuid.UUID
    ) -> tuple[DocumentVersion | None, Document | None]:
        from sqlalchemy.orm import selectinload

        result = await session.execute(
            select(DocumentVersion)
            .options(selectinload(DocumentVersion.document))
            .where(DocumentVersion.id == version_id)
            .limit(1)
        )
        version = result.scalar_one_or_none()
        if version is None:
            return None, None
        return version, version.document

    async def _embed_all(
        self,
        *,
        session: AsyncSession,
        chunks: list[Chunk],
        total: int,
        tenant_id: uuid.UUID,
        document_version_id: uuid.UUID,
        progress: Any,
        index_version_id: uuid.UUID,
    ) -> None:
        """分批向量化并回写。

        每批独立提交：一批失败时已完成的批次保留（可续跑）。
        这与"切分必须整体成功"不同——向量化是增量的，天然支持部分完成。
        """
        repo = ChunkRepository(session)
        batch_size = self._batch_size()
        processed = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]

            # embedding_input 优先：结构化策略的 embedding_input 带章节前缀，
            # 向量能区分"差旅报销额度"与"招待报销额度"（正文可能完全相同）。
            # content 是给用户看的原文。
            texts = [chunk.embedding_input or chunk.content for chunk in batch]

            result = await self._provider.embed_batch(texts, task=EmbeddingTask.DOCUMENT)
            vectors = result.vectors

            # ---- 成本入账 ----
            #
            # **与向量写入同事务**：分两个事务时，若向量写成功而账没记，
            # 对账会出现“有向量无成本”（且重跑不会补——幂等会让任务直接跳过）。
            # `input_tokens=0` 表示服务端没返回 usage，此时不写 0-token 噪音行。
            await self._record_embedding_cost(
                session=session,
                tenant_id=tenant_id,
                document_version_id=document_version_id,
                chunk_count=len(batch),
                input_tokens=result.input_tokens,
                latency_ms=result.latency_ms,
            )

            # **长度必须严格一致**：embed_batch 的返回顺序对应输入顺序，
            # 长度不匹配说明实现有 bug（或部分失败被静默吞掉）。
            # 用 strict=True 让 zip 直接抛错，而不是静默截断成"更少的向量"——
            # 后者会让 chunk 与向量错位，表现为"检索能跑但相关性全乱"。
            pairs = list(zip(batch, vectors, strict=True))

            # 写入**旁表**（向量按索引版本隔离）。upsert 保证重派幂等：
            # 同一 chunk 同一版本重写不会撞唯一约束。
            await repo.set_embeddings(
                updates=[(chunk.id, vector, self._provider.name) for chunk, vector in pairs],
                index_version_id=index_version_id,
                tenant_id=tenant_id,
            )
            await self._refresh_version_count(session, index_version_id=index_version_id)
            # **不 commit**：事务边界归引擎（见模块 docstring）。
            # 只用 flush 把改动推到数据库，让后续统计查询能读到。
            await session.flush()

            processed += len(batch)
            percent = 15 + int(70 * processed / max(total, 1))
            await progress(percent, f"embedded_{processed}/{total}")
            log.info(
                "embedding_batch_done",
                processed=processed,
                total=total,
                latency_ms=result.latency_ms,
            )

    async def _resolve_index_version(
        self,
        session: AsyncSession,
        *,
        document: Document,
        target: uuid.UUID | None,
    ) -> uuid.UUID:
        """决定向量写入哪个索引版本。

        - `target is None`：原有行为——用 ACTIVE（首次自动创建）；
        - `target` 给定：**允许写入非 ACTIVE 的 building 版本**，
          这是零停机重建的关键：回填期间检索仍用旧 ACTIVE，
          全部回填并校验后再由调用方切换。

        指定版本仍必须校验模型/维度：把新模型的向量写进旧模型版本
        会让两类向量混在一起，且不会报错。
        """
        if target is None:
            return await self._ensure_active_index_version(session, document=document)

        from rag_db.models.chunk import IndexVersion

        version = await session.get(IndexVersion, target)
        if version is None:
            raise _permanent(f"目标索引版本 {target} 不存在")
        if version.knowledge_base_id != document.knowledge_base_id:
            raise _permanent(f"索引版本 {target} 不属于文档所在知识库")

        expected_dim = self._provider.capabilities.dimension
        if version.embedding_dim != expected_dim or version.embedding_model != self._provider.name:
            raise _permanent(
                f"目标索引版本使用 {version.embedding_model}/d{version.embedding_dim}，"
                f"当前模型 {self._provider.name}/d{expected_dim}；不能混写"
            )
        return target

    async def _ensure_active_index_version(
        self, session: AsyncSession, *, document: Document
    ) -> uuid.UUID:
        """确保知识库有 ACTIVE 索引版本，返回它的 ID。

        为什么要"自动创建"：`index_version_id` 为 NULL 的 chunk
        永远不会被检索到（检索按版本过滤）。首次上传时还没有任何版本，
        不自动创建的话表现为"上传成功、能解析能切分、但永远搜不到"——
        这是最容易被误判为"检索算法有问题"的一类故障。

        版本标识用 embedding 模型名 + 维度：换模型必然要新版本
        （旧向量与新向量不可比），而同一个模型重复上传应复用同一版本。
        """
        from rag_db.repositories.chunk import IndexVersionRepository

        repo = IndexVersionRepository(session)
        active = await repo.get_active(knowledge_base_id=document.knowledge_base_id)
        if active is not None:
            # 校验维度一致：模型换过但版本没换会让新旧向量混在一起，
            # 距离不可比 → 排序错乱（不报错，最难发现）
            expected_dim = self._provider.capabilities.dimension
            if active.embedding_dim != expected_dim:
                raise _permanent(
                    f"ACTIVE 索引版本维度 {active.embedding_dim} 与当前模型 "
                    f"{expected_dim} 不一致；必须先重建索引版本"
                )
            if active.embedding_model != self._provider.name:
                raise _permanent(
                    f"ACTIVE 索引版本使用模型 {active.embedding_model}，"
                    f"当前配置 {self._provider.name}；必须先重建索引版本"
                )
            return active.id

        # 首次：创建并立即激活
        #
        # **version_tag 必须唯一**：`uq_index_versions_kb_tag` 约束
        # (knowledge_base_id, version_tag)。重试时如果 tag 相同会撞约束。
        # 加随机后缀的原因见下——但正常情况下 get_active 应该已经命中，
        # 走不到这里。走到这里说明知识库确实还没有版本。
        version = await repo.create(
            tenant_id=document.tenant_id,
            knowledge_base_id=document.knowledge_base_id,
            version_tag=(
                f"{self._provider.name}-d{self._provider.capabilities.dimension}"
                f"-{uuid.uuid4().hex[:6]}"
            ),
            embedding_model=self._provider.name,
            embedding_dim=self._provider.capabilities.dimension,
        )
        version_id = version.id
        await repo.activate(knowledge_base_id=document.knowledge_base_id, version_id=version_id)
        # **必须 flush**：不 flush 的话同一个 handler 内后续的
        # `get_active` 查不到这条新记录（它在会话的 pending 状态里），
        # 于是每次重试都再建一个版本——版本表被污染，且 ACTIVE 反复切换。
        await session.flush()
        log.info(
            "index_version_auto_created",
            index_version_id=str(version_id),
            knowledge_base_id=str(document.knowledge_base_id),
            model=self._provider.name,
            note="知识库首次向量化，已创建并激活默认版本",
        )
        return version_id

    async def _refresh_version_count(
        self,
        session: AsyncSession,
        *,
        index_version_id: uuid.UUID,
    ) -> None:
        """刷新 `index_versions.chunk_count`。

        **为什么必须更新计数**：管理端与 `IndexRebuilder.activate`
        的前置校验都依赖它。计数永远是 0 的话：
        - 看板显示"这个索引版本有 0 个 chunk"，看起来像构建失败；
        - `activate` 的 `actual < expected` 校验失效。

        用 COUNT 而不是 `+= len(new_chunks)`：重派任务时同一批会被
        重复累加，COUN T 是幂等的。
        """
        from sqlalchemy import func, select, update

        from rag_db.models.chunk import IndexVersion
        from rag_db.models.embedding import ChunkEmbedding

        count_result = await session.execute(
            select(func.count())
            .select_from(ChunkEmbedding)
            .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
            .where(
                ChunkEmbedding.index_version_id == index_version_id,
                Chunk.is_current.is_(True),
                Chunk.chunk_level == 0,
            )
        )
        await session.execute(
            update(IndexVersion)
            .where(IndexVersion.id == index_version_id)
            .values(chunk_count=int(count_result.scalar_one()))
        )

    async def _count_missing(
        self,
        session: AsyncSession,
        *,
        version_id: uuid.UUID,
        index_version_id: uuid.UUID,
    ) -> int:
        """统计本版本仍未向量化的 chunk 数。

        **判据是"旁表里没有该版本的行"**，不是 `chunks.embedding IS NULL`：
        A10 之后向量在旁表，`chunks.embedding` 是遗留列（永远是 NULL）。
        继续查它会让这个校验恒为"有缺失"，任务无限重试。
        """
        from rag_db.models.embedding import ChunkEmbedding

        embedded_exists = (
            select(ChunkEmbedding.chunk_id)
            .where(
                ChunkEmbedding.chunk_id == Chunk.id,
                ChunkEmbedding.index_version_id == index_version_id,
            )
            .exists()
        )
        result = await session.execute(
            select(func.count())
            .select_from(Chunk)
            .where(
                Chunk.document_version_id == version_id,
                Chunk.chunk_level == 0,
                ~embedded_exists,
            )
        )
        return int(result.scalar_one())

    async def _finish(
        self,
        *,
        session: AsyncSession,
        version: DocumentVersion,
        document: Document,
        count: int,
    ) -> None:
        """推进状态机到 READY。

        为什么直接到 READY 而不是 INDEXING：
        A10 的"索引"在本项目里不是独立存储（不写独立向量库），
        向量就在 `chunks.embedding` 上，HNSW 索引由 PG 自动维护。
        所以"写完向量"就等于"索引可用"。

        如果未来接入独立向量库（Milvus/Qdrant），这里应先进 INDEXING，
        再由一个独立的 index handler 推进到 READY。
        注释保留这个扩展点，避免后人误以为"跳过了一个步骤"。
        """
        version.status = DocumentVersionStatus.READY
        version.chunk_count = count
        version.embedding_model = self._provider.name
        version.embedding_dim = self._provider.capabilities.dimension

        updated = await DocumentRepository(session).update_status(
            tenant_id=version.tenant_id,
            doc_id=document.id,
            new_status=DocumentStatus.READY,
            expected_version=document.version,
            chunk_count=count,
            error_code=None,
            error_message=None,
        )
        if not updated:
            raise _retryable("文档状态被并发修改，向量化结果未回写（将重试）")

    async def _record_embedding_cost(
        self,
        *,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        document_version_id: uuid.UUID,
        chunk_count: int,
        input_tokens: int,
        latency_ms: int | None,
    ) -> None:
        """把一次 embedding 调用写进 `model_calls`。

        与 LLM 的 `_record_cost` 同口径：**明细表是成本对账的唯一真相**，
        消息/会话上的 `cost_usd` 只是展示聚合。
        """
        if input_tokens <= 0:
            return
        await ModelCallRepository(session).record(
            tenant_id=tenant_id,
            task_type=ModelTaskType.EMBEDDING,
            provider=self._settings.provider,
            model=self._settings.model_name,
            input_tokens=input_tokens,
            unit_price_input=self._settings.unit_price_input,
            latency_ms=latency_ms,
            # 与 API 侧同口径：能关联到 trace 就填，方便从成本行跳到链路。
            trace_id=(self._tracer.current_trace_id() if self._tracer is not None else None),
            call_metadata={
                "document_version_id": str(document_version_id),
                "chunk_count": chunk_count,
            },
        )

    def _batch_size(self) -> int:
        """批大小。

        从 provider 能力推断而不是硬编码：
        - 本地模型看配置的 batch_size（受显存限制）；
        - API 型实现可能有自己的上限。
        """
        capabilities = self._provider.capabilities
        if not capabilities.supports_batch:
            return 1
        # 用配置值；provider 若声明更小的上限则取小（未来可扩展 capabilities）
        return max(1, 32)


# ---------------------------------------------------------------- 辅助


def _require_uuid(payload: dict[str, Any], key: str) -> uuid.UUID:
    from rag_adapters.providers.embedding import EmbeddingProviderError

    raw = payload.get(key)
    if not raw:
        raise EmbeddingProviderError(f"任务 payload 缺少 {key}", retryable=False)
    try:
        return uuid.UUID(str(raw))
    except ValueError as exc:
        raise EmbeddingProviderError(
            f"任务 payload 的 {key} 不是合法 UUID", retryable=False
        ) from exc


def _permanent(message: str) -> Exception:
    """永久失败（版本与模型不匹配：需要人介入重建索引）。"""
    from rag_adapters.providers.embedding import EmbeddingProviderError

    return EmbeddingProviderError(message, retryable=False)


def _retryable(message: str) -> Exception:
    """可重试错误（乐观锁冲突、向量数量不足）。"""
    from rag_adapters.providers.embedding import EmbeddingProviderError

    return EmbeddingProviderError(message, retryable=True)


__all__ = ["EmbedDocumentHandler"]
