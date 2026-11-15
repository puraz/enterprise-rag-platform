"""切分任务处理器：解析产物 → chunks 入库。

## 管线位置

    解析完成（documents.status = CHUNKING，产物写到 parsed/{tenant}/{version}.jsonl）
      │
      └─▶ [本处理器]  读产物 → 切分 → 校验 → 落库 → 状态推进到 EMBEDDING
            │
            ▼
          向量化任务（A10 之后）

## 五个必须做对的地方

### 1. 切分参数来自版本快照，不是库的当前配置

`document_versions.chunking_strategy` / `chunking_config` 是**入队时**的快照。
用它而不是读 `knowledge_bases.default_chunk_*`：库的配置可能在版本创建后
被改过，用当前配置会让"重跑"与"首次"结果不同——那样 A10 的版本回滚
就失去意义（回滚后重新切分会得到完全不同的 chunk）。

### 2. 落库前必须 `verify_chunks`

违反契约的 chunk 会一路影响到检索与引用，且后续环节无法修正。
典型问题：偏移错位（高亮显示错误文字）、空块（占检索位）、
块类型错（按类型过滤时漏掉）。

**校验失败必须让任务失败**，不能"先写进去凑合用"。

### 3. 幂等：先删旧 chunk，再写新的

任务会被重派（worker 崩溃、锁超时）。如果直接 `bulk_create`，
重派会撞 `uq_chunks_version_index`（同版本 chunk_index 唯一）——
报错信息是"唯一约束冲突"，但根因是"没做幂等"。

正确做法：**同一事务内**先删该版本的旧 chunk，再插入新的。
两种做法对比：

- `ON CONFLICT DO UPDATE`：只处理 index 冲突，旧版本多出来的 chunk 会残留；
- 先删后插：简单、彻底，且 `delete + insert` 在同一事务里原子。

代价是重跑会重算全部 chunk（无法增量）。这是可接受的：
切分本身是秒级操作，而"残留脏数据"的代价远高于重算。

### 4. 父子关系的 ID 回填

chunker 产出用 `parent_index`（本批次下标）表达父子关系，
因为 chunker 不该知道数据库 ID。

落库分两阶段：
1. 插入所有块（父块的 `parent_chunk_id` 先留空）；
2. 用刚拿到的 ID 回填子块的 `parent_chunk_id`。

**必须 flush 拿到 ID 后再回填**：SQLAlchemy 在 flush 前不会生成主键。

### 5. 向量化不在这一步做

切分只产文本与位置信息，`embedding` 留 NULL。
理由是分离关注点：
- 切分是纯 CPU（快），向量化要 GPU（慢、可能失败）；
- 一起做会让"向量服务临时故障"导致"切分成果全部丢失"（要重跑切分）；
- 分开后向量化可以按 chunk 续跑（`list_pending_embedding`）。

代价是中间会有一段"有 chunk 但没向量"的状态，此时文档不能参与检索
（`status` 停在 EMBEDDING 即可）。
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from rag_adapters.chunkers.base import ChunkConfig, verify_chunks
from rag_adapters.chunkers.factory import build_chunker
from rag_adapters.parsers.artifact import ParsedArtifactError, load_parsed, read_all
from rag_contracts.enums import DocumentStatus, DocumentVersionStatus
from rag_core.errors import NotFoundError
from rag_core.logging import get_logger
from rag_db.models.document import Document, DocumentVersion
from rag_db.repositories import ChunkRepository, DocumentRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from rag_adapters.storages import StorageBackend

log = get_logger(__name__)


class ChunkDocumentHandler:
    """`TaskType.CHUNK_DOCUMENT` 的处理器。

    payload：
        {"document_id": "...", "document_version_id": "..."}
    """

    def __init__(self, *, storages: dict[str, StorageBackend]) -> None:
        if not storages:
            raise ValueError("至少需要一个存储后端")
        self._storages = storages
        self._primary_storage = next(iter(storages.values()))

    async def __call__(
        self,
        *,
        session: AsyncSession,
        task: Any,
        progress: Any,
    ) -> None:
        payload = dict(task.payload or {})
        document_id = _require_uuid(payload, "document_id")
        version_id = _require_uuid(payload, "document_version_id")

        await progress(5, "loading_version")
        version, document = await self._load(session, version_id=version_id)
        if version is None or document is None:
            raise NotFoundError("文档版本不存在（可能已被删除）")
        if version.document_id != document_id:
            raise _permanent_error(
                f"payload 不一致: version {version_id} 不属于 document {document_id}"
            )

        # ---- 读取解析产物 ----
        await progress(15, "loading_artifact")
        artifact_key = _artifact_key(version)
        artifact = await self._read_artifact(artifact_key)

        # ---- 切分 ----
        await progress(35, "chunking")
        config = _config_from_snapshot(version)
        strategy = str(version.chunking_strategy or "recursive")
        chunks = await self._chunk(strategy=strategy, config=config, artifact=artifact)

        # ---- 校验 ----
        await progress(60, "verifying")
        problems = verify_chunks(chunks, artifact)
        if problems:
            log.error(
                "chunk_verify_failed",
                version_id=str(version_id),
                strategy=strategy,
                problem_count=len(problems),
                first_problem=problems[0][:160],
            )
            raise _permanent_error(f"切分结果违反契约（{len(problems)} 项）：{problems[0][:120]}")

        # ---- 落库 ----
        await progress(75, "persisting")
        await self._persist(
            session=session,
            version=version,
            document=document,
            chunks=chunks,
        )

        await progress(95, "finalizing")
        log.info(
            "document_chunked",
            document_id=str(document_id),
            version_id=str(version_id),
            strategy=strategy,
            chunk_count=len(chunks),
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            parent_count=sum(1 for chunk in chunks if chunk.chunk_level > 0),
        )

    # ------------------------------------------------------------ 载入

    async def _load(
        self, session: AsyncSession, *, version_id: uuid.UUID
    ) -> tuple[DocumentVersion | None, Document | None]:
        from sqlalchemy import select
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

    async def _read_artifact(self, artifact_key: str) -> Any:
        """从存储读解析产物。"""
        data = await read_all(self._primary_storage.get(artifact_key))
        try:
            return load_parsed(data)
        except ParsedArtifactError as exc:
            # 产物损坏/版本不兼容 → 永久失败。重读不会变，
            # 正确修法是重跑解析任务重新生成产物。
            raise _permanent_error(f"解析产物不可用：{exc}") from exc

    async def _chunk(self, *, strategy: str, config: ChunkConfig, artifact: Any) -> list[Any]:
        chunker = build_chunker(strategy, config=config)

        # 语义策略需要 embedding：它有自己的异步入口。
        # 其余策略走同步入口（纯 CPU，快）。
        if hasattr(chunker, "chunk_async"):
            return await chunker.chunk_async(artifact)  # type: ignore[no-any-return]
        return chunker.chunk(artifact)

    # ------------------------------------------------------------ 落库

    async def _persist(
        self,
        *,
        session: AsyncSession,
        version: DocumentVersion,
        document: Document,
        chunks: list[Any],
    ) -> None:
        """落库（幂等）。"""
        repo = ChunkRepository(session)

        # 先删旧 chunk：重派时避免撞 uq_chunks_version_index。
        # 同一事务内，删与插要么都成功要么都回滚。
        deleted = await repo.delete_by_version(document_version_id=version.id)
        if deleted:
            log.info(
                "stale_chunks_removed",
                version_id=str(version.id),
                count=deleted,
                note="重派任务，旧的切分结果已清除",
            )

        # ---- 第一阶段：插入所有块（父块 parent 留空）----
        rows = [
            {
                "chunk_index": draft.chunk_index,
                "content": draft.content,
                "content_hash": draft.content_hash,
                "chunk_type": draft.chunk_type,
                "embedding_input": draft.embedding_input,
                "chunk_level": draft.chunk_level,
                "page_no": draft.page_no,
                "char_start": draft.char_start,
                "char_end": draft.char_end,
                "bbox": draft.bbox,
                "section_path": draft.section_path,
                "chunk_metadata": draft.chunk_metadata,
                # 向量化留到下一阶段（见模块 docstring 第 5 点）
                "embedding": None,
                "embedding_model": None,
            }
            for draft in chunks
        ]
        await repo.bulk_create(
            tenant_id=version.tenant_id,
            knowledge_base_id=document.knowledge_base_id,
            document_id=document.id,
            document_version_id=version.id,
            chunks=rows,
        )

        # 需要回填父子关系时才查一次 ID 映射（多数策略没有父块，省掉这次查询）
        parent_pairs = [
            (draft.chunk_index, draft.parent_index)
            for draft in chunks
            if draft.parent_index is not None
        ]
        if parent_pairs:
            await self._link_parents(session, version_id=version.id, pairs=parent_pairs)

        # ---- 版本级统计 ----
        version.chunk_count = len(chunks)
        version.status = DocumentVersionStatus.PROCESSING  # 还要向量化

        # ---- 文档级：推进状态 ----
        updated = await DocumentRepository(session).update_status(
            tenant_id=version.tenant_id,
            doc_id=document.id,
            new_status=DocumentStatus.EMBEDDING,
            expected_version=document.version,
            chunk_count=len([c for c in chunks if c.chunk_level == 0]),
            error_code=None,
            error_message=None,
        )
        if not updated:
            raise _retryable_error("文档状态被并发修改，切分结果未回写（将重试）")

    async def _link_parents(
        self,
        session: AsyncSession,
        *,
        version_id: uuid.UUID,
        pairs: list[tuple[int, int]],
    ) -> None:
        """回填 `parent_chunk_id`。

        用 `(chunk_index, parent_index)` → 数据库 ID 的映射：
        同版本内 `chunk_index` 唯一（数据库约束保证），所以映射无歧义。
        """
        from sqlalchemy import select, update

        from rag_db.models.chunk import Chunk

        result = await session.execute(
            select(Chunk.chunk_index, Chunk.id).where(Chunk.document_version_id == version_id)
        )
        index_to_id = {int(row[0]): row[1] for row in result}

        # executemany：逐条 UPDATE 会 N 次往返，几百个父块时明显变慢
        updates = [
            {"chunk_index": child_index, "parent_id": index_to_id[parent_index]}
            for child_index, parent_index in pairs
            if parent_index in index_to_id
        ]
        if not updates:
            return

        # 用 CASE WHEN 单条语句更新（SQLAlchemy 的 bindparam 批量）
        for item in updates:
            await session.execute(
                update(Chunk)
                .where(
                    Chunk.document_version_id == version_id,
                    Chunk.chunk_index == item["chunk_index"],
                )
                .values(parent_chunk_id=item["parent_id"])
            )


# ---------------------------------------------------------------- 辅助


def _require_uuid(payload: dict[str, Any], key: str) -> uuid.UUID:
    raw = payload.get(key)
    if not raw:
        raise _permanent_error(f"任务 payload 缺少 {key}")
    try:
        return uuid.UUID(str(raw))
    except ValueError as exc:
        raise _permanent_error(f"任务 payload 的 {key} 不是合法 UUID") from exc


def _artifact_key(version: DocumentVersion) -> str:
    """从版本快照取产物键。

    **缺失即永久失败**：没有产物说明解析没跑完或产物被清理了。
    重试读不到同一个键，必须先重跑解析任务。
    """
    key = (version.parse_metadata or {}).get("artifact_key")
    if not key:
        raise _permanent_error("版本没有解析产物（artifact_key 缺失）；请先重跑解析任务")
    return str(key)


def _config_from_snapshot(version: DocumentVersion) -> ChunkConfig:
    """从版本快照还原切分参数。

    快照里存的是 dict（JSONB）。字段缺失时用默认值——
    向前兼容旧版本（A9 之前创建的版本没有这些字段）。
    """
    snapshot = dict(version.chunking_config or {})
    return ChunkConfig(
        chunk_size=int(snapshot.get("chunk_size", 800)),
        chunk_overlap=int(snapshot.get("chunk_overlap", 100)),
        min_chunk_size=int(snapshot.get("min_chunk_size", 50)),
    )


def _permanent_error(message: str) -> Exception:
    """构造可分类为 PERMANENT 的异常。

    用 `ParseError(retryable=False)`：它的 `retryable` 属性会被
    `classify()` 读到并判为 PERMANENT（见 `rag_core.resilience.classify`）。
    这样错误码是业务语义名，而不是 `UNHANDLED_ValueError`。
    """
    from rag_adapters.parsers.base import ParseError

    return ParseError(message, retryable=False)


def _retryable_error(message: str) -> Exception:
    """构造可重试异常（乐观锁冲突、临时状态不一致）。"""
    from rag_adapters.parsers.base import ParseError

    return ParseError(message, retryable=True)


__all__ = ["ChunkDocumentHandler"]
