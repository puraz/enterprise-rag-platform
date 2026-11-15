"""切分 handler 端到端测试（真实 PG + 真实存储 + 真实解析产物）。

## 验证什么

单测（`test_chunkers.py`）证明了切分逻辑正确。这里验证**管线正确**：

    解析产物（JSONL）→ 读产物 → 切分 → 校验 → 落库 → 状态推进

任何一环的字段名/签名不匹配都在这条链上暴露。

## 重点验证

1. **幂等**：同一任务跑两遍不产生重复 chunk（撞唯一约束或残留脏数据）
2. **父子关系落库**：`parent_chunk_id` 真的填上了，且指向同版本的父块
3. **状态推进**：文档从 CHUNKING → EMBEDDING
4. **向量化未做**：`embedding` 全是 NULL（这是 A10 的事）
5. **参数来自版本快照**：不是读库的当前配置
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from rag_contracts.enums import DocumentStatus
from rag_core.settings import get_settings
from rag_worker.handlers.chunk import ChunkDocumentHandler

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------- fixture


@pytest.fixture
async def env() -> AsyncIterator[tuple[Any, Any, uuid.UUID, Path]]:
    """独立引擎 + 临时存储 + 租户。"""
    import shutil
    import tempfile

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from rag_adapters.storages.local import LocalStorageBackend
    from rag_core.settings.storage import StorageSettings

    settings = get_settings()
    db = settings.database
    user = db.migration_user or db.user
    password = (
        db.migration_password.get_secret_value()
        if db.migration_password is not None
        else db.password.get_secret_value()
    )
    dsn = f"postgresql+asyncpg://{user}:{password}@{db.host}:{db.port}/{db.name}"

    engine = create_async_engine(dsn, pool_size=4)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    storage_root = Path(tempfile.mkdtemp(prefix="rag-chunk-test-"))
    storage = LocalStorageBackend(StorageSettings(backend="local", local_root=str(storage_root)))

    tenant_id = uuid.uuid4()
    await _create_tenant(factory, tenant_id=tenant_id)
    try:
        yield factory, {"local": storage}, tenant_id, storage_root
    finally:
        await _cleanup(factory, tenant_id=tenant_id)
        await engine.dispose()
        shutil.rmtree(storage_root, ignore_errors=True)


# ---------------------------------------------------------------- 主流程


async def test_chunk_document_end_to_end(env: tuple[Any, Any, uuid.UUID, Path]) -> None:
    """端到端：产物 → 切分 → 落库 → 状态推进。"""
    factory, storages, tenant_id, _ = env

    document_id, version_id, _artifact_key = await _create_document_with_artifact(
        factory,
        storages["local"],
        tenant_id=tenant_id,
        blocks=[
            ("第一章 总则", "title", 1),
            ("本制度适用于公司全体员工。制度的解释权归人力资源部。", "text", 0),
            ("第二章 差旅", "title", 1),
            ("员工出差前须提交申请，经上级审批后方可预订。", "text", 0),
        ],
        strategy="structured",
        config={"chunk_size": 200, "chunk_overlap": 20, "min_chunk_size": 10},
    )

    handler = ChunkDocumentHandler(storages=storages)
    task = _make_task(
        tenant_id=tenant_id,
        payload={"document_id": str(document_id), "document_version_id": str(version_id)},
    )

    progress_calls: list[int] = []

    async def progress(percent: int, stage: str = "") -> None:
        progress_calls.append(percent)

    async with factory() as session:
        await handler(session=session, task=task, progress=progress)
        await session.commit()

    # ---- 断言 1：chunk 落库 ----
    async with factory() as session:
        row = await session.execute(
            text(
                "SELECT count(*), count(*) FILTER (WHERE embedding IS NOT NULL) "
                "FROM chunks WHERE document_version_id = :vid"
            ),
            {"vid": version_id},
        )
        total, with_embedding = row.one()
    assert total > 0, "没有 chunk 落库"
    assert with_embedding == 0, "切分阶段不该做向量化（那是 A10 的事）"

    # ---- 断言 2：位置信息完整 ----
    async with factory() as session:
        row = await session.execute(
            text(
                "SELECT content, char_start, char_end, section_path, content_hash "
                "FROM chunks WHERE document_version_id = :vid ORDER BY chunk_index LIMIT 1"
            ),
            {"vid": version_id},
        )
        content, char_start, char_end, section_path, content_hash = row.one()
    assert char_start is not None and char_end is not None
    assert char_end > char_start
    assert content.strip(), "chunk 内容不能为空"
    # content_hash 必须是 content 的真实 SHA-256（用于去重与缓存 key）
    import hashlib

    assert content_hash == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert section_path, "结构化策略应给出章节路径"

    # ---- 断言 3：文档状态推进到 EMBEDDING ----
    async with factory() as session:
        row = await session.execute(
            text("SELECT status, chunk_count FROM documents WHERE id = :id"),
            {"id": document_id},
        )
        status, chunk_count = row.one()
    assert status == DocumentStatus.EMBEDDING.value
    assert chunk_count == total

    # ---- 断言 4：进度按里程碑上报，没有 100 ----
    assert progress_calls
    assert 100 not in progress_calls


async def test_chunk_is_idempotent(env: tuple[Any, Any, uuid.UUID, Path]) -> None:
    """同一任务跑两遍不产生重复 chunk。

    任务会被重派（worker 崩溃、锁超时）。如果 handler 不幂等：
    - 撞 `uq_chunks_version_index`（同版本 chunk_index 唯一）→ 报错；
    - 或产生重复内容 → 检索结果里出现两条一样的引用。
    """
    factory, storages, tenant_id, _ = env

    document_id, version_id, _ = await _create_document_with_artifact(
        factory,
        storages["local"],
        tenant_id=tenant_id,
        blocks=[
            ("第一章 总则", "title", 1),
            ("本制度适用于公司全体员工。", "text", 0),
        ],
        strategy="recursive",
        config={"chunk_size": 200, "chunk_overlap": 20, "min_chunk_size": 10},
    )

    handler = ChunkDocumentHandler(storages=storages)
    task = _make_task(
        tenant_id=tenant_id,
        payload={"document_id": str(document_id), "document_version_id": str(version_id)},
    )

    async def progress(percent: int, stage: str = "") -> None:
        return None

    counts: list[int] = []
    for _ in range(2):
        async with factory() as session:
            await handler(session=session, task=task, progress=progress)
            await session.commit()
        async with factory() as session:
            row = await session.execute(
                text("SELECT count(*) FROM chunks WHERE document_version_id = :vid"),
                {"vid": version_id},
            )
            counts.append(int(row.scalar_one()))

    assert counts[0] > 0
    assert counts[0] == counts[1], f"重跑后 chunk 数量变了：{counts}"


async def test_parent_child_relationships_are_persisted(
    env: tuple[Any, Any, uuid.UUID, Path],
) -> None:
    """父子策略的 `parent_chunk_id` 必须真的落库。

    chunker 只知道"父块在本批次的下标"，数据库需要真实 UUID。
    这个映射由 handler 分两阶段完成（插块 → 回填）。
    """
    factory, storages, tenant_id, _ = env

    document_id, version_id, _ = await _create_document_with_artifact(
        factory,
        storages["local"],
        tenant_id=tenant_id,
        blocks=[
            ("第一段：公司差旅制度适用于全体员工，需要严格遵守相关规定。", "text", 0),
            ("第二段：出差前须提交申请，经上级审批后方可预订机票与酒店。", "text", 0),
            ("第三段：住宿标准按城市等级划分，一线城市上限六百元。", "text", 0),
            ("第四段：报销须在费用发生后三十日内提交，附发票原件。", "text", 0),
        ],
        strategy="parent_child",
        config={"chunk_size": 60, "chunk_overlap": 10, "min_chunk_size": 10},
    )

    handler = ChunkDocumentHandler(storages=storages)
    task = _make_task(
        tenant_id=tenant_id,
        payload={"document_id": str(document_id), "document_version_id": str(version_id)},
    )

    async def progress(percent: int, stage: str = "") -> None:
        return None

    async with factory() as session:
        await handler(session=session, task=task, progress=progress)
        await session.commit()

    async with factory() as session:
        # 父块存在
        row = await session.execute(
            text(
                "SELECT count(*) FROM chunks WHERE document_version_id = :vid AND chunk_level > 0"
            ),
            {"vid": version_id},
        )
        parent_count = int(row.scalar_one())

        # 子块指向的父块必须真实存在，且属于同一版本
        row = await session.execute(
            text(
                "SELECT count(*) FROM chunks c "
                "JOIN chunks p ON c.parent_chunk_id = p.id "
                "WHERE c.document_version_id = :vid AND c.chunk_level = 0 "
                "AND p.document_version_id = :vid AND p.chunk_level > 0"
            ),
            {"vid": version_id},
        )
        linked = int(row.scalar_one())

    assert parent_count > 0, "没有父块落库"
    assert linked > 0, "没有任何子块正确关联到父块"


async def test_chunk_uses_version_snapshot_not_kb_default(
    env: tuple[Any, Any, uuid.UUID, Path],
) -> None:
    """切分参数必须用版本快照，不是知识库的当前默认值。

    库的配置可能在版本创建后被改过。用当前配置会让"重跑"
    与"首次"结果不同——A10 的版本回滚就失去意义。
    """
    factory, storages, tenant_id, _ = env

    # 版本快照用小 chunk_size
    document_id, version_id, _ = await _create_document_with_artifact(
        factory,
        storages["local"],
        tenant_id=tenant_id,
        blocks=[("甲" * 500, "text", 0)],
        strategy="recursive",
        config={"chunk_size": 100, "chunk_overlap": 10, "min_chunk_size": 20},
    )

    # 把知识库默认值改成完全不同的（大 chunk_size）
    async with factory() as session:
        await session.execute(
            text(
                "UPDATE knowledge_bases SET default_chunk_size = 5000, "
                "default_chunk_overlap = 0 WHERE tenant_id = :tid"
            ),
            {"tid": tenant_id},
        )
        await session.commit()

    handler = ChunkDocumentHandler(storages=storages)
    task = _make_task(
        tenant_id=tenant_id,
        payload={"document_id": str(document_id), "document_version_id": str(version_id)},
    )

    async def progress(percent: int, stage: str = "") -> None:
        return None

    async with factory() as session:
        await handler(session=session, task=task, progress=progress)
        await session.commit()

    async with factory() as session:
        row = await session.execute(
            text(
                "SELECT count(*), max(length(content)) FROM chunks WHERE document_version_id = :vid"
            ),
            {"vid": version_id},
        )
        count, max_length = row.one()

    # 用快照的 100 → 500 字符切出多块；若误用库默认 5000 → 只有 1 块
    assert int(count) > 1, f"应切成多块（快照 size=100），实际 {count} 块"
    assert int(max_length) <= 100, f"块超长（{max_length}），说明没用快照参数"


async def test_missing_artifact_fails_permanently(
    env: tuple[Any, Any, uuid.UUID, Path],
) -> None:
    """没有解析产物时永久失败（重试读不到，必须先重跑解析）。"""
    from rag_adapters.parsers.base import ParseError

    factory, storages, tenant_id, _ = env

    # 建文档但不写产物
    document_id, version_id = await _create_document(
        factory, tenant_id=tenant_id, with_artifact=False
    )

    handler = ChunkDocumentHandler(storages=storages)
    task = _make_task(
        tenant_id=tenant_id,
        payload={"document_id": str(document_id), "document_version_id": str(version_id)},
    )

    async def progress(percent: int, stage: str = "") -> None:
        return None

    async with factory() as session:
        with pytest.raises(ParseError) as exc_info:
            await handler(session=session, task=task, progress=progress)

    assert exc_info.value.retryable is False, "缺少产物不该重试"


async def test_corrupt_artifact_fails_permanently(
    env: tuple[Any, Any, uuid.UUID, Path],
) -> None:
    """产物损坏时永久失败（不是可重试错误）。"""
    from rag_adapters.parsers.base import ParseError

    factory, storages, tenant_id, _ = env

    document_id, version_id = await _create_document(
        factory, tenant_id=tenant_id, with_artifact=True
    )

    # 把产物覆盖成垃圾
    async with factory() as session:
        row = await session.execute(
            text("SELECT parse_metadata->>'artifact_key' FROM document_versions WHERE id = :id"),
            {"id": version_id},
        )
        artifact_key = row.scalar_one()
    await storages["local"].put(artifact_key, b"not a valid artifact\n")

    handler = ChunkDocumentHandler(storages=storages)
    task = _make_task(
        tenant_id=tenant_id,
        payload={"document_id": str(document_id), "document_version_id": str(version_id)},
    )

    async def progress(percent: int, stage: str = "") -> None:
        return None

    async with factory() as session:
        with pytest.raises(ParseError) as exc_info:
            await handler(session=session, task=task, progress=progress)

    assert exc_info.value.retryable is False


# ---------------------------------------------------------------- 辅助


@dataclass
class _Task:
    id: uuid.UUID
    tenant_id: uuid.UUID
    task_type: str
    payload: dict[str, str]


def _make_task(*, tenant_id: uuid.UUID, payload: dict[str, str]) -> Any:
    return _Task(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        task_type="chunk_document",
        payload=payload,
    )


async def _create_tenant(factory: Any, *, tenant_id: uuid.UUID) -> None:
    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:id, :slug, :name, 'active', now(), now())"
            ),
            {"id": tenant_id, "slug": f"c-{tenant_id.hex[:8]}", "name": "切分测试租户"},
        )
        await session.commit()


async def _create_document(
    factory: Any,
    *,
    tenant_id: uuid.UUID,
    with_artifact: bool,
) -> tuple[uuid.UUID, uuid.UUID]:
    """建最小文档链（不带产物），返回 `(document_id, version_id)`。"""
    import hashlib

    from rag_contracts.enums import DocumentFormat
    from rag_db.repositories import (
        BlobRepository,
        DocumentRepository,
        KnowledgeBaseRepository,
    )

    user_id = uuid.uuid4()
    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "created_at, updated_at) "
                "VALUES (:id, :email, :name, 'active', 'none', now(), now())"
            ),
            {
                "id": user_id,
                "email": f"{user_id.hex[:8]}@test.local",
                "name": "切分测试用户",
            },
        )
        kb = await KnowledgeBaseRepository(session).create(
            tenant_id=tenant_id, name=f"kb-{uuid.uuid4().hex[:8]}", created_by=user_id
        )
        content = b"placeholder content"
        digest = hashlib.sha256(content).hexdigest()
        storage_key = f"blobs/{tenant_id}/{digest[:2]}/{digest}"
        blob, _ = await BlobRepository(session).get_or_create(
            tenant_id=tenant_id,
            content_hash=digest,
            size_bytes=len(content),
            storage_key=storage_key,
            storage_backend="local",
            content_type="application/pdf",
        )
        doc = await DocumentRepository(session).create(
            tenant_id=tenant_id,
            knowledge_base_id=kb.id,
            title="chunk-test.pdf",
            original_filename="chunk-test.pdf",
            format=DocumentFormat.PDF.value,
            created_by=user_id,
        )
        repo = DocumentRepository(session)
        version_no = await repo.allocate_version_no(doc_id=doc.id)
        version = await repo.create_version(
            tenant_id=tenant_id,
            document_id=doc.id,
            version_no=version_no,
            blob_id=blob.id,
            created_by=user_id,
            chunking_strategy="recursive",
            chunking_config={"chunk_size": 200, "chunk_overlap": 20},
        )
        doc.status = DocumentStatus.CHUNKING

        # 只登记产物键；**产物本体由调用方写进存储**
        # （这个函数拿不到 storage，且"缺产物"测试正需要键存在但对象不存在）。
        if with_artifact:
            artifact_key = f"parsed/{tenant_id}/{version.id}.jsonl"
            version.parse_metadata = {"artifact_key": artifact_key}

        await session.commit()
        return doc.id, version.id


async def _create_document_with_artifact(
    factory: Any,
    storage: Any,
    *,
    tenant_id: uuid.UUID,
    blocks: list[tuple[str, str, int]],
    strategy: str,
    config: dict[str, int],
) -> tuple[uuid.UUID, uuid.UUID, str]:
    """建完整文档链并写入解析产物。

    返回 `(document_id, version_id, artifact_key)`。
    """
    import hashlib

    from rag_adapters.parsers.artifact import (
        BlockRecord,
        PageRecord,
        dump_parsed,
    )
    from rag_contracts.enums import DocumentFormat
    from rag_db.repositories import (
        BlobRepository,
        DocumentRepository,
        KnowledgeBaseRepository,
    )

    # 构造源文本与块（分隔符 "\n\n"，与真实解析器一致）
    parts: list[str] = []
    records: list[BlockRecord] = []
    cursor = 0
    for block_text, block_type, level in blocks:
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(block_text)
        cursor += len(block_text)
        records.append(
            BlockRecord(
                text=block_text,
                block_type=block_type,
                char_start=start,
                char_end=cursor,
                page_no=1,
                level=level,
            )
        )
    source_text = "".join(parts)

    artifact_bytes = dump_parsed(
        text=source_text,
        blocks=records,
        pages=[
            PageRecord(page_no=1, width=595, height=842, char_start=0, char_end=len(source_text))
        ],
        parser_name="test",
        document_format="pdf",
        has_scanned_pages=False,
    )

    user_id = uuid.uuid4()
    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "created_at, updated_at) "
                "VALUES (:id, :email, :name, 'active', 'none', now(), now())"
            ),
            {
                "id": user_id,
                "email": f"{user_id.hex[:8]}@test.local",
                "name": "切分测试用户",
            },
        )
        kb = await KnowledgeBaseRepository(session).create(
            tenant_id=tenant_id, name=f"kb-{uuid.uuid4().hex[:8]}", created_by=user_id
        )
        digest = hashlib.sha256(artifact_bytes).hexdigest()
        storage_key = f"blobs/{tenant_id}/{digest[:2]}/{digest}"
        blob, _ = await BlobRepository(session).get_or_create(
            tenant_id=tenant_id,
            content_hash=digest,
            size_bytes=len(artifact_bytes),
            storage_key=storage_key,
            storage_backend="local",
            content_type="application/pdf",
        )
        doc = await DocumentRepository(session).create(
            tenant_id=tenant_id,
            knowledge_base_id=kb.id,
            title="chunk-test.pdf",
            original_filename="chunk-test.pdf",
            format=DocumentFormat.PDF.value,
            created_by=user_id,
        )
        repo = DocumentRepository(session)
        version_no = await repo.allocate_version_no(doc_id=doc.id)
        version = await repo.create_version(
            tenant_id=tenant_id,
            document_id=doc.id,
            version_no=version_no,
            blob_id=blob.id,
            created_by=user_id,
            chunking_strategy=strategy,
            chunking_config=config,
        )
        doc.status = DocumentStatus.CHUNKING

        artifact_key = f"parsed/{tenant_id}/{version.id}.jsonl"
        version.parse_metadata = {"artifact_key": artifact_key}

        await session.commit()
        doc_id, version_id = doc.id, version.id

    # 产物写进存储（事务外，因为是对象存储而非数据库）
    await storage.put(artifact_key, artifact_bytes, content_type="application/x-ndjson")
    return doc_id, version_id, artifact_key


async def _cleanup(factory: Any, *, tenant_id: uuid.UUID) -> None:
    """按外键顺序清理。"""
    async with factory() as session:
        await session.execute(text("DELETE FROM chunks WHERE tenant_id = :tid"), {"tid": tenant_id})
        await session.execute(
            text("DELETE FROM document_versions WHERE tenant_id = :tid"), {"tid": tenant_id}
        )
        await session.execute(
            text("DELETE FROM documents WHERE tenant_id = :tid"), {"tid": tenant_id}
        )
        await session.execute(
            text("DELETE FROM knowledge_bases WHERE tenant_id = :tid"), {"tid": tenant_id}
        )
        await session.execute(text("DELETE FROM blobs WHERE tenant_id = :tid"), {"tid": tenant_id})
        await session.execute(text("DELETE FROM users WHERE email LIKE '%@test.local'"))
        await session.execute(text("DELETE FROM tenants WHERE id = :tid"), {"tid": tenant_id})
        await session.commit()
