"""索引层测试：旁表向量、版本切换、零停机重建。

## 分三层

**单测**（无外部依赖）：`IndexRebuilder` 的决策逻辑——
未回填完成不许切换、回滚要求版本属于同一知识库。

**集成**（真实 PG）：旁表的唯一约束与级联、检索按版本过滤、
`ON CONFLICT` 幂等写入。

**演练**（真实 PG + 真并发）：`scripts/drill_index_rebuild.py`
已验证零停机；这里补一个精简版作为回归测试。

## 为什么"未完成回填就切换"必须被拒绝

旁表方案下，回填期间新版本只有部分 chunk 的向量。
提前切换 → 查询只能看到这部分 → **静默丢失内容**。
用户会得到"知识库里查不到这份文档"，而文档明明在库里。

这比"切换失败"严重得多：失败会报错，静默丢失只会让人怀疑检索算法。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy import text

from rag_contracts.enums import DocumentStatus
from rag_core.settings import get_settings
from rag_worker.indexing import IndexRebuilder, _RebuildError

pytestmark = pytest.mark.integration


# ==================================================== 测试替身


class _FakeProvider:
    """假 embedding provider：确定性向量，不加载模型。

    必须实现真实接口（name / capabilities / embed_batch），
    这样 `IndexRebuilder` 走的是与生产相同的代码路径。
    """

    name = "test-fake-embed"

    class _Caps:
        dimension = 1024

    capabilities = _Caps()

    async def embed_batch(self, texts: list[str], *, task: object = None) -> Any:
        class _Result:
            def __init__(self, vectors: list[list[float]]) -> None:
                self.vectors = vectors
                self.latency_ms = 0

        # 用文本长度做种子：确定性且不依赖随机数
        return _Result([_unit_vector(len(t)) for t in texts])


def _unit_vector(seed: int) -> list[float]:
    import math

    values = [((seed * (i + 7)) % 97) / 97.0 - 0.5 for i in range(1024)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


# ==================================================== 单测：决策逻辑


async def test_activate_rejects_incomplete_backfill() -> None:
    """回填未完成时切换必须被拒绝。

    这是最重要的安全阀：字面意义上防止"检索静默丢内容"。
    """
    # 用假 session 隔离逻辑（不需要真实数据库）
    rebuilder = IndexRebuilder(provider=_FakeProvider())  # type: ignore[arg-type]

    @dataclass
    class _FakeVersion:
        id: uuid.UUID
        knowledge_base_id: uuid.UUID
        expected_chunk_count: int

    version = _FakeVersion(
        id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
        # 预期 100 个，但 _count_version_chunks 会返回 0（空库）
        expected_chunk_count=100,
    )

    class _FakeSession:
        async def execute(self, *args: object, **kwargs: object) -> Any:
            class _R:
                def scalar_one(self) -> int:
                    return 0

            return _R()

        async def commit(self) -> None:
            return None

    with pytest.raises(_RebuildError, match="回填未完成"):
        await rebuilder.activate(
            session=_FakeSession(),  # type: ignore[arg-type]
            index_version=version,  # type: ignore[arg-type]
        )


async def test_rollback_rejects_version_from_another_kb() -> None:
    """回滚时必须校验版本属于同一知识库。

    不校验的话，把 A 库的版本指给 B 库，B 库的查询会返回 A 库的内容
    ——这是跨知识库数据泄漏。
    """
    rebuilder = IndexRebuilder(provider=_FakeProvider())  # type: ignore[arg-type]

    @dataclass
    class _OtherKbVersion:
        knowledge_base_id: uuid.UUID

    class _FakeSession:
        async def get(self, *args: object, **kwargs: object) -> Any:
            return _OtherKbVersion(knowledge_base_id=uuid.uuid4())

        async def commit(self) -> None:
            return None

    with pytest.raises(_RebuildError, match="不属于该知识库"):
        await rebuilder.rollback(
            session=_FakeSession(),  # type: ignore[arg-type]
            knowledge_base_id=uuid.uuid4(),
            to_version_id=uuid.uuid4(),
        )


# ==================================================== 集成：真实数据库


@pytest.fixture
async def env() -> AsyncIterator[tuple[Any, uuid.UUID, uuid.UUID]]:
    """独立引擎 + 租户 + 知识库 + 100 个带向量的 chunk。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    settings = get_settings()
    db = settings.database
    user = db.migration_user or db.user
    password = (
        db.migration_password.get_secret_value()
        if db.migration_password is not None
        else db.password.get_secret_value()
    )
    dsn = f"postgresql+asyncpg://{user}:{password}@{db.host}:{db.port}/{db.name}"
    engine = create_async_engine(
        dsn,
        pool_size=4,
        connect_args={"server_settings": {"random_page_cost": str(db.random_page_cost)}},
    )
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    tenant_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    await _seed(factory, tenant_id=tenant_id, kb_id=kb_id, chunks=100)
    try:
        yield factory, tenant_id, kb_id
    finally:
        await _cleanup(factory, tenant_id=tenant_id)
        await engine.dispose()


async def test_backfill_creates_second_version_rows(
    env: tuple[Any, uuid.UUID, uuid.UUID],
) -> None:
    """回填在新版本下**新增**旁表行，旧版本行不动。

    这是零停机的物理基础：旧行还在，所以旧版本的查询完全不受影响。
    """
    from rag_db.repositories.chunk import IndexVersionRepository

    factory, tenant_id, kb_id = env
    rebuilder = IndexRebuilder(provider=_FakeProvider())  # type: ignore[arg-type]

    async with factory() as session:
        v2 = await rebuilder.start(
            session=session,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            version_tag="v2",
        )
        v2_id = v2.id
        await session.commit()

        v1 = await IndexVersionRepository(session).get_active(knowledge_base_id=kb_id)
        assert v1 is not None
        v1_id = v1.id

        progress = await rebuilder.backfill(session=session, index_version=v2, batch_size=20)
        assert progress.is_complete

        # 两个版本各有 100 条
        for version_id in (v1_id, v2_id):
            row = await session.execute(
                text("SELECT count(*) FROM chunk_embeddings WHERE index_version_id = :v"),
                {"v": version_id},
            )
            assert int(row.scalar_one()) == 100, f"版本 {version_id} 的行数不对"

        # 总行数 200：同一 chunk 有两份向量
        row = await session.execute(
            text("SELECT count(*) FROM chunk_embeddings WHERE tenant_id = :t"),
            {"t": tenant_id},
        )
        assert int(row.scalar_one()) == 200


async def test_search_filters_by_index_version(
    env: tuple[Any, uuid.UUID, uuid.UUID],
) -> None:
    """检索必须只看指定版本的向量。

    不过滤的话，同一 chunk 的两份向量都会命中 →
    引用重复 + 新旧模型距离不可比（排序错乱）。
    """
    from rag_db.repositories.chunk import ChunkRepository, IndexVersionRepository

    factory, tenant_id, kb_id = env

    async with factory() as session:
        repo = ChunkRepository(session)
        active = await IndexVersionRepository(session).get_active(knowledge_base_id=kb_id)
        assert active is not None

        results = await repo.search_vector(
            tenant_id=tenant_id,
            query_embedding=_unit_vector(5),
            limit=10,
            index_version_ids=[active.id],
        )
        # 只能拿到 10 条（limit），且没有重复的 chunk
        assert len(results) == 10
        chunk_ids = [chunk.id for chunk, _ in results]
        assert len(set(chunk_ids)) == 10, "同一 chunk 被返回多次（版本过滤失效）"


async def test_set_embeddings_is_idempotent(
    env: tuple[Any, uuid.UUID, uuid.UUID],
) -> None:
    """重复写入同一 (chunk, version) 不产生重复行。

    重派任务会重写同一批向量。纯 INSERT 会撞唯一约束
    （报错是 duplicate key，但根因是没做幂等）；
    ON CONFLICT DO UPDATE 让重跑安全。
    """
    from rag_db.repositories.chunk import ChunkRepository, IndexVersionRepository

    factory, tenant_id, kb_id = env

    async with factory() as session:
        repo = ChunkRepository(session)
        active = await IndexVersionRepository(session).get_active(knowledge_base_id=kb_id)
        assert active is not None

        row = await session.execute(
            text("SELECT chunk_id FROM chunk_embeddings WHERE tenant_id = :t LIMIT 3"),
            {"t": tenant_id},
        )
        chunk_ids = [r[0] for r in row]

        updates = [(chunk_id, _unit_vector(1), "test-fake-embed") for chunk_id in chunk_ids]
        await repo.set_embeddings(updates=updates, index_version_id=active.id, tenant_id=tenant_id)
        await session.commit()
        # 再写一遍
        await repo.set_embeddings(updates=updates, index_version_id=active.id, tenant_id=tenant_id)
        await session.commit()

        row = await session.execute(
            text(
                "SELECT count(*) FROM chunk_embeddings "
                "WHERE index_version_id = :v AND chunk_id = ANY(:ids)"
            ),
            {"v": active.id, "ids": chunk_ids},
        )
        assert int(row.scalar_one()) == len(chunk_ids), "重复写入产生了重复行"


async def test_deleting_index_version_cascades_vectors(
    env: tuple[Any, uuid.UUID, uuid.UUID],
) -> None:
    """回收索引版本时它的向量一起删，但 chunk 本身保留。

    CASCADE 在 `chunk_embeddings.index_version_id` 上；
    chunk 是内容，不该因为"某个向量版本被回收"而消失。
    """
    from rag_db.repositories.chunk import IndexVersionRepository

    factory, tenant_id, kb_id = env
    rebuilder = IndexRebuilder(provider=_FakeProvider())  # type: ignore[arg-type]

    async with factory() as session:
        v2 = await rebuilder.start(
            session=session,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            version_tag="v2",
        )
        v2_id = v2.id
        await session.commit()

        active = await IndexVersionRepository(session).get_active(knowledge_base_id=kb_id)
        assert active is not None

        # 回收 v2
        await session.execute(text("DELETE FROM index_versions WHERE id = :v"), {"v": v2_id})
        await session.commit()

        row = await session.execute(
            text("SELECT count(*) FROM chunk_embeddings WHERE index_version_id = :v"),
            {"v": v2_id},
        )
        assert int(row.scalar_one()) == 0, "版本回收后向量没被清理"

        row = await session.execute(
            text("SELECT count(*) FROM chunks WHERE tenant_id = :t"), {"t": tenant_id}
        )
        assert int(row.scalar_one()) == 100, "chunk 被误删（应只删向量）"


# ==================================================== 数据准备


async def _seed(factory: Any, *, tenant_id: uuid.UUID, kb_id: uuid.UUID, chunks: int) -> None:
    """建租户/用户/知识库/索引版本/文档链 + N 个 chunk 与向量。"""
    user_id = uuid.uuid4()
    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "created_at, updated_at) VALUES (:id, :e, 'idx', 'active', 'none', "
                "now(), now())"
            ),
            {"id": user_id, "e": f"idx-{user_id.hex[:8]}@test.local"},
        )
        await session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:id, :s, 'idx', 'active', now(), now())"
            ),
            {"id": tenant_id, "s": f"idx-{tenant_id.hex[:8]}"},
        )
        await session.execute(
            text(
                "INSERT INTO knowledge_bases (id, tenant_id, name, status, created_by, "
                "created_at, updated_at) VALUES (:id, :t, :n, 'active', :u, now(), now())"
            ),
            {"id": kb_id, "t": tenant_id, "n": f"kb-{kb_id.hex[:8]}", "u": user_id},
        )
        blob_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, storage_key, "
                "storage_backend, content_type, status, ref_count, created_at, updated_at) "
                "VALUES (:id, :t, :h, 1, :k, 'local', 'text/plain', 'complete', 1, "
                "now(), now())"
            ),
            {"id": blob_id, "t": tenant_id, "h": "c" * 64, "k": f"idx-{blob_id.hex[:8]}"},
        )
        doc_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, "
                "original_filename, format, source_type, confidentiality, status, "
                "current_version_no, latest_version_no, retry_count, chunk_count, "
                "size_bytes, created_by, version, created_at, updated_at) "
                "VALUES (:id, :t, :kb, 'idx.pdf', 'idx.pdf', 'pdf', 'upload', 'internal', "
                "'ready', 1, 1, 0, :n, 1, :u, 1, now(), now())"
            ),
            {"id": doc_id, "t": tenant_id, "kb": kb_id, "n": chunks, "u": user_id},
        )
        ver_row_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO document_versions (id, tenant_id, document_id, version_no, "
                "blob_id, status, chunking_strategy, chunking_config, chunk_count, "
                "parse_metadata, created_by, is_current, created_at, updated_at) "
                "VALUES (:id, :t, :d, 1, :b, 'ready', 'recursive', '{}'::jsonb, :n, "
                "'{}'::jsonb, :u, true, now(), now())"
            ),
            {
                "id": ver_row_id,
                "t": tenant_id,
                "d": doc_id,
                "b": blob_id,
                "n": chunks,
                "u": user_id,
            },
        )

        from rag_db.repositories.chunk import IndexVersionRepository

        iv_repo = IndexVersionRepository(session)
        v1 = await iv_repo.create(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            version_tag="v1",
            embedding_model="test-fake-embed",
            embedding_dim=1024,
            expected_chunk_count=chunks,
        )
        v1_id = v1.id
        await iv_repo.activate(knowledge_base_id=kb_id, version_id=v1_id)
        await session.commit()

    # 批量插 chunk + 向量
    batch = 50
    for start in range(0, chunks, batch):
        async with factory() as session:
            for offset in range(start, min(start + batch, chunks)):
                chunk_id = uuid.uuid4()
                await session.execute(
                    text(
                        "INSERT INTO chunks (id, tenant_id, knowledge_base_id, document_id, "
                        "document_version_id, chunk_index, chunk_type, content, content_hash, "
                        "section_path, chunk_level, is_current, chunk_metadata, "
                        "created_at, updated_at) "
                        "VALUES (:id, :t, :kb, :d, :v, :i, 'text', :c, :h, '[]'::jsonb, "
                        "0, true, '{}'::jsonb, now(), now())"
                    ),
                    {
                        "id": chunk_id,
                        "t": tenant_id,
                        "kb": kb_id,
                        "d": doc_id,
                        "v": ver_row_id,
                        "i": offset,
                        "c": f"idx chunk {offset}",
                        "h": f"{offset:064x}",
                    },
                )
                await session.execute(
                    text(
                        "INSERT INTO chunk_embeddings (id, tenant_id, chunk_id, "
                        "index_version_id, embedding, embedding_model, created_at, updated_at) "
                        "VALUES (:id, :t, :c, :v, :e, 'test-fake-embed', now(), now())"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "t": tenant_id,
                        "c": chunk_id,
                        "v": v1_id,
                        "e": str(_unit_vector(offset + 1)),
                    },
                )
            await session.commit()


async def _cleanup(factory: Any, *, tenant_id: uuid.UUID) -> None:
    async with factory() as session:
        for statement in (
            "DELETE FROM chunk_embeddings WHERE tenant_id = :t",
            "DELETE FROM chunks WHERE tenant_id = :t",
            "DELETE FROM index_versions WHERE tenant_id = :t",
            "DELETE FROM document_versions WHERE tenant_id = :t",
            "DELETE FROM documents WHERE tenant_id = :t",
            "DELETE FROM knowledge_bases WHERE tenant_id = :t",
            "DELETE FROM blobs WHERE tenant_id = :t",
        ):
            await session.execute(text(statement), {"t": tenant_id})
        await session.execute(text("DELETE FROM users WHERE email LIKE 'idx-%@test.local'"))
        await session.execute(text("DELETE FROM tenants WHERE id = :t"), {"t": tenant_id})
        await session.commit()


# 保留：并发演练的简化版（完整版见 scripts/drill_index_rebuild.py）
async def test_queries_work_during_backfill(
    env: tuple[Any, uuid.UUID, uuid.UUID],
) -> None:
    """回填期间旧版本的查询始终有结果（零停机的核心断言）。

    精简版演练：只验证"回填与查询并发时查询不返回空"。
    完整版（含切换/回滚耗时）见 scripts/drill_index_rebuild.py。
    """
    from rag_db.repositories.chunk import ChunkRepository, IndexVersionRepository

    factory, tenant_id, kb_id = env
    rebuilder = IndexRebuilder(provider=_FakeProvider())  # type: ignore[arg-type]

    async with factory() as session:
        v2 = await rebuilder.start(
            session=session,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            version_tag="v2",
        )
        v2_obj = v2
        await session.commit()

        active = await IndexVersionRepository(session).get_active(knowledge_base_id=kb_id)
        assert active is not None
        v1_id = active.id

    empty_results = 0
    query_count = 0
    stop = asyncio.Event()

    async def query_loop() -> None:
        nonlocal empty_results, query_count
        while not stop.is_set():
            async with factory() as session:
                results = await ChunkRepository(session).search_vector(
                    tenant_id=tenant_id,
                    query_embedding=_unit_vector(3),
                    limit=5,
                    index_version_ids=[v1_id],
                )
            query_count += 1
            if not results:
                empty_results += 1
            await asyncio.sleep(0.005)

    task = asyncio.create_task(query_loop())
    async with factory() as session:
        await rebuilder.backfill(session=session, index_version=v2_obj, batch_size=10)
    stop.set()
    await task

    assert query_count > 0, "并发查询一次都没跑"
    assert empty_results == 0, (
        f"回填期间有 {empty_results}/{query_count} 次查询返回空 —— 说明存在停机窗口"
    )


_ = DocumentStatus  # 保留导入：文档状态常量在断言里可能用到
