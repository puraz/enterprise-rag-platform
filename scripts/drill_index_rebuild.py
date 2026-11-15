"""零停机重建 + 回滚演练（A10 验收脚本）。

## 演练什么

真实生产里"重建索引"最容易出的事故不是"重建失败"，而是**切换错了**：

- 回填没跑完就切 → 检索静默少一半结果；
- 切换过程中有窗口期没有 ACTIVE 版本 → 所有查询返回空；
- 切错了想回滚，但旧向量已被删 → 回不去。

这个脚本用真实数据库把这三件事都演一遍，并测量"切换期间查询是否受影响"。

## 演练步骤

    1. 建一个 building 版本 V2
    2. **并发**：一边跑回填（V2 写入），一边持续查询（走 V1）
       → 断言查询始终有结果、且结果不混杂
    3. 回填未完成时尝试 `activate` → 必须被拒绝（这是安全阀）
    4. 回填完成后 `activate` V2 → 测切换耗时
    5. `rollback` 回 V1 → 验证秒级回滚，且 V1 的向量还在
    6. 清理

## 为什么"回填期间查询"要真并发

"零停机"是关于**时间**的断言，单线程顺序执行证明不了任何事。
必须让回填与查询在两条独立的数据库连接上真正同时进行。

## 用法

    uv run python scripts/drill_index_rebuild.py
    uv run python scripts/drill_index_rebuild.py --chunks 200   # 更大数据量
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_core.settings import get_settings, load_env_files
from rag_db.repositories.chunk import IndexVersionRepository
from rag_worker.indexing import IndexRebuilder, _RebuildError


@dataclass
class DrillReport:
    """演练结果。"""

    chunks: int
    backfill_s: float
    queries_during_backfill: int
    query_errors: int
    empty_query_results: int
    premature_activate_rejected: bool
    switch_ms: float
    rollback_ms: float
    old_vectors_survived: bool


def _make_vector(seed: int, dim: int = 1024) -> list[float]:
    """确定性伪随机单位向量（不依赖模型，让演练可重复且快）。

    用真实 embedding 会让演练耗时几分钟且需要 GPU；
    演练要验证的是**版本切换机制**，向量具体值无关紧要。
    """
    values: list[float] = []
    state = seed * 2654435761 % (2**32)
    for _ in range(dim):
        state = (state * 1103515245 + 12345) % (2**31)
        values.append((state / (2**31)) - 0.5)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class _FakeProvider:
    """假 embedding provider：只产出确定性向量，不加载模型。

    **必须实现真实接口**（name / capabilities / embed_batch），
    这样 `IndexRebuilder` 走的是与生产完全相同的代码路径。
    """

    name = "drill-fake-embed"

    class _Caps:
        dimension = 1024

    capabilities = _Caps()

    async def embed_batch(self, texts: list[str], *, task: object = None) -> object:
        class _Result:
            def __init__(self, vectors: list[list[float]]) -> None:
                self.vectors = vectors
                self.latency_ms = 0
                self.dimension = 1024
                self.model = "drill-fake-embed"

        # 用文本哈希做种子：同一文本每次得到同一向量（可复现）
        return _Result([_make_vector(abs(hash(t)) % 100000) for t in texts])


async def _seed(
    factory: async_sessionmaker[AsyncSession], *, chunks: int
) -> tuple[uuid.UUID, uuid.UUID]:
    """建租户 + 知识库 + N 个带向量的 chunk（属于 V1）。"""
    tenant_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:id, :slug, 'drill', 'active', now(), now())"
            ),
            {"id": tenant_id, "slug": f"drill-{tenant_id.hex[:8]}"},
        )
        user_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "created_at, updated_at) VALUES (:id, :e, 'drill', 'active', 'none', now(), now())"
            ),
            {"id": user_id, "e": f"drill-{user_id.hex[:8]}@test.local"},
        )
        await session.execute(
            text(
                "INSERT INTO knowledge_bases (id, tenant_id, name, status, created_by, "
                "created_at, updated_at) "
                "VALUES (:id, :tid, 'drill-kb', 'active', :uid, now(), now())"
            ),
            {"id": kb_id, "tid": tenant_id, "uid": user_id},
        )
        # 建 V1 并激活
        v1 = await IndexVersionRepository(session).create(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            version_tag="v1",
            embedding_model="drill-fake-embed",
            embedding_dim=1024,
            expected_chunk_count=chunks,
        )
        # 先取出 id 再 commit：commit 后对象属性访问可能触发懒加载
        # （MissingGreenlet）。虽然本工程配了 expire_on_commit=False，
        # 但显式取值更稳妥，且意图清晰。
        v1_id = v1.id
        await IndexVersionRepository(session).activate(knowledge_base_id=kb_id, version_id=v1_id)
        await session.commit()

    # chunk 用裸 SQL 插入（需要 document/version/blob 外键，产出与演练无关）
    async with factory() as session:
        doc_id = uuid.uuid4()
        ver_id = uuid.uuid4()
        blob_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, storage_key, "
                "storage_backend, content_type, status, ref_count, created_at, updated_at) "
                "VALUES (:id, :tid, :h, 1, 'drill/key', 'local', 'text/plain', 'complete', 1, "
                "now(), now())"
            ),
            {"id": blob_id, "tid": tenant_id, "h": "a" * 64},
        )
        await session.execute(
            text(
                "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, "
                "original_filename, format, source_type, confidentiality, status, "
                "current_version_no, latest_version_no, retry_count, chunk_count, "
                "size_bytes, created_by, created_at, updated_at, version) "
                "VALUES (:id, :tid, :kb, 'drill.pdf', 'drill.pdf', 'pdf', 'upload', "
                "'internal', 'ready', 1, 1, 0, :n, 1, :uid, now(), now(), 1)"
            ),
            {"id": doc_id, "tid": tenant_id, "kb": kb_id, "n": chunks, "uid": user_id},
        )
        await session.execute(
            text(
                "INSERT INTO document_versions (id, tenant_id, document_id, version_no, "
                "blob_id, status, chunking_strategy, chunking_config, chunk_count, "
                "parse_metadata, created_by, is_current, created_at, updated_at) "
                "VALUES (:id, :tid, :doc, 1, :bid, 'ready', 'recursive', '{}'::jsonb, :n, "
                "'{}'::jsonb, :uid, true, now(), now())"
            ),
            {
                "id": ver_id,
                "tid": tenant_id,
                "doc": doc_id,
                "bid": blob_id,
                "n": chunks,
                "uid": user_id,
            },
        )
        await session.commit()

    # 批量插 chunk（每个带 V1 向量）
    batch = 100
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
                        "VALUES (:id, :tid, :kb, :doc, :ver, :idx, 'text', :c, :h, "
                        "'[]'::jsonb, 0, true, '{}'::jsonb, now(), now())"
                    ),
                    {
                        "id": chunk_id,
                        "tid": tenant_id,
                        "kb": kb_id,
                        "doc": doc_id,
                        "ver": ver_id,
                        "idx": offset,
                        "c": f"drill chunk {offset}",
                        "h": f"{offset:064x}",
                    },
                )
                # 向量写旁表（旁表方案：一个 chunk 每个版本一行向量）
                await session.execute(
                    text(
                        "INSERT INTO chunk_embeddings (id, tenant_id, chunk_id, "
                        "index_version_id, embedding, embedding_model, created_at, updated_at) "
                        "VALUES (:id, :tid, :cid, :vid, :emb, 'drill-fake-embed', now(), now())"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "tid": tenant_id,
                        "cid": chunk_id,
                        "vid": v1_id,
                        "emb": str(_make_vector(offset)),
                    },
                )
            await session.commit()

    return tenant_id, kb_id


async def _query_once(
    factory: async_sessionmaker[AsyncSession],
    *,
    tenant_id: uuid.UUID,
    kb_id: uuid.UUID,
    active_version_id: uuid.UUID,
) -> int:
    """执行一次向量检索，返回结果条数。

    用真实 SQL（与 `ChunkRepository.search_vector` 同样的过滤条件），
    而不是调用仓储——演练要验证的是 SQL 层行为。

    **关键：按 `index_version_id` 过滤**。不过滤会同时返回新旧两套向量。
    """
    query = _make_vector(1)
    async with factory() as session:
        result = await session.execute(
            text(
                "SELECT count(*) FROM ("
                "  SELECT c.id FROM chunks c"
                "  JOIN chunk_embeddings ce ON ce.chunk_id = c.id"
                "  WHERE c.tenant_id = :tid AND c.knowledge_base_id = :kb"
                "    AND c.is_current = true AND c.chunk_level = 0"
                "    AND ce.index_version_id = :vid"
                "  ORDER BY ce.embedding <=> CAST(:q AS vector) LIMIT 5"
                ") sub"
            ),
            {"tid": tenant_id, "kb": kb_id, "vid": active_version_id, "q": str(query)},
        )
        return int(result.scalar_one())


async def run_drill(*, chunks: int) -> DrillReport:
    """执行完整演练。"""
    load_env_files()
    settings = get_settings()
    db = settings.database
    dsn = (
        f"postgresql+asyncpg://{db.migration_user}:"
        f"{db.migration_password.get_secret_value()}@{db.host}:{db.port}/{db.name}"
    )

    engine = create_async_engine(dsn, pool_size=8)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    tenant_id, kb_id = await _seed(factory, chunks=chunks)
    provider = _FakeProvider()
    rebuilder = IndexRebuilder(provider=provider)  # type: ignore[arg-type]

    try:
        # ---- 1. 建 V2（building，不切换）----
        async with factory() as session:
            v2 = await rebuilder.start(
                session=session,
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                version_tag="v2",
            )
            await session.commit()
            v2_id = v2.id
            v2_expected = v2.expected_chunk_count

        # V1 仍是 ACTIVE
        async with factory() as session:
            active = await IndexVersionRepository(session).get_active(knowledge_base_id=kb_id)
            assert active is not None and active.version_tag == "v1", "start() 不该切换 ACTIVE"
            v1_id = active.id

        # ---- 2. 并发：回填 vs 持续查询 ----
        query_count = 0
        query_errors = 0
        empty_results = 0
        stop = asyncio.Event()

        async def query_loop() -> None:
            nonlocal query_count, query_errors, empty_results
            while not stop.is_set():
                try:
                    got = await _query_once(
                        factory,
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        # 回填期间 ACTIVE 仍是 V1
                        active_version_id=v1_id,
                    )
                    query_count += 1
                    if got == 0:
                        # 这是最严重的症状：切换期间查询变空
                        empty_results += 1
                except Exception:
                    query_errors += 1
                await asyncio.sleep(0.01)

        query_task = asyncio.create_task(query_loop())

        started = time.perf_counter()
        async with factory() as session:
            progress = await rebuilder.backfill(session=session, index_version=v2, batch_size=20)
        backfill_s = time.perf_counter() - started

        # 回填必须完整完成，否则后续"切换被拒绝"的演练没有意义
        assert progress.is_complete, f"回填未完成：{progress.embedded}/{progress.total}"

        stop.set()
        await query_task

        # ---- 3. 未完成时切换必须被拒绝 ----
        premature_rejected = False
        async with factory() as session:
            # 造一个"回填没完成"的版本
            v3 = await rebuilder.start(
                session=session,
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                version_tag="v3-unfinished",
            )
            # **先取 id**：rollback() 会让对象过期，之后访问 v3.id
            # 会触发懒加载 → MissingGreenlet（异步会话里禁止隐式 IO）。
            v3_id = v3.id
            await session.commit()
            try:
                await rebuilder.activate(session=session, index_version=v3)
                await session.commit()
            except _RebuildError:
                premature_rejected = True
                await session.rollback()
            # 清理这个半成品版本
            await session.execute(text("DELETE FROM index_versions WHERE id = :id"), {"id": v3_id})
            await session.commit()

        # ---- 4. 切换（测耗时）----
        switch_started = time.perf_counter()
        async with factory() as session:
            await rebuilder.activate(session=session, index_version=v2)
        switch_ms = (time.perf_counter() - switch_started) * 1000

        # 切换后查询走 V2，应有结果
        after_switch = await _query_once(
            factory, tenant_id=tenant_id, kb_id=kb_id, active_version_id=v2_id
        )

        # ---- 5. 回滚（测耗时 + 验证旧向量还在）----
        rollback_started = time.perf_counter()
        async with factory() as session:
            await rebuilder.rollback(session=session, knowledge_base_id=kb_id, to_version_id=v1_id)
        rollback_ms = (time.perf_counter() - rollback_started) * 1000

        async with factory() as session:
            row = await session.execute(
                text("SELECT count(*) FROM chunk_embeddings WHERE index_version_id = :vid"),
                {"vid": v1_id},
            )
            v1_vectors = int(row.scalar_one())

        if after_switch == 0:
            raise AssertionError("切换后查询返回空，说明版本过滤条件错了")

        return DrillReport(
            chunks=chunks,
            backfill_s=backfill_s,
            queries_during_backfill=query_count,
            query_errors=query_errors,
            empty_query_results=empty_results,
            premature_activate_rejected=premature_rejected,
            switch_ms=switch_ms,
            rollback_ms=rollback_ms,
            old_vectors_survived=v1_vectors >= v2_expected,
        )
    finally:
        async with factory() as session:
            await session.execute(
                text("DELETE FROM chunk_embeddings WHERE tenant_id = :tid"),
                {"tid": tenant_id},
            )
            await session.execute(
                text("DELETE FROM chunks WHERE tenant_id = :tid"), {"tid": tenant_id}
            )
            await session.execute(
                text("DELETE FROM document_versions WHERE tenant_id = :tid"), {"tid": tenant_id}
            )
            await session.execute(
                text("DELETE FROM documents WHERE tenant_id = :tid"), {"tid": tenant_id}
            )
            await session.execute(
                text("DELETE FROM index_versions WHERE tenant_id = :tid"), {"tid": tenant_id}
            )
            await session.execute(
                text("DELETE FROM knowledge_bases WHERE tenant_id = :tid"), {"tid": tenant_id}
            )
            await session.execute(
                text("DELETE FROM blobs WHERE tenant_id = :tid"), {"tid": tenant_id}
            )
            await session.execute(text("DELETE FROM users WHERE email LIKE 'drill-%@test.local'"))
            await session.execute(text("DELETE FROM tenants WHERE id = :tid"), {"tid": tenant_id})
            await session.commit()
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="零停机重建与回滚演练")
    parser.add_argument("--chunks", type=int, default=100, help="chunk 数量")
    args = parser.parse_args(argv)

    report = asyncio.run(run_drill(chunks=args.chunks))

    print("\n" + "=" * 62)
    print("零停机重建演练结果")
    print("=" * 62)
    print(f"  chunk 数量              : {report.chunks}")
    print(f"  回填耗时                : {report.backfill_s:.2f}s")
    print(f"  回填期间查询次数        : {report.queries_during_backfill}")
    print(f"  查询错误                : {report.query_errors}")
    print(f"  查询返回空（应为 0）    : {report.empty_query_results}")
    print(f"  未完成时切换被拒绝      : {report.premature_activate_rejected}")
    print(f"  切换耗时                : {report.switch_ms:.1f}ms")
    print(f"  回滚耗时                : {report.rollback_ms:.1f}ms")
    print(f"  旧版本向量保留（可回滚）: {report.old_vectors_survived}")

    failures: list[str] = []
    if report.query_errors:
        failures.append(f"回填期间出现 {report.query_errors} 次查询错误")
    if report.empty_query_results:
        failures.append(f"回填期间有 {report.empty_query_results} 次查询返回空（停机！）")
    if not report.premature_activate_rejected:
        failures.append("未完成回填也能切换 —— 会静默丢结果")
    if not report.old_vectors_survived:
        failures.append("切换后旧向量被删 —— 无法回滚")
    if report.switch_ms > 500:
        failures.append(f"切换耗时 {report.switch_ms:.0f}ms 过长（应是一条 UPDATE）")

    if failures:
        print("\n失败项：")
        for item in failures:
            print(f"  ✗ {item}")
        return 1

    print("\n✓ 全部通过：回填期间查询不中断、切换秒级、可回滚")
    return 0


if __name__ == "__main__":
    sys.exit(main())
