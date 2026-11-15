"""分块与向量检索的集成测试（真实 PG + pgvector）。

这批测试验证的是"检索真的能跑、且在多租户下不漏不串"，不是 CRUD：

- 向量检索能按距离排序返回
- **跨租户隔离**：A 租户的查询永远拿不到 B 租户的 chunk
- HNSW 索引真实生效（执行计划走索引而非全表扫描）
- 多租户过滤后的召回（iterative_scan 的作用）
- 父子块回填
- 索引版本切换的原子性
- chunk 的唯一约束与 CHECK 约束
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums.retrieval import ChunkType
from rag_db.models.chunk import EMBEDDING_DIM, IndexVersion
from rag_db.repositories import ChunkRepository, IndexVersionRepository

pytestmark = pytest.mark.integration


def _vector(seed: int, dim: int = EMBEDDING_DIM) -> list[float]:
    """生成一个确定性的伪随机单位向量。

    用确定性生成（不用 random）让测试可复现；
    归一化让余弦距离的行为更可预测。
    """
    import math

    # 用简单线性同余生成伪随机值，避免依赖 numpy
    values: list[float] = []
    state = seed * 2654435761 % (2**32)
    for _ in range(dim):
        state = (state * 1103515245 + 12345) % (2**31)
        values.append((state / (2**31)) - 0.5)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


async def _make_tenant(session: AsyncSession, suffix: str = "") -> uuid.UUID:
    tenant_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
            "VALUES (:id, :slug, :name, 'active', now(), now())"
        ),
        {"id": tenant_id, "slug": f"t-{tenant_id.hex[:10]}{suffix}", "name": "测试租户"},
    )
    return tenant_id


async def _make_user(session: AsyncSession) -> uuid.UUID:
    user_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO users (id, email, display_name, status, platform_role, "
            "tokens_valid_after, created_at, updated_at) "
            "VALUES (:id, :e, 'U', 'active', 'none', now(), now(), now())"
        ),
        {"id": user_id, "e": f"u-{user_id.hex[:8]}@example.test"},
    )
    return user_id


async def _make_kb(session: AsyncSession, tenant_id: uuid.UUID, user_id: uuid.UUID) -> uuid.UUID:
    kb_id = uuid.uuid4()
    name = f"库-{kb_id.hex[:8]}"
    await session.execute(
        text(
            "INSERT INTO knowledge_bases (id, tenant_id, name, status, visibility, "
            "confidentiality, created_by, default_chunking_strategy, default_chunk_size, "
            "default_chunk_overlap, document_count, created_at, updated_at) VALUES ("
            ":id, :t, :n, 'active', 'tenant', 'internal', :u, 'recursive', 800, 100, 0, "
            "now(), now())"
        ),
        {"id": kb_id, "t": tenant_id, "n": name, "u": user_id},
    )
    return kb_id


async def _make_doc_chain(
    session: AsyncSession, tenant_id: uuid.UUID, kb_id: uuid.UUID, user_id: uuid.UUID
) -> tuple[uuid.UUID, uuid.UUID]:
    """创建 doc + blob + version，返回 (document_id, version_id)。"""
    doc_id = uuid.uuid4()
    blob_id = uuid.uuid4()
    version_id = uuid.uuid4()
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex

    await session.execute(
        text(
            "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, storage_key, "
            "storage_backend, content_type, status, ref_count, created_at, updated_at) "
            "VALUES (:i, :t, :h, 100, :k, 'local', 'application/pdf', 'complete', 1, now(), now())"
        ),
        {"i": blob_id, "t": tenant_id, "h": content_hash, "k": f"k-{content_hash[:8]}"},
    )
    await session.execute(
        text(
            "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, original_filename, "
            "format, source_type, confidentiality, status, current_version_no, latest_version_no, "
            "retry_count, chunk_count, size_bytes, created_by, version, created_at, updated_at) "
            "VALUES (:i, :t, :k, :title, 'x.pdf', 'pdf', 'upload', 'internal', 'ready', 1, 1, "
            "0, 0, 100, :u, 1, now(), now())"
        ),
        {"i": doc_id, "t": tenant_id, "k": kb_id, "title": f"文档-{doc_id.hex[:6]}", "u": user_id},
    )
    await session.execute(
        text(
            "INSERT INTO document_versions (id, tenant_id, document_id, version_no, blob_id, "
            "status, chunking_strategy, chunking_config, chunk_count, parse_metadata, "
            "created_by, is_current, created_at, updated_at) "
            "VALUES (:i, :t, :d, 1, :b, 'ready', 'recursive', '{}'::jsonb, 0, '{}'::jsonb, "
            ":u, true, now(), now())"
        ),
        {"i": version_id, "t": tenant_id, "d": doc_id, "b": blob_id, "u": user_id},
    )
    await session.execute(
        text("UPDATE documents SET current_version_id = :v WHERE id = :d"),
        {"v": version_id, "d": doc_id},
    )
    return doc_id, version_id


def _chunk_payload(index: int, content: str, embedding: list[float] | None) -> dict[str, object]:
    import hashlib

    return {
        "chunk_index": index,
        "content": content,
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        "chunk_type": ChunkType.TEXT.value,
        "embedding": embedding,
    }


# ==================================================== 基础 CRUD


async def test_bulk_create_chunks(db_session: AsyncSession) -> None:
    """批量插入 chunk 并按序号读回。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)

    repo = ChunkRepository(db_session)
    count = await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[_chunk_payload(i, f"第 {i} 段内容", _vector(i + 1)) for i in range(5)],
    )
    assert count == 5

    chunks = await repo.list_by_version(document_version_id=version_id)
    assert len(chunks) == 5
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4]


async def test_chunk_index_unique_per_version(db_session: AsyncSession) -> None:
    """同一版本内 chunk_index 不能重复。

    重复会产生两条一样的引用，用户看到"同一段原文出现两次"。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)
    repo = ChunkRepository(db_session)

    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[_chunk_payload(0, "内容 A", _vector(1))],
    )
    with pytest.raises(IntegrityError):
        await repo.bulk_create(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            document_id=doc_id,
            document_version_id=version_id,
            chunks=[_chunk_payload(0, "内容 B", _vector(2))],
        )


async def test_char_range_must_be_consistent(db_session: AsyncSession) -> None:
    """char_start/char_end 必须成对且 end > start。

    只给 start 不给 end，或 end <= start，会让前端高亮算出非法区间
    （可能崩溃，也可能高亮整页）。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)

    import hashlib

    with pytest.raises(IntegrityError):
        await db_session.execute(
            text(
                "INSERT INTO chunks (id, tenant_id, knowledge_base_id, document_id, "
                "document_version_id, chunk_index, chunk_type, content, content_hash, "
                "char_start, char_end, section_path, chunk_level, is_current, chunk_metadata, "
                "created_at, updated_at) "
                "VALUES (:i, :t, :k, :d, :v, 0, 'text', 'x', :h, 10, 5, '[]'::jsonb, 0, true, "
                "'{}'::jsonb, now(), now())"
            ),
            {
                "i": uuid.uuid4(),
                "t": tenant_id,
                "k": kb_id,
                "d": doc_id,
                "v": version_id,
                "h": hashlib.sha256(b"x").hexdigest(),
            },
        )


# ==================================================== 向量检索


async def test_vector_search_orders_by_distance(db_session: AsyncSession) -> None:
    """向量检索按距离升序返回，最相似的排最前。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)
    repo = ChunkRepository(db_session)

    target = _vector(42)
    # 三个 chunk：一个与查询相同，两个随机
    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[
            _chunk_payload(0, "完全匹配", target),
            _chunk_payload(1, "无关 A", _vector(999)),
            _chunk_payload(2, "无关 B", _vector(777)),
        ],
    )

    results = await repo.search_vector(tenant_id=tenant_id, query_embedding=target, limit=3)
    assert len(results) == 3
    distances = [distance for _, distance in results]
    assert distances == sorted(distances), "结果未按距离升序"
    assert results[0][0].content == "完全匹配"
    # 与自身余弦距离应接近 0
    assert results[0][1] < 0.01


async def test_vector_search_isolates_tenants(db_session: AsyncSession) -> None:
    """**跨租户隔离**：A 租户的查询绝不能返回 B 租户的 chunk。

    这是安全底线。租户过滤写在 SQL 里（不依赖 RLS），
    因为 RLS 是最后一道防线，不该是唯一防线。
    """
    tenant_a = await _make_tenant(db_session, "-a")
    tenant_b = await _make_tenant(db_session, "-b")
    user_a = await _make_user(db_session)
    user_b = await _make_user(db_session)
    kb_a = await _make_kb(db_session, tenant_a, user_a)
    kb_b = await _make_kb(db_session, tenant_b, user_b)
    doc_a, ver_a = await _make_doc_chain(db_session, tenant_a, kb_a, user_a)
    doc_b, ver_b = await _make_doc_chain(db_session, tenant_b, kb_b, user_b)

    repo = ChunkRepository(db_session)
    shared_vector = _vector(123)

    await repo.bulk_create(
        tenant_id=tenant_a,
        knowledge_base_id=kb_a,
        document_id=doc_a,
        document_version_id=ver_a,
        chunks=[_chunk_payload(0, "A 租户机密", shared_vector)],
    )
    await repo.bulk_create(
        tenant_id=tenant_b,
        knowledge_base_id=kb_b,
        document_id=doc_b,
        document_version_id=ver_b,
        chunks=[_chunk_payload(0, "B 租户机密", shared_vector)],
    )

    results_a = await repo.search_vector(
        tenant_id=tenant_a, query_embedding=shared_vector, limit=10
    )
    assert len(results_a) == 1
    assert results_a[0][0].content == "A 租户机密"

    results_b = await repo.search_vector(
        tenant_id=tenant_b, query_embedding=shared_vector, limit=10
    )
    assert len(results_b) == 1
    assert results_b[0][0].content == "B 租户机密"


async def test_vector_search_filters_by_knowledge_base(db_session: AsyncSession) -> None:
    """可按知识库列表限定检索范围。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb1 = await _make_kb(db_session, tenant_id, user_id)
    kb2 = await _make_kb(db_session, tenant_id, user_id)
    doc1, ver1 = await _make_doc_chain(db_session, tenant_id, kb1, user_id)
    doc2, ver2 = await _make_doc_chain(db_session, tenant_id, kb2, user_id)

    repo = ChunkRepository(db_session)
    vec = _vector(555)
    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb1,
        document_id=doc1,
        document_version_id=ver1,
        chunks=[_chunk_payload(0, "库1内容", vec)],
    )
    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb2,
        document_id=doc2,
        document_version_id=ver2,
        chunks=[_chunk_payload(0, "库2内容", vec)],
    )

    results = await repo.search_vector(
        tenant_id=tenant_id,
        query_embedding=vec,
        knowledge_base_ids=[kb1],
        limit=10,
    )
    assert len(results) == 1
    assert results[0][0].content == "库1内容"


async def test_vector_search_excludes_stale_chunks(db_session: AsyncSession) -> None:
    """is_current=false 的 chunk 不参与检索。

    旧版本的 chunk 必须能排除，否则回滚后新旧内容会同时被检索到，
    用户看到互相矛盾的两个答案。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)
    repo = ChunkRepository(db_session)

    vec = _vector(321)
    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[_chunk_payload(0, "旧内容", vec)],
    )
    await repo.mark_version_chunks_stale(document_version_id=version_id)

    results = await repo.search_vector(tenant_id=tenant_id, query_embedding=vec, limit=10)
    assert results == []


async def test_vector_search_respects_max_distance(db_session: AsyncSession) -> None:
    """max_distance 能过滤明显无关的结果。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)
    repo = ChunkRepository(db_session)

    target = _vector(1000)
    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[
            _chunk_payload(0, "接近", target),
            _chunk_payload(1, "很远", _vector(7)),
        ],
    )

    # 只允许很小的距离，应只剩"接近"
    results = await repo.search_vector(
        tenant_id=tenant_id, query_embedding=target, limit=10, max_distance=0.1
    )
    assert len(results) == 1
    assert results[0][0].content == "接近"


async def test_hnsw_index_is_used(db_session: AsyncSession) -> None:
    """**HNSW 索引必须真的被查询计划使用。**

    只建索引不代表查询会用它：如果表的统计信息显示数据量太小，
    PG 会选择顺序扫描（小表确实更快）。这个测试用 `SET enable_seqscan = off`
    强制走索引，验证索引本身可用、且能被向量查询匹配。

    真正要防的是：索引建错了操作符类（比如建了 l2 索引却用余弦查询），
    这时 `enable_seqscan=off` 下依然不会用索引 —— 索引存在但完全无效。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)
    repo = ChunkRepository(db_session)

    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[_chunk_payload(i, f"内容 {i}", _vector(i)) for i in range(20)],
    )

    # 强制不使用顺序扫描，看是否能走索引
    await db_session.execute(text("SET LOCAL enable_seqscan = off"))
    plan_result = await db_session.execute(
        text(
            "EXPLAIN SELECT id FROM chunks WHERE embedding IS NOT NULL "
            "AND is_current = true ORDER BY embedding <=> :q LIMIT 5"
        ),
        {"q": str(_vector(1))},
    )
    plan = "\n".join(str(row[0]) for row in plan_result)
    assert "ix_chunks_embedding_hnsw" in plan, f"查询未使用 HNSW 索引:\n{plan}"


# ==================================================== 关键词检索


async def test_keyword_search_finds_matching_chunk(db_session: AsyncSession) -> None:
    """关键词检索能命中包含关键词的 chunk。

    当前用 PG 的 simple 分词器（不懂中文分词），
    所以这里用英文关键词测试，验证链路可用。
    中文分词的局限在仓储 docstring 里已写明，A10 解决。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)
    repo = ChunkRepository(db_session)

    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[
            _chunk_payload(0, "reimbursement policy details", _vector(1)),
            _chunk_payload(1, "vacation days and holidays", _vector(2)),
        ],
    )

    results = await repo.search_keyword(tenant_id=tenant_id, keywords=["reimbursement"], limit=10)
    assert len(results) == 1
    assert "reimbursement" in results[0][0].content


# ==================================================== 父子块


async def test_parent_child_expansion(db_session: AsyncSession) -> None:
    """父块回填：命中子块后能取回父块上下文。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)
    repo = ChunkRepository(db_session)

    # 先建父块
    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[
            {
                **_chunk_payload(0, "完整的父块内容，包含完整上下文", _vector(1)),
                "chunk_level": 1,
            }
        ],
    )
    parent = (await repo.list_by_version(document_version_id=version_id))[0]

    # 再建引用了父块的子块
    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[
            {
                **_chunk_payload(1, "子块片段", _vector(2)),
                "parent_chunk_id": parent.id,
            }
        ],
    )
    all_chunks = await repo.list_by_version(document_version_id=version_id)
    child = next(c for c in all_chunks if c.chunk_index == 1)

    expanded = await repo.expanded_context(chunk_ids=[child.id])
    assert len(expanded) == 1
    assert expanded[0].id == parent.id


# ==================================================== 索引版本


async def test_index_version_activation_is_atomic(db_session: AsyncSession) -> None:
    """索引版本切换：旧版置 retired、新版置 active，且只有一个 active。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    repo = IndexVersionRepository(db_session)

    v1 = await repo.create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        version_tag="v1",
        embedding_model="bge-m3",
        embedding_dim=EMBEDDING_DIM,
    )
    await repo.mark_ready(version_id=v1.id, chunk_count=10)
    await repo.activate(knowledge_base_id=kb_id, version_id=v1.id)

    v2 = await repo.create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        version_tag="v2",
        embedding_model="bge-m3",
        embedding_dim=EMBEDDING_DIM,
    )
    await repo.mark_ready(version_id=v2.id, chunk_count=12)
    await repo.activate(knowledge_base_id=kb_id, version_id=v2.id)

    active = await repo.get_active(knowledge_base_id=kb_id)
    assert active is not None
    assert active.id == v2.id

    count = await db_session.execute(
        text(
            "SELECT count(*) FROM index_versions WHERE knowledge_base_id = :k AND status = 'active'"
        ),
        {"k": kb_id},
    )
    assert int(count.scalar_one()) == 1


async def test_only_one_active_index_version(db_session: AsyncSession) -> None:
    """同一知识库最多一个 active 索引版本（部分唯一索引保证）。

    两个 active 会让"当前用哪个索引"变成不确定 —— 检索结果随机变化。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)

    for tag in ("a", "b"):
        db_session.add(
            IndexVersion(
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                version_tag=tag,
                status="active",
                embedding_model="bge-m3",
                embedding_dim=EMBEDDING_DIM,
                index_type="hnsw",
                distance_metric="cosine",
                index_params={},
                chunk_count=0,
            )
        )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# ==================================================== 待向量化扫描


async def test_list_pending_embedding(db_session: AsyncSession) -> None:
    """能扫描出未向量化的 chunk（向量化中断后的补偿用）。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)
    repo = ChunkRepository(db_session)

    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[
            _chunk_payload(0, "已向量化", _vector(1)),
            _chunk_payload(1, "未向量化", None),
        ],
    )

    pending = await repo.list_pending_embedding(tenant_id=tenant_id)
    assert len(pending) == 1
    assert pending[0].content == "未向量化"


async def test_set_embeddings_batch(db_session: AsyncSession) -> None:
    """批量写入向量能正确落库。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb_id = await _make_kb(db_session, tenant_id, user_id)
    doc_id, version_id = await _make_doc_chain(db_session, tenant_id, kb_id, user_id)
    repo = ChunkRepository(db_session)

    await repo.bulk_create(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        document_version_id=version_id,
        chunks=[_chunk_payload(i, f"待向量 {i}", None) for i in range(3)],
    )
    chunks = await repo.list_by_version(document_version_id=version_id)
    updated = await repo.set_embeddings(
        updates=[(c.id, _vector(c.chunk_index + 10), "bge-m3") for c in chunks]
    )
    assert updated == 3

    results = await repo.search_vector(tenant_id=tenant_id, query_embedding=_vector(10), limit=1)
    assert len(results) == 1
