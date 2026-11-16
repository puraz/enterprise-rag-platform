"""检索层 ACL 回归测试：**过滤是否真的接进了检索路径**。

## 这批测试补的是什么洞

`tests/test_permissions.py` 已经验证了 `document_access_filter(ctx)` 这个
**SQL 表达式本身**是对的（INTERNAL 上限看不到 SECRET 文档）。

但 chunk 级检索必须用 `confidentiality_filter_for_chunks`（纯子查询），
不能用 `document_access_filter`（引用 Document 列会在 chunk 查询里
变成笛卡尔积）——详见 `chat.py::_retrieve` 的说明。

但"表达式对"不等于"检索用了它"。实测发现的缺口：

    ChatService._retrieve → ChunkRepository.search_vector(...)
                                 ↑ 没有传 access_filter

于是任何成员都能通过提问把机密文档的内容问出来。SQL 过滤函数
写得再对也没用——它根本没被调用。

这类缺口的特征是**没有任何报错**：向量检索正常返回、答案正常生成、
引用正常展示，只是返回了不该返回的内容。只有"两个角色分别检索并
对比结果"才能发现。

## 破坏性验证

每条测试都断言"该看到的看到了、不该看到的没看到"。如果某天有人
把 `access_filter` 从检索路径里摘掉，这些测试会立刻变红——
而不是像现在这样静默泄漏。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.permissions.acl import (
    PermissionContext,
    confidentiality_filter_for_chunks,
)
from rag_contracts.enums import ConfidentialityLevel, TenantRole
from rag_db.repositories.chunk import ChunkRepository

pytestmark = pytest.mark.integration

# 向量维度：与 `chunks.embedding` 的 vector(1024) 一致
_DIM = 1024


def _unit_vector(seed: float) -> list[float]:
    """构造一个确定性的单位向量。

    **不用随机向量**：随机向量的 top-10 与 top-40 距离差极小
    （AGENTS.md 4.13 实测过），任何近似索引都复现不出差异。
    这里需要的是"距离可预测"的向量，让三条文档的排序稳定。

    seed 决定向量方向：seed 越小越接近查询向量 [1, 0, 0, ...]。
    """
    vector = [0.0] * _DIM
    vector[0] = 1.0
    # 按 seed 把少量分量转到别的轴上，距离随 seed 单调增大
    for index in range(1, int(seed) + 1):
        vector[0] -= 0.01
        vector[index] = 0.1
    # 归一化（余弦距离要求单位向量才与内积等价）
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector]


@pytest.fixture
async def seeded_tenant(db_session: AsyncSession) -> AsyncIterator[dict[str, object]]:
    """建一个租户：三个知识库 + 三个文档（public / internal / secret）+ chunks。

    ## 为什么每个密级一个独立知识库

    `document_access_filter` 同时检查**库可见性**与**文档密级**。
    如果三个文档放在同一个 `tenant` 可见的库里，过滤差异就只来自密级。
    这样测试失败时能直接定位是哪一层出的问题。

    用 `db_session`（高权限连接）：准备数据需要跨租户写入权限。
    验证时用同一个 session——`access_filter` 是纯 SQL 表达式，
    不需要 RLS 参与（RLS 只防跨租户，ACL 防同租户越权）。
    """
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()

    async with db_session.begin_nested():
        await db_session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:i, :s, 'acl-test', 'active', now(), now())"
            ),
            {"i": tenant_id, "s": f"acl-{tenant_id.hex[:10]}"},
        )
        await db_session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "tokens_valid_after, created_at, updated_at) "
                "VALUES (:i, :e, 'acl', 'active', 'none', now(), now(), now())"
            ),
            {"i": user_id, "e": f"acl-{user_id.hex[:8]}@example.test"},
        )

        docs: dict[str, uuid.UUID] = {}
        kb_id = uuid.uuid4()
        await db_session.execute(
            text(
                "INSERT INTO knowledge_bases (id, tenant_id, name, status, visibility, "
                "confidentiality, created_by, default_chunking_strategy, "
                "default_chunk_size, default_chunk_overlap, document_count, "
                "created_at, updated_at) VALUES (:i, :t, 'acl-kb', 'active', 'tenant', "
                "'public', :u, 'recursive', 800, 100, 3, now(), now())"
            ),
            {"i": kb_id, "t": tenant_id, "u": user_id},
        )

        # 一个索引版本：三条 chunk 的向量都挂在它下面。
        # 两段式检索的第一段只在 `chunk_embeddings` 上走 HNSW——
        # 如果每行用不同的 index_version_id，第二段按 ID 取的过滤
        # 仍然能工作，但与真实结构不符（真实场景同一知识库共用一个 active 版本）。
        index_version_id = uuid.uuid4()
        await db_session.execute(
            text(
                "INSERT INTO index_versions (id, tenant_id, knowledge_base_id, "
                "version_tag, status, embedding_model, embedding_dim, "
                "chunk_count, expected_chunk_count, activated_at, "
                "created_at, updated_at) VALUES "
                "(:i, :t, :k, 'v1', 'active', 'test-model', 1024, 3, 3, now(), "
                "now(), now())"
            ),
            {"i": index_version_id, "t": tenant_id, "k": kb_id},
        )

        # 三个文档：内容里带唯一关键词，便于断言"检索到的是哪一条"
        for index, (level, keyword) in enumerate(
            (
                ("public", "ALPHAMARKER"),
                ("internal", "BETAMARKER"),
                ("secret", "GAMMAMARKER"),
            )
        ):
            doc_id = uuid.uuid4()
            version_id = uuid.uuid4()
            chunk_id = uuid.uuid4()
            blob_id = uuid.uuid4()
            docs[level] = doc_id

            # blob 是 document_versions.blob_id 的必需外键。
            # content_hash 有唯一约束 (tenant_id, content_hash)，所以
            # 用 keyword 生成不同的 hash，避免三行互相冲突。
            await db_session.execute(
                text(
                    "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, "
                    "storage_key, storage_backend, content_type, status, ref_count, "
                    "created_at, updated_at) VALUES (:i, :t, :h, 100, :k, 'local', "
                    "'text/markdown', 'complete', 1, now(), now())"
                ),
                {
                    "i": blob_id,
                    "t": tenant_id,
                    "h": f"{keyword.lower()}".ljust(64, "0")[:64],
                    "k": f"acl-test/{keyword.lower()}.md",
                },
            )

            await db_session.execute(
                text(
                    "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, "
                    "original_filename, format, source_type, confidentiality, status, "
                    "current_version_id, current_version_no, latest_version_no, "
                    "retry_count, chunk_count, size_bytes, created_by, version, "
                    "created_at, updated_at) VALUES (:i, :t, :k, :title, 'x.md', "
                    "'markdown', 'upload', :lvl, 'ready', NULL, 1, 1, 0, 1, 100, "
                    ":u, 1, now(), now())"
                ),
                {
                    "i": doc_id,
                    "t": tenant_id,
                    "k": kb_id,
                    "title": f"doc-{level}",
                    "lvl": level,
                    "u": user_id,
                },
            )
            await db_session.execute(
                text(
                    "INSERT INTO document_versions (id, tenant_id, document_id, version_no, "
                    "blob_id, status, chunking_strategy, chunking_config, chunk_count, "
                    "is_current, created_by, created_at, updated_at) VALUES "
                    "(:i, :t, :d, 1, :b, 'ready', 'recursive', '{}'::jsonb, 1, true, "
                    ":u, now(), now())"
                ),
                {
                    "i": version_id,
                    "t": tenant_id,
                    "d": doc_id,
                    "b": blob_id,
                    "u": user_id,
                },
            )
            await db_session.execute(
                text(
                    "UPDATE documents SET current_version_id = :v WHERE id = :d"
                ),
                {"v": version_id, "d": doc_id},
            )
            await db_session.execute(
                text(
                    "INSERT INTO chunks (id, tenant_id, knowledge_base_id, document_id, "
                    "document_version_id, chunk_index, content, chunk_level, is_current, "
                    "char_start, char_end, content_hash, token_count, chunk_type, "
                    "chunk_metadata, created_at, updated_at) VALUES "
                    "(:i, :t, :k, :d, :v, 0, :c, 0, true, 0, :len, :h, 1, 'text', "
                    "'{}'::jsonb, now(), now())"
                ),
                {
                    "i": chunk_id,
                    "t": tenant_id,
                    "k": kb_id,
                    "d": doc_id,
                    "v": version_id,
                    "c": f"{keyword} 的内容",
                    "len": len(f"{keyword} 的内容"),
                    # content_hash NOT NULL：用一个确定性的值，
                    # 不用真实 hash（本测试不校验内容完整性）
                    "h": f"{keyword.lower()}-hash".ljust(64, "0")[:64],
                },
            )
            # 向量写进旁表（A10 的两段式检索只读这里）
            await db_session.execute(
                text(
                    "INSERT INTO chunk_embeddings (id, tenant_id, chunk_id, "
                    "index_version_id, embedding, embedding_model, token_count, "
                    "created_at, updated_at) VALUES "
                    "(:i, :t, :c, :v, CAST(:e AS vector), 'test', 1, now(), now())"
                ),
                {
                    "i": uuid.uuid4(),
                    "t": tenant_id,
                    "c": chunk_id,
                    # 每个文档的向量略微偏离查询方向，保证排序稳定
                    "v": index_version_id,
                    "e": str(_unit_vector(index)),
                },
            )

    yield {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "kb_id": kb_id,
        "index_version_id": index_version_id,
        "docs": docs,
    }


@pytest.mark.integration
async def test_search_vector_filters_by_confidentiality(
    db_session: AsyncSession, seeded_tenant: dict[str, object]
) -> None:
    """**核心回归**：MEMBER 用向量检索时看不到 SECRET chunk。

    ## 为什么要用 `search_vector` 而不是直接查 documents

    漏洞就在检索路径上。直接查 documents 表能通过，但检索不传
    `access_filter` 时这条测试会红——这正是我们要钉住的行为。
    """
    tenant_id = seeded_tenant["tenant_id"]
    user_id = seeded_tenant["user_id"]
    assert isinstance(tenant_id, uuid.UUID)
    assert isinstance(user_id, uuid.UUID)

    # MEMBER：密级上限 INTERNAL，看不到 SECRET
    ctx = PermissionContext(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=frozenset({TenantRole.MEMBER}),
    )

    repo = ChunkRepository(db_session)
    hits = await repo.search_vector(
        tenant_id=tenant_id,
        query_embedding=_unit_vector(0),
        limit=10,
        access_filter=confidentiality_filter_for_chunks(ctx),
    )

    contents = [chunk.content for chunk, _distance in hits]
    assert any("ALPHAMARKER" in c for c in contents), "PUBLIC 文档应该能检索到"
    assert any("BETAMARKER" in c for c in contents), "INTERNAL 文档应该能检索到"
    assert not any("GAMMAMARKER" in c for c in contents), (
        "**SECRET 文档不该被 MEMBER 检索到**——"
        "过滤没接进 search_vector 时这条会红"
    )


@pytest.mark.integration
async def test_search_vector_without_filter_leaks_secret(
    db_session: AsyncSession, seeded_tenant: dict[str, object]
) -> None:
    """**破坏性对照**：不传 `access_filter` 时 SECRET 会泄漏。

    ## 为什么这条测试是"故意证明漏洞存在"

    上一条件测试证明"传了 filter 就拦住"。但如果不传 filter 时也拦住了，
    说明过滤是在别处做的（比如仓储内部硬编码），上一条件测试就没意义。

    这条断言"不传就泄漏"，从而证明：**安全性完全依赖于调用方
    记得传 `access_filter`**。所以 `ChatService._retrieve` 必须传，
    而这条测试是那个约定的书面记录。
    """
    tenant_id = seeded_tenant["tenant_id"]
    assert isinstance(tenant_id, uuid.UUID)

    repo = ChunkRepository(db_session)
    hits = await repo.search_vector(
        tenant_id=tenant_id,
        query_embedding=_unit_vector(0),
        limit=10,
        # 刻意不传 access_filter
    )

    contents = [chunk.content for chunk, _distance in hits]
    assert any("GAMMAMARKER" in c for c in contents), (
        "不传 access_filter 时应当能检索到 SECRET——"
        "如果这条失败，说明过滤被硬编码进了仓储，"
        "那么 ChatService 传不传 filter 就无所谓了（安全性不依赖调用方）"
    )


@pytest.mark.integration
async def test_viewer_sees_only_public(
    db_session: AsyncSession, seeded_tenant: dict[str, object]
) -> None:
    """VIEWER 的密级上限是 INTERNAL（不是 PUBLIC）。

    `ROLE_MAX_LEVEL[VIEWER] = INTERNAL` 是有意为之：只读用户不该
    "什么都看不到"——否则只读角色没有意义。这条测试把这个取舍钉住，
    避免有人"为了安全"把它改成 PUBLIC 而没人察觉。
    """
    tenant_id = seeded_tenant["tenant_id"]
    user_id = seeded_tenant["user_id"]
    assert isinstance(tenant_id, uuid.UUID)
    assert isinstance(user_id, uuid.UUID)

    ctx = PermissionContext(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=frozenset({TenantRole.VIEWER}),
    )
    assert ctx.max_confidentiality == ConfidentialityLevel.INTERNAL

    repo = ChunkRepository(db_session)
    hits = await repo.search_vector(
        tenant_id=tenant_id,
        query_embedding=_unit_vector(0),
        limit=10,
        access_filter=confidentiality_filter_for_chunks(ctx),
    )
    contents = [chunk.content for chunk, _distance in hits]
    assert any("ALPHAMARKER" in c for c in contents)
    assert any("BETAMARKER" in c for c in contents), "VIEWER 应该能看 INTERNAL"
    assert not any("GAMMAMARKER" in c for c in contents)


@pytest.mark.integration
async def test_search_keyword_also_filters(
    db_session: AsyncSession, seeded_tenant: dict[str, object]
) -> None:
    """关键词检索也必须过滤。

    ## 为什么单独测这一条

    双路检索（向量 + 关键词）做 RRF 融合。如果只给向量路加了 ACL
    而关键词路没加，融合结果里会混进越权内容——而且因为 RRF 只看
    排名，越权项可能被抬到很前面。**漏一处等于全漏**。
    """
    tenant_id = seeded_tenant["tenant_id"]
    user_id = seeded_tenant["user_id"]
    assert isinstance(tenant_id, uuid.UUID)
    assert isinstance(user_id, uuid.UUID)

    ctx = PermissionContext(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=frozenset({TenantRole.MEMBER}),
    )

    repo = ChunkRepository(db_session)
    hits = await repo.search_keyword(
        tenant_id=tenant_id,
        keywords=["GAMMAMARKER", "ALPHAMARKER", "BETAMARKER"],
        limit=10,
        access_filter=confidentiality_filter_for_chunks(ctx),
    )
    contents = [chunk.content for chunk, _score in hits]
    assert not any("GAMMAMARKER" in c for c in contents), (
        "关键词路也必须过滤 SECRET——漏一处等于全漏"
    )
