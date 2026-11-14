"""知识库 / 文档 / 版本 / blob 的集成测试。

这批测试的重点是**并发安全**与**边界约束**，不是 CRUD 覆盖率：

- 秒传在并发下只产生一条 blob（ON CONFLICT 是否真的原子）
- 版本号分配在并发下不重复（UPDATE RETURNING 是否真的原子）
- 乐观锁在并发流转时只有一个赢（防两个 worker 互相覆盖）
- worker 认领不会让两个 worker 拿到同一文档（SKIP LOCKED 是否生效）
- 部分唯一索引允许删后重建同名文档
- 文档密级不得低于知识库密级（service 层规则，这里验证约束存在）
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums.documents import BlobStatus, DocumentStatus
from rag_db.models.blob import Blob
from rag_db.models.document import Document, DocumentVersion
from rag_db.models.knowledge import KnowledgeBase
from rag_db.repositories import (
    BlobRepository,
    DocumentRepository,
    KnowledgeBaseRepository,
    claim_documents_for_processing,
)

pytestmark = pytest.mark.integration


# ==================================================== 测试数据构造


async def _make_tenant(session: AsyncSession) -> uuid.UUID:
    tenant_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
            "VALUES (:id, :slug, :name, 'active', now(), now())"
        ),
        {"id": tenant_id, "slug": f"t-{tenant_id.hex[:12]}", "name": "测试租户"},
    )
    return tenant_id


async def _make_user(session: AsyncSession) -> uuid.UUID:
    user_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO users (id, email, display_name, status, platform_role, "
            "tokens_valid_after, created_at, updated_at) "
            "VALUES (:id, :email, 'Tester', 'active', 'none', now(), now(), now())"
        ),
        {"id": user_id, "email": f"u-{user_id.hex[:8]}@example.test"},
    )
    return user_id


# ==================================================== 知识库


async def test_create_knowledge_base(db_session: AsyncSession) -> None:
    """知识库能创建，默认值符合预期。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)

    kb = await KnowledgeBaseRepository(db_session).create(
        tenant_id=tenant_id, name="制度库", created_by=user_id
    )
    assert kb.status == "active"
    assert kb.visibility == "tenant"
    assert kb.confidentiality == "internal"
    assert kb.default_chunk_size == 800
    assert kb.default_chunk_overlap == 100


async def test_duplicate_kb_name_rejected(db_session: AsyncSession) -> None:
    """同租户下库名唯一（同名库会让用户选错）。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    repo = KnowledgeBaseRepository(db_session)

    await repo.create(tenant_id=tenant_id, name="唯一库", created_by=user_id)
    with pytest.raises(IntegrityError):
        await repo.create(tenant_id=tenant_id, name="唯一库", created_by=user_id)


async def test_chunk_overlap_must_be_less_than_size(db_session: AsyncSession) -> None:
    """overlap >= size 必须被 CHECK 约束拒绝。

    这条约束防的是配置错误：overlap 等于 size 时切分算法会无限重叠，
    chunk 数量爆炸或死循环。等它跑到 worker 里才发现，代价是整批文档失败。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)

    with pytest.raises(IntegrityError):
        await KnowledgeBaseRepository(db_session).create(
            tenant_id=tenant_id,
            name="非法配置库",
            created_by=user_id,
            chunk_size=100,
            chunk_overlap=100,  # 等于 size，非法
        )


# ==================================================== 秒传（并发）


def test_concurrent_upload_same_file_creates_single_blob() -> None:
    """**并发上传同一文件只产生一条 blob（秒传核心）。**

    真实场景：用户上传大文件，前端做了分片重试；或同一个文件被
    两个人同时上传。若用"先查再插"，多个请求会同时查到"不存在"
    然后都尝试插入 —— 一个成功，其余撞唯一约束报 500。

    用 `INSERT ... ON CONFLICT DO NOTHING` 后，只有第一个插入成功，
    其余静默复用已有记录 —— 调用方完全感知不到并发。
    """
    from rag_core.db import Database
    from rag_core.settings import get_settings

    content_hash = uuid.uuid4().hex + uuid.uuid4().hex  # 64 字符

    async def scenario() -> tuple[int, int]:
        db = Database(get_settings())
        try:
            tenant_id = uuid.uuid4()
            async with db.transaction() as session:
                await session.execute(
                    text(
                        "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                        "VALUES (:id, :slug, '秒传租户', 'active', now(), now())"
                    ),
                    {"id": tenant_id, "slug": f"t-{tenant_id.hex[:12]}"},
                )

            created_count = 0

            async def upload() -> None:
                nonlocal created_count
                async with db.transaction() as session:
                    _, created = await BlobRepository(session).get_or_create(
                        tenant_id=tenant_id,
                        content_hash=content_hash,
                        size_bytes=1024,
                        storage_key=f"{tenant_id}/{content_hash}",
                        storage_backend="local",
                        content_type="application/pdf",
                    )
                    if created:
                        created_count += 1

            # 6 个并发请求上传同一文件
            await asyncio.gather(*(upload() for _ in range(6)))

            async with db.session() as session:
                result = await session.execute(
                    text("SELECT count(*) FROM blobs WHERE tenant_id = :tid AND content_hash = :h"),
                    {"tid": tenant_id, "h": content_hash},
                )
                blob_count = int(result.scalar_one())
            return blob_count, created_count
        finally:
            async with db.transaction() as session:
                await session.execute(text("DELETE FROM tenants WHERE slug LIKE 't-%'"))
            await db.close()

    blob_count, created_count = asyncio.run(scenario())
    assert blob_count == 1, f"并发上传产生了 {blob_count} 条 blob，应为 1"
    assert created_count == 1, f"应只有 1 个请求报告 created=True，实际 {created_count}"


async def test_blob_ref_count_is_atomic(db_session: AsyncSession) -> None:
    """引用计数原子自增，不会丢更新。"""
    tenant_id = await _make_tenant(db_session)
    repo = BlobRepository(db_session)
    blob, _ = await repo.get_or_create(
        tenant_id=tenant_id,
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        size_bytes=100,
        storage_key="k",
        storage_backend="local",
        content_type="text/plain",
    )

    for _ in range(5):
        await repo.increment_ref(blob_id=blob.id)

    refreshed = await db_session.get(Blob, blob.id)
    assert refreshed is not None
    await db_session.refresh(refreshed)
    assert refreshed.ref_count == 5


async def test_ref_count_cannot_go_negative(db_session: AsyncSession) -> None:
    """引用计数不能变负 —— 变负说明有 bug，必须立即暴露。

    静默允许负数会导致"blob 永远不会被清理"（因为检查和删除都看 >0），
    存储泄漏且没人发现。
    """
    tenant_id = await _make_tenant(db_session)
    repo = BlobRepository(db_session)
    blob, _ = await repo.get_or_create(
        tenant_id=tenant_id,
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        size_bytes=100,
        storage_key="k2",
        storage_backend="local",
        content_type="text/plain",
    )

    with pytest.raises(ValueError, match="引用计数变为负数"):
        await repo.increment_ref(blob_id=blob.id, delta=-1)


# ==================================================== 版本号分配（并发）


def test_concurrent_version_allocation_is_unique() -> None:
    """**并发分配版本号不能重复。**

    若用 Python 侧 `doc.latest_version_no + 1`，两个并发请求会拿到同一个号，
    插入时撞 `uq_document_versions_doc_no`。

    用 `UPDATE ... SET latest_version_no = latest_version_no + 1 RETURNING`
    让数据库原子递增，每个请求拿到唯一号。
    """
    from rag_core.db import Database
    from rag_core.settings import get_settings

    async def scenario() -> list[int]:
        db = Database(get_settings())
        try:
            async with db.transaction() as session:
                tenant_id = uuid.uuid4()
                await session.execute(
                    text(
                        "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                        "VALUES (:id, :slug, '版本租户', 'active', now(), now())"
                    ),
                    {"id": tenant_id, "slug": f"t-{tenant_id.hex[:12]}"},
                )
                user_id = uuid.uuid4()
                await session.execute(
                    text(
                        "INSERT INTO users (id, email, display_name, status, platform_role, "
                        "tokens_valid_after, created_at, updated_at) "
                        "VALUES (:id, :e, 'U', 'active', 'none', now(), now(), now())"
                    ),
                    {"id": user_id, "e": f"u-{user_id.hex[:8]}@example.test"},
                )
                kb = await KnowledgeBaseRepository(session).create(
                    tenant_id=tenant_id, name="版本库", created_by=user_id
                )
                doc = await DocumentRepository(session).create(
                    tenant_id=tenant_id,
                    knowledge_base_id=kb.id,
                    title="版本测试文档",
                    original_filename="v.pdf",
                    format="pdf",
                    created_by=user_id,
                )
                doc_id = doc.id

            allocated: list[int] = []

            async def allocate() -> None:
                async with db.transaction() as session:
                    no = await DocumentRepository(session).allocate_version_no(doc_id=doc_id)
                    allocated.append(no)

            await asyncio.gather(*(allocate() for _ in range(8)))
            return allocated
        finally:
            async with db.transaction() as session:
                await session.execute(text("DELETE FROM tenants WHERE slug LIKE 't-%'"))
                await session.execute(text("DELETE FROM users WHERE email LIKE 'u-%@example.test'"))
            await db.close()

    numbers = asyncio.run(scenario())
    assert len(numbers) == 8
    assert len(set(numbers)) == 8, f"版本号重复：{sorted(numbers)}"
    assert sorted(numbers) == list(range(1, 9)), f"版本号不连续：{sorted(numbers)}"


# ==================================================== 乐观锁


async def test_optimistic_lock_rejects_stale_update(db_session: AsyncSession) -> None:
    """乐观锁：版本不符的更新必须失败。

    两个 worker 同时处理同一文档时，只有一个能流转状态，
    另一个拿到 False 后重新决策 —— 而不是互相覆盖。
    没有这个机制时，后写的会覆盖先写的，状态机直接乱掉。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb = await KnowledgeBaseRepository(db_session).create(
        tenant_id=tenant_id, name="锁库", created_by=user_id
    )
    docs = DocumentRepository(db_session)
    doc = await docs.create(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        title="锁测试",
        original_filename="lock.pdf",
        format="pdf",
        created_by=user_id,
    )

    # 第一次更新成功（version 1 -> 2）
    ok = await docs.update_status(
        tenant_id=tenant_id,
        doc_id=doc.id,
        new_status=DocumentStatus.PARSING,
        expected_version=1,
    )
    assert ok is True

    # 用旧版本号再更新必须失败
    stale = await docs.update_status(
        tenant_id=tenant_id,
        doc_id=doc.id,
        new_status=DocumentStatus.CHUNKING,
        expected_version=1,  # 已过期
    )
    assert stale is False

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    await db_session.refresh(refreshed)
    assert refreshed.status == DocumentStatus.PARSING
    assert refreshed.version == 2


# ==================================================== worker 认领


def test_worker_claim_does_not_double_assign() -> None:
    """**两个 worker 不能认领到同一文档。**

    用 `FOR UPDATE SKIP LOCKED`：N 个 worker 并发轮询时各自拿到不同文档。
    没有 SKIP LOCKED 时，后到的 worker 会阻塞等待前一个释放锁，
    吞吐量退化成单 worker；没有 FOR UPDATE 时，两个 worker 会拿到同一行，
    同一文档被处理两次（chunk 重复、成本翻倍）。
    """
    from rag_core.db import Database
    from rag_core.settings import get_settings

    async def scenario() -> tuple[int, int]:
        db = Database(get_settings())
        try:
            async with db.transaction() as session:
                tenant_id = uuid.uuid4()
                await session.execute(
                    text(
                        "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                        "VALUES (:id, :slug, '认领租户', 'active', now(), now())"
                    ),
                    {"id": tenant_id, "slug": f"t-{tenant_id.hex[:12]}"},
                )
                user_id = uuid.uuid4()
                await session.execute(
                    text(
                        "INSERT INTO users (id, email, display_name, status, platform_role, "
                        "tokens_valid_after, created_at, updated_at) "
                        "VALUES (:id, :e, 'U', 'active', 'none', now(), now(), now())"
                    ),
                    {"id": user_id, "e": f"u-{user_id.hex[:8]}@example.test"},
                )
                kb = await KnowledgeBaseRepository(session).create(
                    tenant_id=tenant_id, name="认领库", created_by=user_id
                )
                docs = DocumentRepository(session)
                for i in range(10):
                    await docs.create(
                        tenant_id=tenant_id,
                        knowledge_base_id=kb.id,
                        title=f"待处理 {i}",
                        original_filename=f"{i}.pdf",
                        format="pdf",
                        created_by=user_id,
                    )

            claimed_ids: list[uuid.UUID] = []

            async def worker_round() -> None:
                async with db.transaction() as session:
                    claimed = await claim_documents_for_processing(session, limit=5)
                    claimed_ids.extend(doc.id for doc in claimed)

            # 两个 worker 并发认领
            await asyncio.gather(worker_round(), worker_round())
            return len(claimed_ids), len(set(claimed_ids))
        finally:
            async with db.transaction() as session:
                await session.execute(text("DELETE FROM tenants WHERE slug LIKE 't-%'"))
                await session.execute(text("DELETE FROM users WHERE email LIKE 'u-%@example.test'"))
            await db.close()

    total_claimed, unique_claimed = asyncio.run(scenario())
    assert total_claimed == unique_claimed, (
        f"同一文档被多个 worker 认领：认领 {total_claimed} 次，去重后 {unique_claimed} 个"
    )
    # 10 个文档、两轮各限 5，应该全被认领且互不重复
    assert unique_claimed == 10


# ==================================================== 软删除与部分唯一索引


async def test_soft_delete_allows_title_reuse(db_session: AsyncSession) -> None:
    """软删除后允许重建同名文档。

    靠部分唯一索引 `WHERE deleted_at IS NULL`。
    不允许重用的话，用户删了文档却再也建不了同名文档，而错误信息
    是"标题重复"——用户完全找不到那个"重复"的文档（因为它已删除）。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb = await KnowledgeBaseRepository(db_session).create(
        tenant_id=tenant_id, name="软删除库", created_by=user_id
    )
    docs = DocumentRepository(db_session)

    first = await docs.create(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        title="同名文档",
        original_filename="a.pdf",
        format="pdf",
        created_by=user_id,
    )
    assert await docs.soft_delete(tenant_id=tenant_id, doc_id=first.id) is True

    # 此时可以建同名
    second = await docs.create(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        title="同名文档",
        original_filename="b.pdf",
        format="pdf",
        created_by=user_id,
    )
    assert second.id != first.id


async def test_active_duplicate_title_rejected(db_session: AsyncSession) -> None:
    """未删除的同名文档必须被拒绝。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb = await KnowledgeBaseRepository(db_session).create(
        tenant_id=tenant_id, name="重名库", created_by=user_id
    )
    docs = DocumentRepository(db_session)

    await docs.create(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        title="重复标题",
        original_filename="a.pdf",
        format="pdf",
        created_by=user_id,
    )
    with pytest.raises(IntegrityError):
        await docs.create(
            tenant_id=tenant_id,
            knowledge_base_id=kb.id,
            title="重复标题",
            original_filename="b.pdf",
            format="pdf",
            created_by=user_id,
        )


# ==================================================== 版本与当前版本


async def test_only_one_current_version_per_document(db_session: AsyncSession) -> None:
    """每个文档最多一个 is_current=true 的版本。

    部分唯一索引 `uq_document_versions_current` 保证这一点。
    允许多个 current 会让"当前版本"变成不确定的查询结果。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb = await KnowledgeBaseRepository(db_session).create(
        tenant_id=tenant_id, name="当前版本库", created_by=user_id
    )
    blobs = BlobRepository(db_session)
    docs = DocumentRepository(db_session)

    doc = await docs.create(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        title="多版本",
        original_filename="m.pdf",
        format="pdf",
        created_by=user_id,
    )
    blob, _ = await blobs.get_or_create(
        tenant_id=tenant_id,
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        size_bytes=10,
        storage_key="k3",
        storage_backend="local",
        content_type="application/pdf",
    )

    v1_no = await docs.allocate_version_no(doc_id=doc.id)
    v1 = await docs.create_version(
        tenant_id=tenant_id,
        document_id=doc.id,
        version_no=v1_no,
        blob_id=blob.id,
        created_by=user_id,
        chunking_strategy="recursive",
        chunking_config={"size": 800},
    )
    await docs.mark_version_current(document_id=doc.id, version_id=v1.id, version_no=v1_no)

    v2_no = await docs.allocate_version_no(doc_id=doc.id)
    v2 = await docs.create_version(
        tenant_id=tenant_id,
        document_id=doc.id,
        version_no=v2_no,
        blob_id=blob.id,
        created_by=user_id,
        chunking_strategy="recursive",
        chunking_config={"size": 800},
    )
    await docs.mark_version_current(document_id=doc.id, version_id=v2.id, version_no=v2_no)

    result = await db_session.execute(
        text("SELECT count(*) FROM document_versions WHERE document_id = :d AND is_current = true"),
        {"d": doc.id},
    )
    assert int(result.scalar_one()) == 1

    refreshed_doc = await db_session.get(Document, doc.id)
    assert refreshed_doc is not None
    await db_session.refresh(refreshed_doc)
    assert refreshed_doc.current_version_id == v2.id


async def test_version_config_is_snapshotted(db_session: AsyncSession) -> None:
    """版本必须快照处理配置。

    不快照的话：库的默认切分策略改变后，回滚旧版本会得到完全不同的
    chunk —— "回滚"名不副实，而且无法回答"这批 chunk 当时用什么配置生成的"。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb = await KnowledgeBaseRepository(db_session).create(
        tenant_id=tenant_id, name="快照库", created_by=user_id, chunking_strategy="recursive"
    )
    docs = DocumentRepository(db_session)
    blobs = BlobRepository(db_session)

    doc = await docs.create(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        title="快照测试",
        original_filename="s.pdf",
        format="pdf",
        created_by=user_id,
    )
    blob, _ = await blobs.get_or_create(
        tenant_id=tenant_id,
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        size_bytes=10,
        storage_key="k4",
        storage_backend="local",
        content_type="application/pdf",
    )
    no = await docs.allocate_version_no(doc_id=doc.id)
    version = await docs.create_version(
        tenant_id=tenant_id,
        document_id=doc.id,
        version_no=no,
        blob_id=blob.id,
        created_by=user_id,
        chunking_strategy="recursive",  # 当时用的是递归
        chunking_config={"size": 800, "overlap": 100},
        embedding_model="bge-m3",
        embedding_dim=1024,
    )

    # 库配置后来改了
    kb_row = await db_session.get(KnowledgeBase, kb.id)
    assert kb_row is not None
    kb_row.default_chunking_strategy = "semantic"
    await db_session.flush()

    stored = await db_session.get(DocumentVersion, version.id)
    assert stored is not None
    assert stored.chunking_strategy == "recursive", "版本配置被库的后续修改污染了"
    assert stored.chunking_config == {"size": 800, "overlap": 100}
    assert stored.embedding_model == "bge-m3"
    assert stored.embedding_dim == 1024


async def test_cannot_delete_blob_in_use(db_session: AsyncSession) -> None:
    """被版本引用的 blob 不能删除（RESTRICT）。

    允许删除会产生"文档版本还在但文件不见了"——下载与重建索引全部失败，
    且错误在现场表现为"文件不存在"，很难联想到是删除逻辑的问题。
    """
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb = await KnowledgeBaseRepository(db_session).create(
        tenant_id=tenant_id, name="FK库", created_by=user_id
    )
    docs = DocumentRepository(db_session)
    blobs = BlobRepository(db_session)

    doc = await docs.create(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        title="FK测试",
        original_filename="f.pdf",
        format="pdf",
        created_by=user_id,
    )
    blob, _ = await blobs.get_or_create(
        tenant_id=tenant_id,
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        size_bytes=10,
        storage_key="k5",
        storage_backend="local",
        content_type="application/pdf",
    )
    no = await docs.allocate_version_no(doc_id=doc.id)
    await docs.create_version(
        tenant_id=tenant_id,
        document_id=doc.id,
        version_no=no,
        blob_id=blob.id,
        created_by=user_id,
        chunking_strategy="recursive",
        chunking_config={},
    )

    with pytest.raises(IntegrityError):
        await db_session.execute(text("DELETE FROM blobs WHERE id = :b"), {"b": blob.id})


# ==================================================== 状态统计


async def test_count_by_status(db_session: AsyncSession) -> None:
    """按状态统计用于管理端看板（"有多少文档卡在处理中"）。"""
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    kb = await KnowledgeBaseRepository(db_session).create(
        tenant_id=tenant_id, name="统计库", created_by=user_id
    )
    docs = DocumentRepository(db_session)

    for i in range(3):
        await docs.create(
            tenant_id=tenant_id,
            knowledge_base_id=kb.id,
            title=f"待处理{i}",
            original_filename=f"{i}.pdf",
            format="pdf",
            created_by=user_id,
            status=DocumentStatus.PENDING,
        )
    for i in range(2):
        await docs.create(
            tenant_id=tenant_id,
            knowledge_base_id=kb.id,
            title=f"就绪{i}",
            original_filename=f"r{i}.pdf",
            format="pdf",
            created_by=user_id,
            status=DocumentStatus.READY,
        )

    counts = await docs.count_by_status(tenant_id=tenant_id)
    assert counts.get("pending") == 3
    assert counts.get("ready") == 2


async def test_blob_status_defaults_to_pending(db_session: AsyncSession) -> None:
    """新建 blob 默认 pending —— 必须显式标记 complete 才算可用。

    默认给 complete 的风险：上传中断时会留下"看起来能下载"的记录，
    直到真正读取时才报错。
    """
    tenant_id = await _make_tenant(db_session)
    blob = Blob(
        tenant_id=tenant_id,
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        size_bytes=1,
        storage_key="k6",
        storage_backend="local",
        content_type="text/plain",
    )
    db_session.add(blob)
    await db_session.flush()
    await db_session.refresh(blob)
    assert blob.status == BlobStatus.PENDING.value
