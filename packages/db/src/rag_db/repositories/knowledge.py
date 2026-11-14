"""知识库 / 文档 / 版本 / 物理文件仓储。

三条设计要点：

1. **秒传路径必须原子**：`get_or_create_blob` 用 PostgreSQL 的
   `INSERT ... ON CONFLICT DO NOTHING RETURNING`，而不是"先查再插"。
   并发上传同一文件时，先查再插会撞唯一约束报错，而 ON CONFLICT
   能静默复用已有记录并返回它。

2. **版本号分配必须原子**：用 `UPDATE documents SET latest_version_no =
   latest_version_no + 1 RETURNING latest_version_no`，一条语句完成
   "读-加-写"。用 Python 侧 `doc.latest_version_no + 1` 会并发丢更新。

3. **软删除 + 部分唯一索引**：删掉文档后允许重建同名文档，
   靠 `uq_documents_kb_title_active` 的 `WHERE deleted_at IS NULL`。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums.documents import (
    BlobStatus,
    DocumentStatus,
    DocumentVersionStatus,
    KnowledgeBaseStatus,
)
from rag_db.models.blob import Blob
from rag_db.models.document import Document, DocumentVersion
from rag_db.models.knowledge import KnowledgeBase


class KnowledgeBaseRepository:
    """知识库数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        name: str,
        created_by: uuid.UUID,
        description: str | None = None,
        visibility: str = "tenant",
        confidentiality: str = "internal",
        chunking_strategy: str = "recursive",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> KnowledgeBase:
        kb = KnowledgeBase(
            tenant_id=tenant_id,
            name=name,
            created_by=created_by,
            description=description,
            visibility=visibility,
            confidentiality=confidentiality,
            default_chunking_strategy=chunking_strategy,
            default_chunk_size=chunk_size,
            default_chunk_overlap=chunk_overlap,
        )
        self._session.add(kb)
        await self._session.flush()
        return kb

    async def get(self, *, tenant_id: uuid.UUID, kb_id: uuid.UUID) -> KnowledgeBase | None:
        """按租户 + ID 取知识库。

        **必须带 tenant_id**：只用 kb_id 查询等于把租户隔离交给上层自觉。
        RLS 是最后一道防线，仓储层自己也要写对。
        """
        result = await self._session.execute(
            select(KnowledgeBase)
            .where(KnowledgeBase.id == kb_id, KnowledgeBase.tenant_id == tenant_id)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, *, tenant_id: uuid.UUID, name: str) -> KnowledgeBase | None:
        result = await self._session.execute(
            select(KnowledgeBase)
            .where(KnowledgeBase.tenant_id == tenant_id, KnowledgeBase.name == name)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_active(
        self,
        *,
        tenant_id: uuid.UUID,
        limit: int = 50,
        after_id: uuid.UUID | None = None,
    ) -> list[KnowledgeBase]:
        """keyset 分页列出活跃知识库。

        排序键用 `(updated_at DESC, id DESC)`：只按 updated_at 排序时，
        同一毫秒更新的两行顺序不稳定，分页会漏读或重复。
        id 作为决胜键保证全序。
        """
        stmt = (
            select(KnowledgeBase)
            .where(
                KnowledgeBase.tenant_id == tenant_id,
                KnowledgeBase.status == KnowledgeBaseStatus.ACTIVE,
            )
            .order_by(KnowledgeBase.updated_at.desc(), KnowledgeBase.id.desc())
            .limit(limit)
        )
        if after_id is not None:
            # keyset：取 updated_at 早于游标行的，或同一时刻但 id 更小的
            cursor_row = await self._session.get(KnowledgeBase, after_id)
            if cursor_row is not None:
                stmt = stmt.where(
                    (KnowledgeBase.updated_at < cursor_row.updated_at)
                    | (
                        (KnowledgeBase.updated_at == cursor_row.updated_at)
                        & (KnowledgeBase.id < cursor_row.id)
                    )
                )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def increment_document_count(self, *, kb_id: uuid.UUID, delta: int) -> None:
        """原子调整文档计数。

        用 `col = col + delta` 让数据库做加法，避免"读出再写回"丢更新。
        """
        await self._session.execute(
            update(KnowledgeBase)
            .where(KnowledgeBase.id == kb_id)
            .values(document_count=KnowledgeBase.document_count + delta)
        )


class BlobRepository:
    """物理文件数据访问。秒传逻辑在这里。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(
        self,
        *,
        tenant_id: uuid.UUID,
        content_hash: str,
        size_bytes: int,
        storage_key: str,
        storage_backend: str,
        content_type: str,
        status: BlobStatus = BlobStatus.COMPLETE,
    ) -> tuple[Blob, bool]:
        """按内容 hash 取或创建 blob。

        返回 `(blob, created)`。`created=False` 表示秒传命中。

        **必须用 ON CONFLICT DO NOTHING**：
        两个客户端同时上传同一文件时，"先 SELECT 再 INSERT"会有一个
        撞唯一约束抛 IntegrityError，调用方还得重试一次查询。
        ON CONFLICT 让数据库原子地完成"存在则忽略"，随后统一查一次。

        注意这里**不更新**已有记录的 size/storage_key：
        同一 hash 的内容必然相同，这些字段也该相同；若不同说明
        有 bug（hash 碰撞或数据损坏），覆盖反而会掩盖问题。
        """
        stmt = (
            pg_insert(Blob)
            .values(
                tenant_id=tenant_id,
                content_hash=content_hash,
                size_bytes=size_bytes,
                storage_key=storage_key,
                storage_backend=storage_backend,
                content_type=content_type,
                status=status,
                ref_count=0,
            )
            # 冲突时不插入，返回空 —— 后续统一查一次拿到已有记录
            .on_conflict_do_nothing(index_elements=["tenant_id", "content_hash"])
            .returning(Blob)
        )
        result = await self._session.execute(stmt)
        created_blob = result.scalar_one_or_none()

        if created_blob is not None:
            return created_blob, True

        # 冲突：已被并发请求或历史上传创建，查出已有记录
        existing = await self._session.execute(
            select(Blob)
            .where(Blob.tenant_id == tenant_id, Blob.content_hash == content_hash)
            .limit(1)
        )
        blob = existing.scalar_one()
        return blob, False

    async def increment_ref(self, *, blob_id: uuid.UUID, delta: int = 1) -> int:
        """原子调整引用计数，返回新值。

        用 `ref_count = ref_count + delta`：并发引用同一 blob 时
        （多个文档同时上传同一个文件）必须原子自增，否则计数会少算，
        导致 blob 被提前物理删除 —— 后果是"文档还在但文件不见了"。
        """
        result = await self._session.execute(
            update(Blob)
            .where(Blob.id == blob_id)
            .values(ref_count=Blob.ref_count + delta)
            .returning(Blob.ref_count)
        )
        new_count = result.scalar_one()
        if new_count < 0:
            # 负引用说明计数逻辑有 bug，立即暴露而不是静默继续
            raise ValueError(f"blob {blob_id} 引用计数变为负数：{new_count}")
        return int(new_count)

    async def find_by_hash(self, *, tenant_id: uuid.UUID, content_hash: str) -> Blob | None:
        """秒传的查询入口。"""
        result = await self._session.execute(
            select(Blob)
            .where(Blob.tenant_id == tenant_id, Blob.content_hash == content_hash)
            .limit(1)
        )
        return result.scalar_one_or_none()


class DocumentRepository:
    """文档与版本文档数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        title: str,
        original_filename: str,
        format: str,
        created_by: uuid.UUID,
        confidentiality: str = "internal",
        source_type: str = "upload",
        status: DocumentStatus = DocumentStatus.PENDING,
    ) -> Document:
        doc = Document(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            title=title,
            original_filename=original_filename,
            format=format,
            created_by=created_by,
            confidentiality=confidentiality,
            source_type=source_type,
            status=status,
        )
        self._session.add(doc)
        await self._session.flush()
        return doc

    async def get(
        self, *, tenant_id: uuid.UUID, doc_id: uuid.UUID, include_deleted: bool = False
    ) -> Document | None:
        """取文档。默认排除软删除的。"""
        stmt = select(Document).where(Document.id == doc_id, Document.tenant_id == tenant_id)
        if not include_deleted:
            stmt = stmt.where(Document.deleted_at.is_(None))
        result = await self._session.execute(stmt.limit(1))
        return result.scalar_one_or_none()

    async def list_by_kb(
        self,
        *,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        limit: int = 50,
        after_id: uuid.UUID | None = None,
    ) -> list[Document]:
        stmt = (
            select(Document)
            .where(
                Document.tenant_id == tenant_id,
                Document.knowledge_base_id == kb_id,
                Document.deleted_at.is_(None),
            )
            .order_by(Document.updated_at.desc(), Document.id.desc())
            .limit(limit)
        )
        if after_id is not None:
            cursor_row = await self._session.get(Document, after_id)
            if cursor_row is not None:
                stmt = stmt.where(
                    (Document.updated_at < cursor_row.updated_at)
                    | (
                        (Document.updated_at == cursor_row.updated_at)
                        & (Document.id < cursor_row.id)
                    )
                )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def allocate_version_no(self, *, doc_id: uuid.UUID) -> int:
        """原子分配下一个版本号，返回新版本号。

        **必须用单条 UPDATE ... RETURNING**：
        Python 侧 `doc.latest_version_no + 1` 在并发下会两个请求拿到同一个号，
        随后插入时撞 `uq_document_versions_doc_no` 唯一约束 ——
        报错信息是"版本号重复"，但根因是读取方式不对。

        这条语句同时递增 `latest_version_no`，不需要额外更新文档行。
        """
        result = await self._session.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(latest_version_no=Document.latest_version_no + 1)
            .returning(Document.latest_version_no)
        )
        new_no = result.scalar_one()
        return int(new_no)

    async def create_version(
        self,
        *,
        tenant_id: uuid.UUID,
        document_id: uuid.UUID,
        version_no: int,
        blob_id: uuid.UUID,
        created_by: uuid.UUID,
        chunking_strategy: str,
        chunking_config: dict[str, Any],
        parser_name: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        status: DocumentVersionStatus = DocumentVersionStatus.PROCESSING,
    ) -> DocumentVersion:
        version = DocumentVersion(
            tenant_id=tenant_id,
            document_id=document_id,
            version_no=version_no,
            blob_id=blob_id,
            created_by=created_by,
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
            parser_name=parser_name,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            status=status,
        )
        self._session.add(version)
        await self._session.flush()
        return version

    async def mark_version_current(
        self, *, document_id: uuid.UUID, version_id: uuid.UUID, version_no: int
    ) -> None:
        """把某个版本设为当前生效版本。

        两步都要做：
        1. 先把该文档所有版本的 `is_current` 清成 false
           （部分唯一索引 `uq_document_versions_current` 只允许一条 true）；
        2. 再把目标版本设为 true，并更新文档的指针。

        顺序不能反：先设 true 会与旧版本冲突。
        """
        await self._session.execute(
            update(DocumentVersion)
            .where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.is_current.is_(True),
            )
            .values(is_current=False)
        )
        await self._session.execute(
            update(DocumentVersion).where(DocumentVersion.id == version_id).values(is_current=True)
        )
        await self._session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(current_version_id=version_id, current_version_no=version_no)
        )

    async def supersede_older_versions(
        self, *, document_id: uuid.UUID, keep_version_no: int
    ) -> None:
        """把除 keep 以外的 ready 版本标记为 superseded。

        保留数据（供回滚与审计），只改状态。
        """
        await self._session.execute(
            update(DocumentVersion)
            .where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.version_no != keep_version_no,
                DocumentVersion.status == DocumentVersionStatus.READY,
            )
            .values(status=DocumentVersionStatus.SUPERSEDED, superseded_at=func.now())
        )

    async def list_versions(
        self, *, document_id: uuid.UUID, limit: int = 100
    ) -> Sequence[DocumentVersion]:
        result = await self._session.execute(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_no.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def update_status(
        self,
        *,
        tenant_id: uuid.UUID,
        doc_id: uuid.UUID,
        new_status: DocumentStatus,
        expected_version: int,
        **fields: Any,
    ) -> bool:
        """乐观锁更新文档状态。

        返回 False 表示版本不符（并发冲突），调用方应重新读取后决策。

        `WHERE version = :expected` 是乐观锁的核心：
        两个 worker 同时处理同一文档时，只有一个能成功流转状态，
        另一个会拿到 0 行受影响并重新决策 —— 而不是互相覆盖。
        """
        result = await self._session.execute(
            update(Document)
            .where(
                Document.id == doc_id,
                Document.tenant_id == tenant_id,
                Document.version == expected_version,
            )
            .values(
                status=new_status,
                version=Document.version + 1,
                **fields,
            )
            .returning(Document.version)
        )
        return result.scalar_one_or_none() is not None

    async def soft_delete(self, *, tenant_id: uuid.UUID, doc_id: uuid.UUID) -> bool:
        """软删除文档。

        不物理删的原因：chunk 与索引清理是异步的，立即删会留下孤儿 chunk；
        且审计需要知道"这个文档曾经存在过、谁删的"。
        """
        result = await self._session.execute(
            update(Document)
            .where(
                Document.id == doc_id,
                Document.tenant_id == tenant_id,
                Document.deleted_at.is_(None),
            )
            .values(deleted_at=func.now(), status=DocumentStatus.PENDING)
            .returning(Document.id)
        )
        return result.scalar_one_or_none() is not None

    async def count_by_status(self, *, tenant_id: uuid.UUID) -> dict[str, int]:
        """按状态统计文档数（管理端看板用）。"""
        result = await self._session.execute(
            select(Document.status, func.count())
            .where(Document.tenant_id == tenant_id, Document.deleted_at.is_(None))
            .group_by(Document.status)
        )
        return {str(row[0]): int(row[1]) for row in result}


async def claim_documents_for_processing(
    session: AsyncSession,
    *,
    limit: int = 10,
    from_status: DocumentStatus = DocumentStatus.PENDING,
    to_status: DocumentStatus = DocumentStatus.PARSING,
) -> list[Document]:
    """原子认领待处理文档。

    **这是 worker 并发安全的核心**：多个 worker 同时轮询时，
    必须保证同一文档不被两个 worker 处理。

    实现用 `UPDATE ... WHERE id IN (SELECT ... FOR UPDATE SKIP LOCKED)
    RETURNING`：
    - `FOR UPDATE` 锁住候选行；
    - `SKIP LOCKED` 让其它 worker 跳过已锁定的行，而不是阻塞等待 ——
      没有它，N 个 worker 会串行排队，吞吐量掉到单 worker 水平；
    - `RETURNING` 一次返回认领成功的行，不需要二次查询。
    """
    subq = (
        select(Document.id)
        .where(
            Document.status == from_status,
            Document.deleted_at.is_(None),
        )
        .order_by(Document.created_at)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    result = await session.execute(
        update(Document)
        .where(Document.id.in_(subq))
        .values(status=to_status, version=Document.version + 1)
        .returning(Document)
    )
    return list(result.scalars())


async def count_orphan_blobs(session: AsyncSession, *, tenant_id: uuid.UUID) -> int:
    """统计可回收的 blob（引用为 0 且已完成）。

    对账任务用：正常情况下 ref_count=0 的 complete blob 是"上传了
    但没建成文档"的残留，应当清理。这个函数供清理任务与测试使用。
    """
    result = await session.execute(
        text(
            "SELECT count(*) FROM blobs WHERE tenant_id = :tid "
            "AND status = 'complete' AND ref_count = 0"
        ),
        {"tid": tenant_id},
    )
    return int(result.scalar_one())
