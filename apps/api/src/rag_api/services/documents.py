"""文档生命周期服务：登记 / 查询 / 删除 / 重试。

## 为什么这些代码不在路由里

AGENTS.md 的分层要求是「routes 只做参数接收与响应，业务在 services」。
路由里出现 blob 引用计数、库计数、审计写入、重试状态机，会带来两个问题：

1. **同一条规则散落多处**：删除文档要同时处理软删、blob 引用、库计数、
   审计——写在路由里就没法被 worker 或定时任务复用；
2. **难以独立测试**：路由测试需要构造 HTTP 上下文，业务测试不需要。

本模块只返回领域对象（ORM / 元组），**不感知 HTTP 响应契约**；
响应实体的映射在 `rag_api.schemas.documents`。
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.permissions.acl import PermissionContext, document_access_filter
from rag_api.services.audit import AuditService
from rag_api.services.ingestion import IngestionResult, IngestionService
from rag_contracts.api.knowledge import DocumentDownloadResponse
from rag_contracts.enums import (
    AuditAction,
    AuditTargetType,
    BlobStatus,
    ConfidentialityLevel,
    DocumentStatus,
    DocumentVersionStatus,
    TaskType,
)
from rag_core.errors import ConflictError, NotFoundError, ValidationFailedError
from rag_core.logging import get_logger
from rag_db.models.document import Document, DocumentVersion
from rag_db.repositories import (
    BlobRepository,
    ChunkRepository,
    DocumentRepository,
    KnowledgeBaseRepository,
    TaskRepository,
)

log = get_logger(__name__)

# 类里的方法名 `list` 会遮蔽内建 `list`，后续方法的注解里不能再写
# `list[Document]`（mypy 会把它解析成方法）。用模块级别名规避。
type _DocumentList = list[Document]
type _DocumentVersionList = list[DocumentVersion]


class DocumentService:
    """文档资源的业务编排。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._docs = DocumentRepository(session)
        self._blobs = BlobRepository(session)
        self._kbs = KnowledgeBaseRepository(session)
        self._tasks = TaskRepository(session)
        # chunk 仓储：版本回滚要切换 is_current 标记
        self._chunks = ChunkRepository(session)

    # ------------------------------------------------------------ 创建

    async def create(
        self,
        *,
        ctx: PermissionContext,
        blob_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        original_filename: str,
        title: str | None = None,
        confidentiality: ConfidentialityLevel | None = None,
    ) -> IngestionResult:
        """把已上传的 blob 登记为文档并触发解析。"""
        return await IngestionService(self._session).create_document(
            ctx=ctx,
            blob_id=blob_id,
            knowledge_base_id=knowledge_base_id,
            original_filename=original_filename,
            title=title,
            confidentiality=confidentiality,
        )

    # ------------------------------------------------------------ 查询

    async def list(
        self,
        *,
        ctx: PermissionContext,
        limit: int,
        offset: int,
        include_total: bool,
        kb_id: uuid.UUID | None = None,
        status: str | None = None,
    ) -> tuple[_DocumentList, int | None]:
        """按 ACL 过滤的文档列表。

        `status` 的合法性在这里校验（而不是静默忽略）：未知值会让
        「我只想看失败的文档」拿到全量数据，用户据此误判没有失败文档。
        """
        status_value = _parse_status(status)
        rows, total = await self._docs.list_by_kb_filtered(
            tenant_id=ctx.tenant_id,
            kb_id=kb_id,
            limit=limit,
            offset=offset,
            # **ACL 必须在 SQL 里**：查出来再过滤既浪费 I/O，
            # 又会让越权内容进应用内存（日志/异常都可能带出标题）。
            access_filter=document_access_filter(ctx),
            status=status_value,
            include_total=include_total,
        )
        return list(rows), total

    async def get(
        self, *, ctx: PermissionContext, doc_id: uuid.UUID
    ) -> tuple[Document, _DocumentVersionList]:
        """文档详情（含版本历史）。不可读或不存在都返回 404。"""
        doc = await self._require_visible(ctx=ctx, doc_id=doc_id)
        versions = await self._docs.list_versions(document_id=doc.id)
        return doc, list(versions)

    # ------------------------------------------------------------ 删除

    async def delete(self, *, ctx: PermissionContext, doc_id: uuid.UUID) -> None:
        """软删除文档，并维护 blob 引用与库计数。

        三件事必须一起做，缺任何一件都会留下不可见的不一致：
        - 软删本身；
        - blob 引用计数减 1（否则存储永不释放）；
        - 库文档计数减 1（否则库永远显示还有文档却删不掉）。
        """
        doc = await self._require_visible(ctx=ctx, doc_id=doc_id)

        deleted = await self._docs.soft_delete(tenant_id=ctx.tenant_id, doc_id=doc_id)
        if not deleted:
            # 并发删除：两个请求同时进来，一个成功一个拿到 0 行。
            # 对用户来说资源已不存在，返回 404 比 500 更准确。
            raise NotFoundError("文档不存在")

        version = await self._docs.get_current_version(doc_id=doc_id)
        if version is not None:
            try:
                await self._blobs.increment_ref(blob_id=version.blob_id, delta=-1)
            except ValueError:
                # 计数减到负数说明数据已经不一致（可能重复删除）。
                # 不回滚删除（用户要的结果已经达成），但要留下明确日志
                # 供对账任务发现。
                log.error(
                    "blob_ref_count_underflow",
                    doc_id=str(doc_id),
                    blob_id=str(version.blob_id),
                )

        await self._kbs.increment_document_count(kb_id=doc.knowledge_base_id, delta=-1)

        await AuditService(self._session).record(
            ctx=ctx,
            action=AuditAction.DOC_DELETE,
            target_type=AuditTargetType.DOCUMENT,
            target_id=doc.id,
            target_name=doc.title,
            details={"knowledge_base_id": str(doc.knowledge_base_id)},
        )
        log.info(
            "document_deleted",
            doc_id=str(doc_id),
            deleted_by=str(ctx.user_id),
        )

    # ------------------------------------------------------------ 重试

    async def retry(
        self, *, ctx: PermissionContext, doc_id: uuid.UUID
    ) -> tuple[uuid.UUID, uuid.UUID]:
        """重试失败的文档：重置状态并重新入队解析。

        返回 `(document_id, task_id)`。只有 `failed` 可重试。其他状态
        重试会排出第二个解析任务，worker 的 `retry_count` 语义也被污染
        （用户重试与自动重试混在一起）。
        """
        doc = await self._require_visible(ctx=ctx, doc_id=doc_id)

        # 先看状态再改：`reset_for_retry` 内部也有 `status=failed` 条件，
        # 但那条 UPDATE 成功与否只能给出"行数为 0"，无法区分
        # "状态不对"和"并发改了"。这里先判断给出可读的错误信息。
        if doc.status != DocumentStatus.FAILED.value:
            raise ConflictError(
                f"只有失败的文档可以重试（当前状态：{doc.status}）",
                details={"status": str(doc.status)},
            )

        reset = await self._docs.reset_for_retry(tenant_id=ctx.tenant_id, doc_id=doc_id)
        if reset is None:
            # 并发：另一个请求抢先重置了状态
            raise ConflictError("文档状态已变化，请刷新后重试")

        version = await self._docs.get_current_version(doc_id=doc_id)
        if version is None:
            # 没有版本的失败文档不可重试
            # （重建版本的语义应由"重新上传"承担）
            raise NotFoundError("文档版本不存在")

        task, _created = await self._tasks.enqueue(
            tenant_id=ctx.tenant_id,
            task_type=TaskType.PARSE_DOCUMENT.value,
            # 幂等键含 `retry:{version}` 而不是 `parse:{version}`：
            # 原解析任务的 key 已经存在（`parse:{version}`），复用它会让
            # `enqueue` 返回那个已完成的旧任务，新任务永远不会被创建。
            idempotency_key=f"retry:{version.id}:{uuid.uuid4().hex[:8]}",
            payload={
                "document_id": str(doc_id),
                "document_version_id": str(version.id),
            },
            document_id=doc_id,
            document_version_id=version.id,
            knowledge_base_id=doc.knowledge_base_id,
        )

        await AuditService(self._session).record(
            ctx=ctx,
            action=AuditAction.DOC_UPLOAD,
            target_type=AuditTargetType.DOCUMENT,
            target_id=doc_id,
            target_name=doc.title,
            details={"retry": True, "task_id": str(task.id)},
        )
        log.info(
            "document_retry_enqueued",
            doc_id=str(doc_id),
            task_id=str(task.id),
            retried_by=str(ctx.user_id),
        )

        return doc_id, task.id

    # ------------------------------------------------------------ 下载

    async def create_download(
        self,
        *,
        ctx: PermissionContext,
        doc_id: uuid.UUID,
        storage: Any,
        version_no: int | None = None,
        expires_in_s: int = 900,
    ) -> DocumentDownloadResponse:
        """签发下载地址（或返回代理地址）。

        ## 为什么权限校验必须在这里，而不是在路由

        `StorageBackend.presigned_url()` 一旦签发，**有效期内任何人拿到 URL
        都能下载**，绕开所有 API 鉴权。所以"能不能下载"必须在签发前
        就判定完毕：

        1. 文档可读（ACL + 密级，用 `document_access_filter`）；
        2. 版本属于该文档；
        3. 只有 `blob` 存在且状态 complete 时才能签。

        任何一步在路由里做，都会留下"别的调用方忘了校验"的缺口。

        ## 为什么下载要审计

        下载是"把文件带走"的动作，比在线阅读更敏感（可以脱离系统传播）。
        `AuditAction.DOC_DOWNLOAD` 早在契约里定义，但此前没有任何调用点。
        """
        doc = await self._require_visible(ctx=ctx, doc_id=doc_id)

        version = (
            await self._docs.get_current_version(doc_id=doc.id)
            if version_no is None
            else await self._find_version_by_no(doc_id=doc.id, version_no=version_no)
        )
        if version is None:
            raise NotFoundError("文档版本不存在")

        if version.status not in (
            DocumentVersionStatus.READY.value,
            DocumentVersionStatus.PROCESSING.value,
        ):
            # superseded / failed 的版本仍可下载（历史版本要能取回），
            # 但这里显式排除"不存在于任何可用状态"的脏数据
            log.warning(
                "download_unexpected_version_status",
                doc_id=str(doc_id),
                version_id=str(version.id),
                status=str(version.status),
            )

        from rag_db.repositories import BlobRepository

        blob = await BlobRepository(self._session).get(
            tenant_id=ctx.tenant_id, blob_id=version.blob_id
        )
        if blob is None:
            raise NotFoundError("文件不存在")
        if blob.status != BlobStatus.COMPLETE.value:
            raise ConflictError(
                f"文件当前不可下载（状态：{blob.status}）",
                details={"blob_status": str(blob.status)},
            )

        # ---- 签发地址 ----
        #
        # S3 走临时直链（客户端直连存储，不占 API 带宽）；
        # local 后端没有签名概念，返回本服务的代理地址。
        # 前端不需要区分：两种情况都是"GET 这个 URL 就能拿到文件"。
        capabilities = storage.capabilities
        if capabilities.supports_presigned_url:
            url = await storage.presigned_url(blob.storage_key, expires_in_s=expires_in_s)
        else:
            url = f"/api/v1/documents/{doc.id}/content?version_no={version.version_no}"
            expires_in_s = 0  # 代理地址不过期（每次请求都会重新鉴权）

        await AuditService(self._session).record(
            ctx=ctx,
            action=AuditAction.DOC_DOWNLOAD,
            target_type=AuditTargetType.DOCUMENT,
            target_id=doc.id,
            target_name=doc.title,
            details={
                "version_no": version.version_no,
                "size_bytes": blob.size_bytes,
                "via_presigned": capabilities.supports_presigned_url,
            },
        )
        log.info(
            "document_download_url_issued",
            doc_id=str(doc_id),
            version_no=version.version_no,
            presigned=capabilities.supports_presigned_url,
            user_id=str(ctx.user_id),
        )

        return DocumentDownloadResponse(
            document_id=doc.id,
            version_no=version.version_no,
            filename=doc.original_filename,
            content_type=blob.content_type,
            size_bytes=blob.size_bytes,
            download_url=url,
            # 代理地址的"有效期"没有意义，用 0 表达"不过期"会违反契约的 ge=1，
            # 所以给一个足够大的值并让前端忽略它（URL 每次都会重新鉴权）
            expires_in_s=max(expires_in_s, 1),
        )

    async def _find_version_by_no(
        self, *, doc_id: uuid.UUID, version_no: int
    ) -> DocumentVersion | None:
        """按版本号取版本（下载指定历史版本时用）。"""
        versions = await self._docs.list_versions(document_id=doc_id)
        for version in versions:
            if version.version_no == version_no:
                return version
        return None

    # ------------------------------------------------------------ 回滚

    async def rollback(
        self, *, ctx: PermissionContext, doc_id: uuid.UUID, version_id: uuid.UUID
    ) -> tuple[Document, _DocumentVersionList]:
        """回滚到指定版本（B5 的"版本查看与回滚"）。

        ## 为什么回滚是纯元数据操作

        `DocumentVersion` 在创建时就快照了切分参数、解析器、blob 指针，
        chunk 也按版本存着（`is_current` 标记当前版本）。
        所以回滚 = 改两个标记：

            - 目标版本的 chunk  → is_current=true
            - 其他版本的 chunk  → is_current=false
            - documents.current_version_id / current_version_no 指向目标

        **不重新解析、不重新向量化** —— 这正是双版本设计的价值
        （ADR 0002）。代价是旧版本的向量必须保留到确认不再需要回滚。

        ## 为什么要校验目标版本是 ready

        回滚到 processing / failed 版本等于把文档变成"不可检索"，
        而用户以为"回滚成功"。显式拒绝比静默接受安全。
        """
        from rag_contracts.enums import DocumentVersionStatus

        doc = await self._require_visible(ctx=ctx, doc_id=doc_id)

        target = await self._docs.get_version(doc_id=doc_id, version_id=version_id)
        if target is None:
            raise NotFoundError("文档版本不存在")
        if target.is_current:
            raise ConflictError("该版本已是当前版本")

        if str(target.status) != DocumentVersionStatus.READY.value:
            raise ConflictError(
                f"只能回滚到 ready 版本（目标版本状态：{target.status}）",
                details={"version_status": str(target.status)},
            )

        current = await self._docs.get_current_version(doc_id=doc_id)

        # 1) 旧版本的 chunk 置为非当前（保留数据供再次回滚）
        if current is not None:
            await self._chunks.mark_version_chunks_stale(document_version_id=current.id)
            # 旧版本状态置 superseded（保留，供追溯）
            await self._docs.supersede_older_versions(
                document_id=doc_id, keep_version_no=target.version_no
            )

        # 2) 目标版本的 chunk 置回当前
        await self._chunks.mark_version_chunks_current(document_version_id=target.id)

        # 3) 文档指针指向目标版本
        await self._docs.mark_version_current(
            document_id=doc_id, version_id=target.id, version_no=target.version_no
        )
        # mark_version_current 只改 version 表的 is_current 与文档指针，
        # 文档业务状态需要同步回 READY（否则列表显示的 status 还是旧值）
        await self._docs.update_status(
            tenant_id=ctx.tenant_id,
            doc_id=doc_id,
            new_status=DocumentStatus.READY,
            expected_version=doc.version,
            chunk_count=target.chunk_count,
            page_count=target.page_count,
            error_code=None,
            error_message=None,
        )

        await AuditService(self._session).record(
            ctx=ctx,
            action=AuditAction.DOC_VERSION_ROLLBACK,
            target_type=AuditTargetType.DOCUMENT_VERSION,
            target_id=target.id,
            target_name=doc.title,
            details={
                "document_id": str(doc_id),
                "from_version_no": current.version_no if current else None,
                "to_version_no": target.version_no,
            },
        )
        log.info(
            "document_version_rolled_back",
            doc_id=str(doc_id),
            to_version_no=target.version_no,
            from_version_no=current.version_no if current else None,
        )

        versions = await self._docs.list_versions(document_id=doc_id)
        refreshed = await self._docs.get_visible(
            tenant_id=ctx.tenant_id,
            doc_id=doc_id,
            access_filter=document_access_filter(ctx),
        )
        return refreshed or doc, list(versions)

    # ------------------------------------------------------------ 内部

    async def _require_visible(self, *, ctx: PermissionContext, doc_id: uuid.UUID) -> Document:
        """取当前用户可读的文档；不可读与不存在一样返回 404。"""
        doc = await self._docs.get_visible(
            tenant_id=ctx.tenant_id,
            doc_id=doc_id,
            access_filter=document_access_filter(ctx),
        )
        if doc is None:
            raise NotFoundError("文档不存在")
        return doc


def _parse_status(value: str | None) -> str | None:
    """校验状态过滤参数。

    **未知值返回 422 而不是静默忽略**：静默忽略会让用户以为过滤生效了，
    实际拿到全量——在"我只想看失败的"这个场景下会导致误判
    "没有失败的文档"。
    """
    if value is None:
        return None
    try:
        DocumentStatus(value)
    except ValueError as exc:
        from rag_contracts.errors import FieldError

        raise ValidationFailedError(
            fields=[
                FieldError(
                    location=["query", "status"],
                    message=f"未知的文档状态: {value}",
                    type="value_error",
                )
            ]
        ) from exc
    return value


__all__ = ["DocumentService"]
