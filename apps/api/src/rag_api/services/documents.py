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

from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.permissions.acl import PermissionContext, document_access_filter
from rag_api.services.audit import AuditService
from rag_api.services.ingestion import IngestionResult, IngestionService
from rag_contracts.enums import (
    AuditAction,
    AuditTargetType,
    ConfidentialityLevel,
    DocumentStatus,
    TaskType,
)
from rag_core.errors import ConflictError, NotFoundError, ValidationFailedError
from rag_core.logging import get_logger
from rag_db.models.document import Document, DocumentVersion
from rag_db.repositories import (
    BlobRepository,
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
