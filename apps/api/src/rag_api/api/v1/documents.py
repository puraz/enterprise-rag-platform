"""文档路由。

## 端点设计

| 方法 | 路径 | 状态码 | 说明 |
|---|---|---|---|
| POST | `/documents` | **202** | blob → 文档 + 版本 + 入队解析 |
| GET | `/documents` | 200 | 列表（可按 `kb_id` / `status` 过滤） |
| GET | `/documents/{id}` | 200 | 详情 + 版本历史 |
| DELETE | `/documents/{id}` | **204** | 软删除 |
| POST | `/documents/{id}/retry` | **202** | 仅 failed 可重试 |

## 上传与建文档为什么分成两步

见 `CreateDocumentRequest` 的 docstring：秒传与首次上传在 complete
之后都只拿到 `blob_id`，走 `POST /documents` 这一条路径。

      POST /uploads → PUT parts × N → POST /uploads/{id}/complete
                                                    │
                                                    ▼ blob_id
                                             POST /documents  ← 秒传也走这里

一份代码覆盖两条路径，不会出现"秒传建的文档少了某个初始化步骤"。

## 为什么建文档是 202 而不是 201

201 意味着"资源已完整创建"。这里只写了文档元数据，chunk 与向量
都还在异步管线里。用 202 + `task_id` 让前端明确知道"要轮询"。

若返回 201，前端会以为文档立刻可检索，用户提问时检索不到内容
——这种行为差异很难归因。

## 删除为什么不能级联清理

文档删除只是标记（`deleted_at`），因为：
- chunk 与向量清理要走异步任务（可能几万行）；
- 立即物理删会让"删了但索引还在"的窗口里检索返回空结果，
  用户看到的是"问不出来但文档还在列表里"。

blob 引用计数减 1，减到 0 后由清理任务回收对象存储空间。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import Principal, get_current_principal, get_db
from rag_api.deps.pagination import PageQuery, page_query
from rag_api.openapi import (
    ASYNC_ENDPOINT,
    DELETE_ENDPOINT,
    TENANT_ENDPOINT,
)
from rag_api.permissions import PermissionContext, get_permission_context
from rag_api.permissions.acl import document_access_filter
from rag_api.services.ingestion import IngestionService
from rag_contracts.api.knowledge import (
    CreateDocumentRequest,
    CreateDocumentResponse,
    DocumentDetail,
    DocumentItem,
    DocumentVersionItem,
)
from rag_contracts.api.pagination import Page, PageMeta
from rag_contracts.enums import AuditAction, AuditTargetType, DocumentStatus, TaskType
from rag_core.errors import ConflictError, NotFoundError
from rag_core.logging import get_logger
from rag_db.models.document import Document
from rag_db.repositories import (
    BlobRepository,
    DocumentRepository,
    KnowledgeBaseRepository,
    TaskRepository,
)

log = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=CreateDocumentResponse,
    status_code=202,
    summary="登记文档并触发解析",
    description=(
        "把已上传完成的 blob 登记为文档，并异步入队解析。\n\n"
        "**秒传与首次上传走同一条路径**：两者在完成后都只拿到 `blob_id`。\n\n"
        "响应 `202` + `task_id`：文档元数据已建，内容处理是异步的。"
        "前端应轮询 `GET /documents/{id}` 直到 `status=ready`。\n\n"
        "**文档密级不得低于所属库密级**（省略则继承库级）。"
    ),
    responses=ASYNC_ENDPOINT,
)
async def create_document(
    payload: CreateDocumentRequest,
    principal: Principal = Depends(get_current_principal),
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> CreateDocumentResponse:
    ctx.require_kb_write()

    result = await IngestionService(session).create_document(
        ctx=ctx,
        blob_id=payload.blob_id,
        knowledge_base_id=payload.knowledge_base_id,
        original_filename=payload.original_filename,
        title=payload.title,
        confidentiality=payload.confidentiality,
    )

    return CreateDocumentResponse(
        document_id=result.document.id,
        task_id=result.task_id,
        status=DocumentStatus(str(result.document.status)),
        deduplicated_blob=result.deduplicated_blob,
    )


@router.get(
    "",
    response_model=Page[DocumentItem],
    summary="文档列表",
    description=(
        "按 ACL 过滤后的文档列表。\n\n"
        "过滤条件（在 SQL 的 WHERE 里，不是查出来再筛）：\n"
        "1. 未删除；\n"
        "2. 文档密级 ≤ 用户密级上限；\n"
        "3. 所属知识库对该用户可见。"
    ),
    responses=TENANT_ENDPOINT,
)
async def list_documents(
    principal: Principal = Depends(get_current_principal),
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
    page: PageQuery = Depends(page_query),
    kb_id: uuid.UUID | None = Query(default=None, description="按知识库过滤"),
    status: str | None = Query(
        default=None,
        description="按处理状态过滤（pending/parsing/.../ready/failed）",
    ),
) -> Page[DocumentItem]:
    # 状态过滤参数校验：未知值返回 422 而不是静默忽略。
    # 静默忽略会让用户以为过滤生效了，实际拿到全量——在"我只想看失败的"
    # 这个场景下会导致误判"没有失败的文档"。
    if status is not None:
        try:
            DocumentStatus(status)
        except ValueError as exc:
            from rag_contracts.errors import FieldError
            from rag_core.errors import ValidationFailedError

            raise ValidationFailedError(
                fields=[
                    FieldError(
                        location=["query", "status"],
                        message=f"未知的文档状态: {status}",
                        type="value_error",
                    )
                ]
            ) from exc

    repo = DocumentRepository(session)
    rows, total = await repo.list_by_kb_filtered(
        tenant_id=ctx.tenant_id,
        kb_id=kb_id,
        limit=page.limit,
        offset=page.offset,
        # **ACL 必须在 SQL 里**：文档比知识库多得多，"查出来再过滤"
        # 既浪费 I/O 又会让越权内容进应用内存（日志/异常都可能带出标题）。
        access_filter=document_access_filter(ctx),
        status=status,
        include_total=page.include_total,
    )
    items = [_to_item(doc) for doc in rows]
    return Page(
        items=items,
        meta=PageMeta(
            limit=page.limit,
            offset=page.offset,
            total=total,
            has_more=len(items) == page.limit,
        ),
    )


@router.get(
    "/{doc_id}",
    response_model=DocumentDetail,
    summary="文档详情（含版本历史）",
    description="不可读或不存在都返回 404。版本历史按版本号倒序。",
    responses=TENANT_ENDPOINT,
)
async def get_document(
    doc_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> DocumentDetail:
    repo = DocumentRepository(session)
    doc = await repo.get_visible(
        tenant_id=ctx.tenant_id,
        doc_id=doc_id,
        access_filter=document_access_filter(ctx),
    )
    if doc is None:
        raise NotFoundError("文档不存在")

    versions = await repo.list_versions(document_id=doc.id)
    return _to_detail(doc, versions)


@router.delete(
    "/{doc_id}",
    status_code=204,
    summary="删除文档",
    description=(
        "软删除（保留元数据与审计追溯），blob 引用计数减 1。\n\n"
        "chunk 与向量的物理清理由后台任务处理——立即删会留下"
        "\"索引还在但文档已删\"的窗口。"
    ),
    responses=DELETE_ENDPOINT,
)
async def delete_document(
    doc_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> None:
    ctx.require_kb_write()

    repo = DocumentRepository(session)
    doc = await repo.get_visible(
        tenant_id=ctx.tenant_id,
        doc_id=doc_id,
        access_filter=document_access_filter(ctx),
    )
    if doc is None:
        raise NotFoundError("文档不存在")

    deleted = await repo.soft_delete(tenant_id=ctx.tenant_id, doc_id=doc_id)
    if not deleted:
        # 并发删除：两个请求同时进来，一个成功一个拿到 0 行。
        # 对用户来说资源已不存在，返回 404 比 500 更准确。
        raise NotFoundError("文档不存在")

    # blob 引用计数减 1：减到 0 后清理任务才能回收对象存储。
    # 不减会导致存储空间永远不释放（"删了文档但磁盘不降"）。
    version = await repo.get_current_version(doc_id=doc_id)
    if version is not None:
        blobs = BlobRepository(session)
        try:
            await blobs.increment_ref(blob_id=version.blob_id, delta=-1)
        except ValueError:
            # 计数减到负数说明数据已经不一致（可能重复删除）。
            # 不回滚删除（用户要的结果已经达成），但要留下明确日志
            # 供对账任务发现。
            log.error(
                "blob_ref_count_underflow",
                doc_id=str(doc_id),
                blob_id=str(version.blob_id),
            )

    # 库的文档计数减 1。同样只在成功删除时执行。
    await KnowledgeBaseRepository(session).increment_document_count(
        kb_id=doc.knowledge_base_id, delta=-1
    )

    from rag_api.services.audit import AuditService

    await AuditService(session).record(
        ctx=ctx,
        action=AuditAction.DOC_DELETE,
        target_type=AuditTargetType.DOCUMENT,
        target_id=doc.id,
        target_name=doc.title,
        details={"knowledge_base_id": str(doc.knowledge_base_id)},
    )
    log.info("document_deleted", doc_id=str(doc_id), deleted_by=str(ctx.user_id))


@router.post(
    "/{doc_id}/retry",
    response_model=CreateDocumentResponse,
    status_code=202,
    summary="重试失败的文档",
    description=(
        "仅 `status=failed` 的文档可重试：重置为 `pending` 并重新入队解析。\n\n"
        "**不需要重新上传**：原始 blob 一直保留（`model_document.py` 的设计约束）。"
        "其他状态返回 409。"
    ),
    responses=ASYNC_ENDPOINT,
)
async def retry_document(
    doc_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> CreateDocumentResponse:
    ctx.require_kb_write()

    repo = DocumentRepository(session)
    doc = await repo.get_visible(
        tenant_id=ctx.tenant_id,
        doc_id=doc_id,
        access_filter=document_access_filter(ctx),
    )
    if doc is None:
        raise NotFoundError("文档不存在")

    # 先看状态再改：`reset_for_retry` 内部也有 `status=failed` 条件，
    # 但那条 UPDATE 成功与否只能给出"行数为 0"，无法区分
    # "状态不对"和"并发改了"。这里先判断给出可读的错误信息。
    if doc.status != DocumentStatus.FAILED.value:
        raise ConflictError(
            f"只有失败的文档可以重试（当前状态：{doc.status}）",
            details={"status": str(doc.status)},
        )

    reset = await repo.reset_for_retry(tenant_id=ctx.tenant_id, doc_id=doc_id)
    if reset is None:
        # 并发：另一个请求抢先重置了状态
        raise ConflictError("文档状态已变化，请刷新后重试")

    version = await repo.get_current_version(doc_id=doc_id)
    if version is None:
        # 没有版本的失败文档不可重试（重建版本的语义应由"重新上传"承担）
        raise NotFoundError("文档版本不存在")

    task, _created = await TaskRepository(session).enqueue(
        tenant_id=ctx.tenant_id,
        task_type=TaskType.PARSE_DOCUMENT.value,
        # 幂等键含 `retry:{version}` 而不是 `parse:{version}`：
        # 原解析任务的 key 已经存在（`parse:{version}`），复用它会让
        # `enqueue` 返回那个已完成的旧任务，新任务永远不会被创建。
        # 用不同前缀保证重试真的产生新任务。
        #
        # 代价是同一版本可以重试多次 -> 多个任务。这是期望的：
        # 每次重试都是一次独立执行，worker 的 `retry_count` 会记录。
        idempotency_key=f"retry:{version.id}:{uuid.uuid4().hex[:8]}",
        payload={
            "document_id": str(doc_id),
            "document_version_id": str(version.id),
        },
        document_id=doc_id,
        document_version_id=version.id,
        knowledge_base_id=doc.knowledge_base_id,
    )

    from rag_api.services.audit import AuditService

    await AuditService(session).record(
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

    return CreateDocumentResponse(
        document_id=doc_id,
        task_id=task.id,
        status=DocumentStatus.PENDING,
        deduplicated_blob=False,
    )


# ------------------------------------------------------------------ 映射


def _to_item(doc: Document) -> DocumentItem:
    """ORM → 列表项。

    显式逐字段构造：`from_attributes` 在字段增删时会静默给默认值，
    表现为"前端某个字段永远是 0"。显式写法漏字段 mypy 会报。
    """
    return DocumentItem(
        id=doc.id,
        knowledge_base_id=doc.knowledge_base_id,
        title=doc.title,
        original_filename=doc.original_filename,
        format=doc.format,
        status=doc.status,
        confidentiality=doc.confidentiality,
        chunk_count=doc.chunk_count,
        page_count=doc.page_count,
        size_bytes=doc.size_bytes,
        current_version_no=doc.current_version_no,
        error_code=doc.error_code,
        error_message=doc.error_message,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def _to_detail(doc: Document, versions: object) -> DocumentDetail:
    """ORM → 详情（含版本列表）。"""
    base = _to_item(doc)
    version_items = [
        DocumentVersionItem(
            id=version.id,
            version_no=version.version_no,
            status=version.status,
            is_current=version.is_current,
            chunk_count=version.chunk_count,
            page_count=version.page_count,
            parser_name=version.parser_name,
            created_at=version.created_at,
        )
        for version in versions  # type: ignore[attr-defined]
    ]
    return DocumentDetail(**base.model_dump(), versions=version_items)
