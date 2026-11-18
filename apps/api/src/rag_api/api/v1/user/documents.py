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

blob 引用计数与库文档计数的维护在 `DocumentService.delete` 里，
与软删一起完成（详见该模块 docstring）。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from rag_adapters.storages.base import StorageBackend
from rag_api.deps import get_db, get_storage
from rag_api.deps.pagination import PageQuery, page_query
from rag_api.openapi import (
    ASYNC_ENDPOINT,
    DELETE_ENDPOINT,
    TENANT_ENDPOINT,
    TENANT_WRITE_ENDPOINT,
)
from rag_api.permissions import PermissionContext, get_permission_context
from rag_api.schemas.common import Page
from rag_api.schemas.documents import (
    CreateDocumentRequest,
    CreateDocumentResponse,
    DocumentDetail,
    DocumentItem,
    to_create_response,
    to_detail,
    to_page,
    to_retry_response,
)
from rag_api.services.documents import DocumentService
from rag_contracts.api.knowledge import DocumentDownloadResponse

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
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> CreateDocumentResponse:
    ctx.require_kb_write()
    result = await DocumentService(session).create(
        ctx=ctx,
        blob_id=payload.blob_id,
        knowledge_base_id=payload.knowledge_base_id,
        original_filename=payload.original_filename,
        title=payload.title,
        confidentiality=payload.confidentiality,
    )
    return to_create_response(result)


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
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
    page: PageQuery = Depends(page_query),
    kb_id: uuid.UUID | None = Query(default=None, description="按知识库过滤"),
    status: str | None = Query(
        default=None,
        description="按处理状态过滤（pending/parsing/.../ready/failed）",
    ),
) -> Page[DocumentItem]:
    rows, total = await DocumentService(session).list(
        ctx=ctx,
        limit=page.limit,
        offset=page.offset,
        include_total=page.include_total,
        kb_id=kb_id,
        status=status,
    )
    return to_page(rows, total=total, limit=page.limit, offset=page.offset)


@router.get(
    "/{doc_id}",
    response_model=DocumentDetail,
    summary="文档详情（含版本历史）",
    description="不可读或不存在都返回 404。版本历史按版本号倒序。",
    responses=TENANT_ENDPOINT,
)
async def get_document(
    doc_id: uuid.UUID,
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> DocumentDetail:
    doc, versions = await DocumentService(session).get(ctx=ctx, doc_id=doc_id)
    return to_detail(doc, versions)


@router.delete(
    "/{doc_id}",
    status_code=204,
    summary="删除文档",
    description=(
        "软删除（保留元数据与审计追溯），blob 引用计数减 1。\n\n"
        "chunk 与向量的物理清理由后台任务处理——立即删会留下"
        '"索引还在但文档已删"的窗口。'
    ),
    responses=DELETE_ENDPOINT,
)
async def delete_document(
    doc_id: uuid.UUID,
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> None:
    ctx.require_kb_write()
    await DocumentService(session).delete(ctx=ctx, doc_id=doc_id)


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
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> CreateDocumentResponse:
    ctx.require_kb_write()
    document_id, task_id = await DocumentService(session).retry(ctx=ctx, doc_id=doc_id)
    return to_retry_response(document_id=document_id, task_id=task_id)


@router.get(
    "/{doc_id}/download",
    response_model=DocumentDownloadResponse,
    summary="获取下载地址",
    description=(
        "返回临时下载地址（默认 15 分钟）。\n\n"
        "**S3 后端返回签名直链**（客户端直连存储）；**local 后端返回本服务代理地址**。"
        "前端两种情况都是 GET 这个 URL，不需要区分。\n\n"
        "**签名 URL 一旦发出，有效期内任何人都能下载**：所以权限与密级校验"
        "在签发前完成（文档不可读返回 404，不泄漏存在性）。\n\n"
        "下载动作会写审计（`document.download`）——把文件带走比在线阅读更敏感。\n\n"
        "`version_no` 可指定历史版本；省略则下载当前版本。"
    ),
    responses=TENANT_ENDPOINT,
)
async def download_document(
    doc_id: uuid.UUID,
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
    version_no: int | None = Query(
        default=None, ge=1, description="指定版本号；省略则下载当前版本"
    ),
) -> DocumentDownloadResponse:
    return await DocumentService(session).create_download(
        ctx=ctx, doc_id=doc_id, storage=storage, version_no=version_no
    )


@router.post(
    "/{doc_id}/versions/{version_id}/activate",
    response_model=DocumentDetail,
    summary="回滚到指定版本",
    description=(
        "把指定版本设为当前生效版本。\n\n"
        "**纯元数据操作，秒级完成**：改版本指针与 chunk 的 `is_current` 标记，"
        "不重新解析、不重新向量化——这正是版本快照设计的价值。\n\n"
        "约束：\n"
        "- 目标版本必须属于该文档且状态为 `ready`（回滚到失败版本等于把文档变不可检索）；\n"
        "- 已是当前版本时返回 409；\n"
        "- 需要知识库写权限。\n\n"
        "响应返回回滚后的文档详情（含版本列表），前端可直接刷新界面。"
    ),
    responses=TENANT_WRITE_ENDPOINT,
)
async def rollback_document_version(  # pyright: ignore[reportUnusedFunction]
    doc_id: uuid.UUID,
    version_id: uuid.UUID,
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> DocumentDetail:
    ctx.require_kb_write()
    doc, versions = await DocumentService(session).rollback(
        ctx=ctx, doc_id=doc_id, version_id=version_id
    )
    return to_detail(doc, versions)
