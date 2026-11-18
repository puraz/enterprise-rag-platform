"""管理面：文档台账、版本快照、chunk 预览、索引版本。

## 为什么需要这些接口（都是有原料但没出口）

排障时运营要回答的问题清单，每一个都对应一个已有但从未被调用的能力：

| 问题 | 需要的接口 | 已有原料 |
|---|---|---|
| "这个文档卡在 parsing 三天了" | 文档台账（按 status 过滤） | `DocumentRepository` |
| "当时用的什么切分参数？" | 版本快照列表 | `DocumentVersion.chunking_*` |
| "切得好不好？有没有把表格切坏？" | chunk 预览 | `ChunkRepository.list_by_version` |
| "为什么检索不到它？" | 索引版本列表 | `IndexVersionRepository.list_for_kb` |
| "重建/回滚时切到哪一版了？" | 同上 | 同上 |

## chunk 预览为什么单独能力（OPERATE 而不是 READ）

它是管理面**唯一返回文档正文**的接口。管理员可能没有该文档的读权限
（密级不够 / 不在可见库），但排障需要看切分质量。所以：

1. 能力要求提到 `OPERATE`（不是只读能力）；
2. 每次调用写 `admin.chunk_view` 审计（谁在何时看了哪个版本）；
3. 正文按 4000 字符截断（一次 20 条不至于把浏览器拖死）。

这不是"管理权限可以读一切「——而是」读正文这个动作必须可追溯到人"。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps.pagination import PageQuery, page_query
from rag_api.openapi import ADMIN_ENDPOINT
from rag_api.permissions import (
    AdminCapability,
    AdminContext,
    get_admin_context,
    get_admin_db,
)
from rag_api.permissions.admin import require_admin_capability
from rag_api.schemas.admin import (
    to_admin_chunk,
    to_admin_document,
    to_admin_version,
    to_index_version,
)
from rag_api.schemas.common import Page
from rag_contracts.api.admin import (
    AdminChunkItem,
    AdminDocumentItem,
    AdminDocumentVersionItem,
    IndexVersionItem,
)
from rag_contracts.api.pagination import PageMeta
from rag_contracts.enums import (
    AuditAction,
    AuditOutcome,
    AuditTargetType,
    DocumentStatus,
)
from rag_core.errors import NotFoundError, ValidationFailedError
from rag_db.repositories import (
    AuditRepository,
    ChunkRepository,
    DocumentRepository,
    IndexVersionRepository,
    KnowledgeBaseRepository,
)

router = APIRouter(prefix="/documents", tags=["admin"])


@router.get(
    "",
    response_model=Page[AdminDocumentItem],
    summary="文档台账",
    description=(
        "列出**租户内全部**未删除文档（不做 ACL 过滤——管理端看的是"
        '"系统里有什么"，不是"我能读什么"）。\n\n'
        "**不返回正文**：文档内容属于数据面，管理权限管的是系统运行。\n\n"
        "过滤：`kb_id` / `status`。`status=failed` 是排障入口。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.READ))],
    responses=ADMIN_ENDPOINT,
)
async def list_documents(
    ctx: AdminContext = Depends(get_admin_context),
    session: AsyncSession = Depends(get_admin_db),
    page: PageQuery = Depends(page_query),
    kb_id: uuid.UUID | None = Query(default=None, description="按知识库过滤"),
    status: str | None = Query(default=None, description="按处理状态过滤"),
) -> Page[AdminDocumentItem]:
    if status is not None:
        try:
            DocumentStatus(status)
        except ValueError as exc:
            # 静默忽略会让「我只想看失败的」拿到全量，据此误判没有失败文档
            raise ValidationFailedError("未知的文档状态", fields=[]) from exc

    rows, total = await DocumentRepository(session).list_for_admin(
        tenant_id=ctx.data_tenant_id,
        limit=page.limit,
        offset=page.offset,
        kb_id=kb_id,
        status=status,
        include_total=page.include_total,
    )
    items = [to_admin_document(row) for row in rows]
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
    "/{doc_id}/versions",
    response_model=list[AdminDocumentVersionItem],
    summary="文档版本快照（含切分参数）",
    description=(
        "返回版本列表，每个带**当次切分参数快照**（`chunking_strategy` / "
        "`chunk_size` / `chunk_overlap` / `parser_name` / `embedding_model`）。\n\n"
        "为什么是列表而不是分页：一个文档的版本数天然很少（每次重传一个），"
        "且管理端要在一次请求里看到全部版本才能判断「哪一版更好」。\n\n"
        "上限 100 个版本（`list_versions` 的默认 limit）。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.READ))],
    responses=ADMIN_ENDPOINT,
)
async def list_document_versions(
    doc_id: uuid.UUID,
    ctx: AdminContext = Depends(get_admin_context),
    session: AsyncSession = Depends(get_admin_db),
) -> list[AdminDocumentVersionItem]:
    doc = await DocumentRepository(session).get(tenant_id=ctx.data_tenant_id, doc_id=doc_id)
    if doc is None:
        raise NotFoundError("文档不存在")

    versions = await DocumentRepository(session).list_versions(document_id=doc_id)
    return [to_admin_version(version) for version in versions]


@router.get(
    "/{doc_id}/versions/{version_id}/chunks",
    response_model=Page[AdminChunkItem],
    summary="chunk 预览（切分质量检查）",
    description=(
        "返回某版本的 chunk 列表（按原文顺序）。\n\n"
        "**这是管理面唯一返回正文的接口**，每次调用都会写 "
        "`admin.chunk_view` 审计：管理员可能没有该文档的读权限，"
        "读正文必须可追溯到人。\n\n"
        "正文按 4000 字符截断（`is_truncated=true` 时表示只看了一部分），"
        "`char_count` 是原始长度。\n\n"
        "`has_bbox=true` 的 chunk 能在前端框选原文；"
        "`embedding_model` 为空表示尚未向量化（检索不到它）。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.OPERATE))],
    responses=ADMIN_ENDPOINT,
)
async def preview_chunks(
    doc_id: uuid.UUID,
    version_id: uuid.UUID,
    ctx: AdminContext = Depends(get_admin_context),
    session: AsyncSession = Depends(get_admin_db),
    page: PageQuery = Depends(page_query),
) -> Page[AdminChunkItem]:
    docs = DocumentRepository(session)
    doc = await docs.get(tenant_id=ctx.data_tenant_id, doc_id=doc_id)
    if doc is None:
        raise NotFoundError("文档不存在")

    # 必须校验版本属于该文档：只按 version_id 查会让「知道版本 ID」
    # 就能看别的文档的正文（chunk 内容就是文档内容）
    version = await docs.get_version(doc_id=doc_id, version_id=version_id)
    if version is None:
        raise NotFoundError("文档版本不存在")

    rows, total = await ChunkRepository(session).list_by_version_paginated(
        document_version_id=version_id,
        limit=page.limit,
        offset=page.offset,
        include_total=page.include_total,
    )

    await AuditRepository(session).record(
        tenant_id=ctx.data_tenant_id,
        action=AuditAction.ADMIN_CHUNK_VIEW,
        outcome=AuditOutcome.SUCCESS,
        actor_user_id=ctx.user_id,
        target_type=AuditTargetType.DOCUMENT_VERSION,
        target_id=version_id,
        details={
            "document_id": str(doc_id),
            "chunk_count": len(rows),
            "platform_role": ctx.platform_role.value,
        },
    )

    items = [to_admin_chunk(row) for row in rows]
    return Page(
        items=items,
        meta=PageMeta(
            limit=page.limit,
            offset=page.offset,
            total=total,
            has_more=len(items) == page.limit,
        ),
    )


index_router = APIRouter(prefix="/knowledge-bases", tags=["admin"])


@index_router.get(
    "/{kb_id}/index-versions",
    response_model=list[IndexVersionItem],
    summary="索引版本列表",
    description=(
        "列出知识库的索引版本（按创建时间倒序）。\n\n"
        "每一版带 `status`（building/ready/active/retired/failed）与"
        "`chunk_count` / `expected_chunk_count`。\n\n"
        "**看 `chunk_count < expected_chunk_count` 就知道回填还没完成**——"
        "此时切换会静默丢失检索结果，这是最该被拦住的误操作。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.READ))],
    responses=ADMIN_ENDPOINT,
)
async def list_index_versions(
    kb_id: uuid.UUID,
    ctx: AdminContext = Depends(get_admin_context),
    session: AsyncSession = Depends(get_admin_db),
) -> list[IndexVersionItem]:
    kb = await KnowledgeBaseRepository(session).get(tenant_id=ctx.data_tenant_id, kb_id=kb_id)
    if kb is None:
        raise NotFoundError("知识库不存在")

    versions = await IndexVersionRepository(session).list_for_kb(knowledge_base_id=kb_id)
    return [to_index_version(version) for version in versions]


__all__ = ["index_router", "router"]
