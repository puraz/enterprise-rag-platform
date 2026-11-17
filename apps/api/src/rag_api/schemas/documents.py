"""文档请求/响应实体与映射。

契约形状来自 `rag_contracts.api.knowledge`；本模块只做 re-export 与
domain → 响应实体的映射。
"""

from __future__ import annotations

import uuid

from rag_api.services.ingestion import IngestionResult
from rag_contracts.api.knowledge import (
    CreateDocumentRequest,
    CreateDocumentResponse,
    DocumentDetail,
    DocumentItem,
    DocumentVersionItem,
)
from rag_contracts.api.pagination import Page, PageMeta
from rag_contracts.enums import DocumentStatus
from rag_db.models.document import Document, DocumentVersion


def to_item(doc: Document) -> DocumentItem:
    """ORM → 文档列表项。

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


def to_detail(doc: Document, versions: list[DocumentVersion]) -> DocumentDetail:
    """ORM → 文档详情（含版本列表）。"""
    base = to_item(doc)
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
        for version in versions
    ]
    return DocumentDetail(**base.model_dump(), versions=version_items)


def to_create_response(result: IngestionResult) -> CreateDocumentResponse:
    """入库结果 → 202 响应实体。"""
    return CreateDocumentResponse(
        document_id=result.document.id,
        task_id=result.task_id,
        status=DocumentStatus(str(result.document.status)),
        deduplicated_blob=result.deduplicated_blob,
    )


def to_retry_response(*, document_id: uuid.UUID, task_id: uuid.UUID) -> CreateDocumentResponse:
    """重试任务 → 202 响应实体（重试后状态固定为 pending）。"""
    return CreateDocumentResponse(
        document_id=document_id,
        task_id=task_id,
        status=DocumentStatus.PENDING,
        deduplicated_blob=False,
    )


def to_page(
    rows: list[Document], *, total: int | None, limit: int, offset: int
) -> Page[DocumentItem]:
    """一页文档 → 分页响应实体。"""
    items = [to_item(doc) for doc in rows]
    return Page(
        items=items,
        meta=PageMeta(
            limit=limit,
            offset=offset,
            total=total,
            has_more=len(items) == limit,
        ),
    )


__all__ = [
    "CreateDocumentRequest",
    "CreateDocumentResponse",
    "DocumentDetail",
    "DocumentItem",
    "DocumentVersionItem",
    "to_create_response",
    "to_detail",
    "to_item",
    "to_page",
    "to_retry_response",
]
