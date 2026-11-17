"""知识库请求/响应实体与映射。

契约形状来自 `rag_contracts.api.knowledge`；本模块只做 re-export 与
ORM → 响应实体的映射。
"""

from __future__ import annotations

from rag_contracts.api.knowledge import (
    CreateKnowledgeBaseRequest,
    KnowledgeBaseItem,
    UpdateKnowledgeBaseRequest,
)
from rag_contracts.api.pagination import Page, PageMeta
from rag_db.models.knowledge import KnowledgeBase


def to_item(kb: KnowledgeBase) -> KnowledgeBaseItem:
    """ORM → 知识库响应实体。

    显式逐字段构造而不是 `from_attributes` 自动映射：
    后者在字段名不同时（`default_chunking_strategy` → `chunking_strategy`）
    会静默用默认值，表现为"前端拿到的 chunk_size 永远是 0"。
    显式写出来，漏字段 mypy 会报。
    """
    return KnowledgeBaseItem(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        status=kb.status,
        visibility=kb.visibility,
        confidentiality=kb.confidentiality,
        document_count=kb.document_count,
        chunking_strategy=kb.default_chunking_strategy,
        chunk_size=kb.default_chunk_size,
        chunk_overlap=kb.default_chunk_overlap,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


def to_page(
    rows: list[KnowledgeBase], *, total: int | None, limit: int, offset: int
) -> Page[KnowledgeBaseItem]:
    """一页知识库 → 分页响应实体。

    `has_more` 用"本页是否取满"判断：不满说明后面没有了。
    这比再查一次 COUNT 便宜，也是 offset 分页的标准做法
    （代价是最后一页恰好取满时会多一次空请求）。
    """
    items = [to_item(kb) for kb in rows]
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
    "CreateKnowledgeBaseRequest",
    "KnowledgeBaseItem",
    "UpdateKnowledgeBaseRequest",
    "to_item",
    "to_page",
]
