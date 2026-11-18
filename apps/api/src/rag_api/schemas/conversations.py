"""会话/消息请求响应实体与映射。

契约形状来自 `rag_contracts.api.chat`；本模块只做 re-export 与
ORM → 响应实体的映射。
"""

from __future__ import annotations

from rag_contracts.api.chat import (
    ChatRequest,
    ConversationItem,
    FeedbackRequest,
    MessageItem,
    UpdateConversationRequest,
)
from rag_contracts.api.citations import Citation
from rag_contracts.api.pagination import Page, PageMeta
from rag_contracts.enums.conversations import FeedbackRating, FeedbackReason
from rag_core.logging import get_logger
from rag_db.models.conversation import Conversation, Message

log = get_logger(__name__)


def to_conversation_item(row: Conversation) -> ConversationItem:
    """ORM → 会话列表项。"""
    return ConversationItem(
        id=row.id,
        title=row.title,
        status=row.status,
        knowledge_base_id=row.knowledge_base_id,
        message_count=row.message_count,
        last_message_at=row.last_message_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
        total_cost_usd=row.total_cost_usd,
    )


def to_message_item(row: Message) -> MessageItem:
    """ORM → 消息项（含引用解析）。"""
    citations: list[Citation] = []
    for raw in row.citations or []:
        try:
            citations.append(Citation.model_validate(raw))
        except Exception:
            # 单条引用损坏不该让整条消息读不出来
            log.warning("citation_parse_failed", message_id=str(row.id))

    return MessageItem(
        id=row.id,
        conversation_id=row.conversation_id,
        sequence=row.sequence,
        role=row.role,
        content=row.content,
        status=row.status,
        refused=row.refused,
        refusal_reason=row.refusal_reason,
        citations=citations,
        created_at=row.created_at,
        rating=FeedbackRating(row.rating) if row.rating else None,
        feedback_reason=FeedbackReason(row.feedback_reason) if row.feedback_reason else None,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cost_usd=row.cost_usd,
        latency_ms=row.latency_ms,
    )


def to_conversation_page(
    rows: list[Conversation], *, total: int | None, limit: int, offset: int
) -> Page[ConversationItem]:
    """一页会话 → 分页响应实体。"""
    items = [to_conversation_item(row) for row in rows]
    return Page(
        items=items,
        meta=PageMeta(
            limit=limit,
            offset=offset,
            total=total,
            has_more=len(items) == limit,
        ),
    )


def to_message_page(
    rows: list[Message], *, total: int | None, limit: int, offset: int
) -> Page[MessageItem]:
    """一页消息 → 分页响应实体。"""
    items = [to_message_item(row) for row in rows]
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
    "ChatRequest",
    "ConversationItem",
    "FeedbackRequest",
    "MessageItem",
    "UpdateConversationRequest",
    "to_conversation_item",
    "to_conversation_page",
    "to_message_item",
    "to_message_page",
]
