"""会话与消息服务：列表 / 历史 / 删除 / 反馈。

## 为什么与 `ChatService` 分开

`ChatService` 负责「一次问答」的完整编排（改写、检索、生成、落库），
构造它必须注入 LLM / Embedding provider（进程级单例、可能加载 2GB 模型）。
会话列表、历史、删除、点赞这些操作**不需要任何模型能力**，放进同一个
服务会让「打开会话列表」也去触碰 LLM 工厂——配置缺失时列表页直接 503。

本模块只依赖数据库，返回领域对象；响应实体的映射在
`rag_api.schemas.conversations`。
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums.conversations import (
    FeedbackRating,
    FeedbackReason,
    MessageRole,
)
from rag_core.errors import NotFoundError, ValidationFailedError
from rag_db.models.conversation import Conversation, Message
from rag_db.repositories import ConversationRepository, MessageRepository


class ConversationService:
    """会话与消息的读取与轻量写操作（不涉及模型调用）。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)

    async def list_for_user(
        self,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int,
        offset: int,
        include_total: bool,
    ) -> tuple[list[Conversation], int | None]:
        """当前用户的会话列表（按最后消息时间倒序）。"""
        rows, total = await self._conversations.list_for_user(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
            include_total=include_total,
        )
        return list(rows), total

    async def list_messages(
        self,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        limit: int,
        offset: int,
        include_total: bool,
    ) -> tuple[list[Message], int | None]:
        """会话历史。先校验归属（跨用户 → 404，不泄漏存在性）。"""
        await self.require_owned(
            tenant_id=tenant_id, user_id=user_id, conversation_id=conversation_id
        )
        rows, total = await self._messages.page(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
            include_total=include_total,
        )
        return list(rows), total

    async def delete(
        self, *, tenant_id: uuid.UUID, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> None:
        """软删除会话。

        软删而不是物理删：审计链需要"这个会话存在过"，消息里的 feedback
        是评估集来源。跨用户/跨租户访问返回 404（不泄漏存在性）。
        """
        conversation = await self._conversations.get(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        if conversation is None:
            raise NotFoundError("会话不存在")
        await self._conversations.soft_delete(conversation_id=conversation_id)

    async def submit_feedback(
        self,
        *,
        tenant_id: uuid.UUID,
        message_id: uuid.UUID,
        rating: FeedbackRating,
        reason: FeedbackReason | None,
        comment: str | None,
    ) -> None:
        """记录点赞/点踩。

        只允许对助手消息反馈：给用户自己的提问点赞没有意义。
        点踩必须带原因（自由文本无法聚合统计，固定枚举才能回答
        "哪类问题最常被踩"）。
        """
        message = await self._messages.get(tenant_id=tenant_id, message_id=message_id)
        if message is None:
            raise NotFoundError("消息不存在")

        if message.role != MessageRole.ASSISTANT:
            raise ValidationFailedError("只能对助手回答提交反馈")

        if rating == FeedbackRating.DISLIKE and reason is None:
            raise ValidationFailedError("点踩时请选择原因")

        await self._messages.set_feedback(
            tenant_id=tenant_id,
            message_id=message_id,
            rating=rating,
            reason=reason,
            comment=comment,
        )

    async def require_owned(
        self, *, tenant_id: uuid.UUID, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation:
        """取会话并要求属于当前用户。"""
        conversation = await self._conversations.get(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        if conversation is None:
            raise NotFoundError("会话不存在")
        return conversation


__all__ = ["ConversationService"]
