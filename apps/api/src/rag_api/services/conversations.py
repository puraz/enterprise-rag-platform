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
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums.conversations import (
    FeedbackRating,
    FeedbackReason,
    MessageRole,
)
from rag_core.errors import NotFoundError, ValidationFailedError
from rag_core.logging import get_logger
from rag_core.pii import redact_text
from rag_db.models.conversation import Conversation, Message
from rag_db.repositories import ConversationRepository, MessageRepository
from rag_db.repositories.audit import ModelCallRepository

log = get_logger(__name__)


class ConversationService:
    """会话与消息的读取与轻量写操作（不涉及模型调用）。"""

    def __init__(self, session: AsyncSession, *, tracer: Any = None) -> None:
        self._session = session
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)
        # 追踪器（可选）：反馈要回写到对应 trace 的 score 上。
        # None（未开启追踪 / 单测）时只落库，不报错。
        self._tracer = tracer

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

        # 落库之后再写 score：库里是**事实**，Langfuse 是**旁路**。
        # 顺序反了会出现"Langfuse 记了差评但数据库没有"的不可信状态。
        await self._write_feedback_score(
            tenant_id=tenant_id,
            message_id=message_id,
            rating=rating,
            reason=reason,
            comment=comment,
        )

    async def _write_feedback_score(
        self,
        *,
        tenant_id: uuid.UUID,
        message_id: uuid.UUID,
        rating: FeedbackRating,
        reason: FeedbackReason | None,
        comment: str | None,
    ) -> None:
        """把用户反馈写回对应 trace 的 score（评估数据闭环）。

        ## 三条边界

        1. **失败不影响入库**：反馈已经落到 `messages.feedback_*`，
           score 只是让评估集能直接按 trace 追溯；
        2. **找不到 trace 就跳过**：追踪未开 / 采样丢弃 / 旧消息，
           都是正常情况，记 info 而不是 warning（否则日志噪音）；
        3. **评论文本必须脱敏**：Langfuse 是另一个系统，出口规则与日志一致。
        """
        if self._tracer is None:
            return
        try:
            trace_id = await ModelCallRepository(self._session).find_trace_id(
                tenant_id=tenant_id, message_id=message_id
            )
        except Exception as exc:
            log.warning(
                "feedback_trace_lookup_failed",
                message_id=str(message_id),
                error_type=type(exc).__name__,
            )
            return

        if not trace_id:
            log.info(
                "feedback_trace_missing",
                message_id=str(message_id),
                note="该消息没有关联 trace（未开启追踪或已被采样丢弃）",
            )
            return

        detail = " ".join(part for part in (reason.value if reason else "", comment or "") if part)
        safe_comment = redact_text(detail)[:200] if detail else None
        try:
            await self._tracer.score_trace(
                trace_id=trace_id,
                name="user_feedback",
                # 1.0 / 0.0 而不是 1/2：Langfuse 的数值 score 支持负数（例如 -1），
                # 用 0/1 便于直接算"好评率 = mean"。
                value=1.0 if rating == FeedbackRating.LIKE else 0.0,
                comment=safe_comment,
            )
        except Exception as exc:
            log.warning(
                "feedback_score_failed",
                message_id=str(message_id),
                error_type=type(exc).__name__,
                note="反馈已入库，score 写入失败（旁路）",
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
