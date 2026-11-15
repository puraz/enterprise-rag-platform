"""对话与会话消息模型。

## 为什么两层（conversations + messages）而不是一张表

一张表存所有消息会让"会话级属性"（标题、归属、归档状态、最后活跃时间）
在每一行上重复：
- 改会话标题要 UPDATE 全部消息行（或只改一行，然后靠"取第一条"约定）；
- 会话列表查询要 `GROUP BY conversation_id` 扫全表；
- 归档会话要批量改 N 行。

两层后会话列表是一次索引扫描（`ix_conversations_tenant_updated`），
与消息量无关。

## 为什么 citations 存进 messages 的 JSONB 而不是独立表

引用是"这条回答的一部分"，不具备独立生命周期：
- 不按引用查询（"哪些回答引用了文档 X" 是低频审计需求，可扫 JSONB）；
- 不需要独立更新（回答定稿时引用一起定稿）；
- 数量少（通常 1~10 条），JSONB 完全够用。

独立表的价值在于"多对多关系"与"高频按引用反查"，本场景都不成立，
反而要多一次 JOIN 并引入"引用行与回答内容不一致"的风险。

## 多轮对话对 RAG 的核心影响（设计要点）

**检索必须用"独立化后的问题"，生成必须用"完整历史"。**

    用户第 2 轮："那二线呢？"
      ├─ 直接检索 → 召回 0（"那二线呢"没有语义信息）
      └─ 独立化 → "二线城市差旅住宿标准是多少" → 能召回

    生成时又要带上前文：
      system + [user: 一线住宿多少, assistant: 600..., user: 那二线呢] + 材料

所以本模型上：
- `messages.content` 存**原始**用户输入（用户看到的）；
- `messages.standalone_query` 存独立化后的检索用问题（见下）；
- 裁剪历史时按 token 预算保留最近 N 轮（`context_turns` 配置）。

**不把历史里的引用混进当前答案**：`citations` 只存本轮检索的来源。
上一轮引用了文档 A，这一轮引用 B，最终答案只该标 B——
混入 A 会让用户以为本轮答案也有 A 的依据。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_contracts.enums.conversations import (
    ConversationStatus,
    FeedbackRating,
    FeedbackReason,
    MessageRole,
    MessageStatus,
    RefusalReason,
)
from rag_db.base import (
    Base,
    TenantScopedMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from rag_db.models.knowledge import KnowledgeBase


class Conversation(Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin):
    """一个会话（多轮问答的容器）。"""

    __tablename__ = "conversations"

    user_id: Mapped[uuid.UUID] = mapped_column(
        # CASCADE：用户被删除时其会话一并清理。
        # 与审计日志不同（审计要保留），会话属于个人数据，
        # 删除用户时应随之删除（数据最小化原则）。
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属用户（会话是私有的，不共享）",
    )
    # 会话可以绑定到知识库（检索范围）。NULL 表示"整个租户可见范围"。
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"),
        nullable=True,
        comment="绑定知识库（NULL = 租户内全部可见库）",
    )
    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        default="新会话",
        server_default="新会话",
        comment="会话标题（首条问题生成，用户可改）",
    )
    status: Mapped[ConversationStatus] = mapped_column(
        String(32),
        nullable=False,
        default=ConversationStatus.ACTIVE,
        server_default=ConversationStatus.ACTIVE.value,
        comment="active / archived（归档后只读）",
    )

    # ---- 统计与游标 ----
    message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="消息数（冗余，避免列表页 JOIN）",
    )
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后一条消息时间（会话列表按它排序）",
    )
    # 累计成本：让用户在个人页看到"这个会话花了多少"
    total_cost_usd: Mapped[float] = mapped_column(
        # 用 double precision 而不是 Numeric：这里是**展示用**的粗略值，
        # 精确账目在 model_calls 表（A13 的对账以那里为准）。
        # 声明成 Numeric 会让人误以为它是账本。
        nullable=False,
        default=0.0,
        server_default="0",
        comment="本会话累计成本（USD，展示用；精确账目见 model_calls）",
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="软删除时间（NULL 表示未删除）。审计链需要记录存在过",
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.sequence",
    )
    knowledge_base: Mapped[KnowledgeBase | None] = relationship()

    __table_args__ = (
        # 会话列表主查询：WHERE tenant_id=? AND user_id=? AND deleted_at IS NULL
        # ORDER BY last_message_at DESC
        Index(
            "ix_conversations_tenant_user_last",
            "tenant_id",
            "user_id",
            "last_message_at",
            postgresql_where=text("deleted_at IS NULL"),
        ),
        CheckConstraint("total_cost_usd >= 0", name="cost_non_negative"),
        CheckConstraint("message_count >= 0", name="message_count_non_negative"),
    )

    def __repr__(self) -> str:
        return f"<Conversation {self.title!r} ({self.message_count} msgs)>"


class Message(Base, UUIDPrimaryKeyMixin, TenantScopedMixin, TimestampMixin):
    """会话中的一条消息（用户提问或助手回答）。"""

    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属会话",
    )
    sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="在会话内的序号（从 1 开始），用于稳定排序与分页",
    )
    role: Mapped[MessageRole] = mapped_column(
        String(16),
        nullable=False,
        comment="user / assistant",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
        comment="消息正文。用户消息为原始输入；助手消息为回答",
    )

    # ---- 助手的检索与生成信息（用户消息为 NULL）----
    standalone_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "独立化后的检索问题（多轮指代消解后的结果）。"
            "用户消息存这个也合理——它是本轮检索实际用的查询"
        ),
    )
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
        comment="引用列表（Citation 契约的 JSON 形式），只含本轮检索来源",
    )
    retrieval_meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="检索元信息（各节点耗时、召回数、是否降级），供链路可视化",
    )

    # ---- 状态 ----
    status: Mapped[MessageStatus] = mapped_column(
        String(32),
        nullable=False,
        default=MessageStatus.COMPLETED,
        server_default=MessageStatus.COMPLETED.value,
        comment="pending/streaming/completed/interrupted/failed",
    )
    refused: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="false",
        comment="是否为拒答（材料不足，没有编造）",
    )
    refusal_reason: Mapped[RefusalReason | None] = mapped_column(
        String(32),
        nullable=True,
        comment="拒答原因（用于统计'为什么拒答率上升'）",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="失败错误码（与 tasks.error_code 同一套码名）",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="失败详情（已脱敏、截断）",
    )

    # ---- 成本与用量 ----
    input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0", comment="输入 token"
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0", comment="输出 token"
    )
    cost_usd: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
        server_default="0",
        comment="本条消息的成本（展示用；精确账目见 model_calls）",
    )
    latency_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="端到端耗时（从提问到回答完成）"
    )

    # ---- 反馈 ----
    rating: Mapped[FeedbackRating | None] = mapped_column(
        String(16),
        nullable=True,
        comment="用户反馈：like / dislike",
    )
    feedback_reason: Mapped[FeedbackReason | None] = mapped_column(
        String(32),
        nullable=True,
        comment="点踩原因（固定枚举，便于聚合分析）",
    )
    feedback_comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="反馈补充说明（可空；进评估集前需脱敏）",
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        # 同一会话内序号唯一：重复序号说明"分配序号"有并发 bug
        # （两个请求同时读到 max(sequence)=4 都写 5），
        # 表现为历史顺序错乱或消息消失。
        Index(
            "uq_messages_conversation_sequence",
            "conversation_id",
            "sequence",
            unique=True,
        ),
        # 会话历史查询：WHERE conversation_id=? ORDER BY sequence
        Index("ix_messages_conversation_sequence", "conversation_id", "sequence"),
        # 清理滞留的 streaming 消息：WHERE status='streaming' AND created_at < ...
        Index(
            "ix_messages_streaming",
            "created_at",
            postgresql_where=text("status = 'streaming'"),
        ),
        # 反馈汇总（评估集来源）：WHERE tenant_id=? AND rating IS NOT NULL
        Index(
            "ix_messages_feedback",
            "tenant_id",
            "created_at",
            postgresql_where=text("rating IS NOT NULL"),
        ),
        CheckConstraint("sequence >= 1", name="sequence_positive"),
        CheckConstraint("input_tokens >= 0", name="input_tokens_non_negative"),
        CheckConstraint("output_tokens >= 0", name="output_tokens_non_negative"),
        CheckConstraint("cost_usd >= 0", name="cost_non_negative"),
        # 拒答必须带原因：否则无法统计"为什么拒答"
        CheckConstraint(
            "(refused = false) OR (refused = true AND refusal_reason IS NOT NULL)",
            name="refusal_needs_reason",
        ),
    )

    def __repr__(self) -> str:
        return f"<Message #{self.sequence} {self.role} ({len(self.content)} chars)>"


__all__ = ["Conversation", "Message"]
