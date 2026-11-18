"""对话仓储。

## 一个必须做对的并发点：`sequence` 分配

朴素做法：

    seq = session.scalar(select(func.max(Message.sequence)).where(...)) or 0
    message = Message(sequence=seq + 1, ...)

两个请求同时读到 `max=4`，都写 `5` → 撞 `uq_messages_conversation_sequence`
唯一约束。报错信息是"duplicate key"，但根因是"读取方式不对"。

本项目在 A3 已用同样的方式修过版本号分配：**在会话行上原子递增**：

    UPDATE conversations SET message_count = message_count + 1
    WHERE id = :id RETURNING message_count

用 `message_count` 当作"已分配的最大序号"——两个值天然同步：
每追加一条消息，计数 +1 且序号 = 新计数。

**代价**：`message_count` 不能独立于消息数存在（不能手工修正）。
这是有意的：它本来就该等于消息数，任何偏离都是 bug。

## 为什么"同一会话并发提问"要串行化

两个请求同时追加到同一会话：
- 序号分配虽然原子，但**上下文会错乱**：两个请求都基于"历史到第 4 条"
  去独立化改写，各自写第 5、6 条。最终历史里第 5 条与第 6 条
  是并列的（互不知情），追问时模型看到的是错乱的对话。

所以本仓储提供 `try_lock_conversation()`：用会话行的 `SELECT FOR UPDATE`
（短暂持锁，只覆盖"分配序号 + 读历史"），拿不到锁就返回 409。
**不让两个问题在同一会话里并发跑**——这是产品语义的要求，不是技术限制。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums.conversations import (
    ConversationStatus,
    FeedbackRating,
    FeedbackReason,
    MessageRole,
    MessageStatus,
    RefusalReason,
)
from rag_db.models.conversation import Conversation, Message


class ConversationRepository:
    """会话数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        title: str = "新会话",
        knowledge_base_id: uuid.UUID | None = None,
    ) -> Conversation:
        conversation = Conversation(
            tenant_id=tenant_id,
            user_id=user_id,
            title=title[:200],
            knowledge_base_id=knowledge_base_id,
            status=ConversationStatus.ACTIVE,
            message_count=0,
            last_message_at=datetime.now(UTC),
        )
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def get(
        self,
        *,
        tenant_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        include_deleted: bool = False,
    ) -> Conversation | None:
        """取会话。

        **必须同时校验 tenant_id 与 user_id**：
        会话是私有的（同一租户下 A 用户不该读到 B 用户的对话）。
        只校验 tenant 是常见疏漏——RLS 只保证租户隔离，不保证用户隔离。

        跨用户访问返回 None（由上层转成 404，不泄漏"这个 ID 是否存在"）。
        """
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == tenant_id,
            Conversation.user_id == user_id,
        )
        if not include_deleted:
            stmt = stmt.where(Conversation.deleted_at.is_(None))
        result = await self._session.execute(stmt.limit(1))
        return result.scalar_one_or_none()

    async def list_for_user(
        self,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int = 20,
        offset: int = 0,
        include_total: bool = False,
        status: str | None = None,
    ) -> tuple[Sequence[Conversation], int | None]:
        """会话列表（按最后消息时间倒序）。

        走 `ix_conversations_tenant_user_last` 部分索引（`deleted_at IS NULL`）。
        `last_message_at` 为 NULL 的新会话（还没提问）排在最后——
        用 `NULLS LAST`，否则 PG 默认把 NULL 排在最前（DESC 时）。

        `status` 过滤：归档的会话默认不出现（调用方传 active），
        但要能通过 `status=archived` 查回来——否则"归档"就等于"永久看不见"。
        """
        conditions = [
            Conversation.tenant_id == tenant_id,
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
        ]
        if status is not None:
            conditions.append(Conversation.status == status)
        base = select(Conversation).where(*conditions)

        stmt = (
            base.order_by(
                Conversation.last_message_at.desc().nullslast(),
                Conversation.created_at.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        rows = list((await self._session.execute(stmt)).scalars())

        total: int | None = None
        if include_total:
            count_stmt = select(func.count()).select_from(base.subquery())
            total = int((await self._session.execute(count_stmt)).scalar_one())

        return rows, total

    async def allocate_sequence(self, *, conversation_id: uuid.UUID, tenant_id: uuid.UUID) -> int:
        """原子分配下一个消息序号（同时递增 message_count）。

        用**单条 UPDATE ... RETURNING**：
        Python 侧 `max(sequence) + 1` 在并发下会两个请求拿到同一个号，
        随后插入时撞唯一约束——报错信息是"序号重复"，
        但根因是读取方式不对（A3 已用同样方式修过版本号分配）。

        Returns:
            新的序号（从 1 开始）。
        """
        result = await self._session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == tenant_id,
            )
            .values(
                message_count=Conversation.message_count + 1,
                last_message_at=func.now(),
                updated_at=func.now(),
            )
            .returning(Conversation.message_count)
        )
        new_count = result.scalar_one()
        return int(new_count)

    async def add_cost(self, *, conversation_id: uuid.UUID, cost_usd: float) -> None:
        """累加会话成本（原子自增）。

        用 SQL `+` 而不是"读出再写回"：并发追加消息时会丢更新，
        表现为会话成本比实际少（用户看到的花费偏低）。
        """
        if cost_usd <= 0:
            return
        await self._session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(total_cost_usd=Conversation.total_cost_usd + cost_usd)
        )

    async def rename(self, *, conversation_id: uuid.UUID, title: str) -> None:
        await self._session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(title=title[:200], updated_at=func.now())
        )

    async def archive(self, *, conversation_id: uuid.UUID) -> None:
        """归档（只读）。

        归档的会话默认不出现在列表里，但仍可通过 `status=archived` 查回——
        它是"收起"而不是"删除"。
        """
        await self._session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(status=ConversationStatus.ARCHIVED.value, updated_at=func.now())
        )

    async def unarchive(self, *, conversation_id: uuid.UUID) -> None:
        """取消归档（恢复为 active）。

        与 `archive` 成对：只用一个端点做"归档"，就没有地方安放"恢复"，
        而用户误点归档后必须能撤回。写成两个方法而不是传一个布尔参数，
        是因为调用点用 `if archived: archive() else: unarchive()` 时，
        意图比 `set_status(not archived)` 清楚得多。
        """
        await self._session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(status=ConversationStatus.ACTIVE.value, updated_at=func.now())
        )

    async def soft_delete(self, *, conversation_id: uuid.UUID) -> bool:
        """软删除。

        不物理删的原因：
        - 审计链需要"这个会话存在过";
        - 消息里的 feedback 是评估集来源，删了会丢训练数据。
        物理清理交给离线任务（按保留策略）。
        """
        result = await self._session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.deleted_at.is_(None),
            )
            .values(deleted_at=func.now(), updated_at=func.now())
            .returning(Conversation.id)
        )
        return result.scalar_one_or_none() is not None

    async def lock_for_turn(self, *, tenant_id: uuid.UUID, conversation_id: uuid.UUID) -> bool:
        """尝试锁住会话以串行化"一轮问答"。

        ## 为什么同一会话不能并发提问

        两个请求同时基于"历史到第 4 条"去独立化改写，各自追加第 5、6 条。
        历史里这两条是并列的（互不知情），下一轮追问时模型看到错乱的对话。

        所以用 `SELECT ... FOR UPDATE NOWAIT`：
        - 拿不到锁**立即失败**（不是等待）——等待会让用户以为卡住了；
        - 上层返回 409 + "该会话正在处理另一条消息"。

        **持锁时间必须极短**（只覆盖分配序号 + 读历史），
        绝不能覆盖 LLM 生成（几秒钟）——那会让用户连追问都要排队。

        Returns:
            True 表示拿到锁；False 表示会话正在被处理。
        """
        from sqlalchemy import text

        try:
            await self._session.execute(
                text(
                    "SELECT id FROM conversations "
                    "WHERE id = :cid AND tenant_id = :tid FOR UPDATE NOWAIT"
                ),
                {"cid": conversation_id, "tid": tenant_id},
            )
        except Exception:
            # NOWAIT 拿不到锁时 asyncpg 抛 LockNotAvailableError
            # （SQLAlchemy 包成 OperationalError/DBAPIError）。
            # 这里不做类型判断（驱动差异大），交给上层用 bool 处理，
            # 但**必须回滚当前事务**，否则后续语句都会失败。
            await self._session.rollback()
            return False
        return True


class MessageRepository:
    """消息数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        *,
        tenant_id: uuid.UUID,
        conversation_id: uuid.UUID,
        sequence: int,
        role: MessageRole,
        content: str = "",
        status: MessageStatus = MessageStatus.COMPLETED,
        standalone_query: str | None = None,
        citations: list[dict[str, Any]] | None = None,
        retrieval_meta: dict[str, Any] | None = None,
        refused: bool = False,
        refusal_reason: RefusalReason | None = None,
    ) -> Message:
        message = Message(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            sequence=sequence,
            role=role,
            content=content,
            status=status,
            standalone_query=standalone_query,
            citations=citations or [],
            retrieval_meta=retrieval_meta or {},
            refused=refused,
            refusal_reason=refusal_reason,
        )
        self._session.add(message)
        await self._session.flush()
        return message

    async def get(self, *, tenant_id: uuid.UUID, message_id: uuid.UUID) -> Message | None:
        result = await self._session.execute(
            select(Message).where(Message.id == message_id, Message.tenant_id == tenant_id).limit(1)
        )
        return result.scalar_one_or_none()

    async def list_recent(
        self,
        *,
        tenant_id: uuid.UUID,
        conversation_id: uuid.UUID,
        limit: int = 20,
    ) -> list[Message]:
        """取最近 N 条消息（按序号**升序**返回，便于拼 prompt）。

        **必须取最近的而不是最早的**：多轮对话的上下文价值随距离衰减，
        20 轮前的对话对当前问题几乎无帮助，还会占用 token 预算。

        实现上是"倒序取 N 条再反转"，避免 `OFFSET`（大 OFFSET 会扫全表）。
        """
        result = await self._session.execute(
            select(Message)
            .where(
                Message.tenant_id == tenant_id,
                Message.conversation_id == conversation_id,
            )
            .order_by(Message.sequence.desc())
            .limit(limit)
        )
        return list(reversed(list(result.scalars())))

    async def page(
        self,
        *,
        tenant_id: uuid.UUID,
        conversation_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        include_total: bool = False,
    ) -> tuple[Sequence[Message], int | None]:
        """分页读取会话历史（展示用，正序）。"""
        base = select(Message).where(
            Message.tenant_id == tenant_id,
            Message.conversation_id == conversation_id,
        )
        stmt = base.order_by(Message.sequence).limit(limit).offset(offset)
        rows = list((await self._session.execute(stmt)).scalars())

        total: int | None = None
        if include_total:
            count_stmt = select(func.count()).select_from(base.subquery())
            total = int((await self._session.execute(count_stmt)).scalar_one())
        return rows, total

    async def mark_completed(
        self,
        *,
        message_id: uuid.UUID,
        content: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: int,
        citations: list[dict[str, Any]],
    ) -> bool:
        """标记助手消息完成并写入结果。

        用一个方法一次写全部结果字段，而不是分多次 update：
        分多次时若中途失败会出现"内容写了但 token 没写"的半成品，
        对账时无法判断这条消息是否完整。
        """
        result = await self._session.execute(
            update(Message)
            .where(Message.id == message_id)
            .values(
                content=content,
                status=MessageStatus.COMPLETED,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                citations=citations,
                updated_at=func.now(),
            )
            .returning(Message.id)
        )
        return result.scalar_one_or_none() is not None

    async def mark_failed(
        self,
        *,
        message_id: uuid.UUID,
        status: MessageStatus,
        content: str = "",
        error_code: str | None = None,
        error_message: str | None = None,
        latency_ms: int | None = None,
    ) -> bool:
        """标记失败/中断。

        `status` 区分 `FAILED`（明确错误）与 `INTERRUPTED`（断连/超时）：
        前者内容通常为空，后者可能已有部分内容（对用户仍有用）。
        """
        values: dict[str, Any] = {
            "status": status,
            "content": content,
            "error_code": error_code,
            "error_message": error_message,
            "updated_at": func.now(),
        }
        if latency_ms is not None:
            values["latency_ms"] = latency_ms

        result = await self._session.execute(
            update(Message).where(Message.id == message_id).values(**values).returning(Message.id)
        )
        return result.scalar_one_or_none() is not None

    async def set_feedback(
        self,
        *,
        tenant_id: uuid.UUID,
        message_id: uuid.UUID,
        rating: FeedbackRating,
        reason: FeedbackReason | None = None,
        comment: str | None = None,
    ) -> bool:
        """写入用户反馈。

        允许覆盖（用户改主意）：同一条消息再次提交反馈会更新而不是报错。
        """
        result = await self._session.execute(
            update(Message)
            .where(Message.id == message_id, Message.tenant_id == tenant_id)
            .values(
                rating=rating,
                feedback_reason=reason,
                feedback_comment=comment,
                updated_at=func.now(),
            )
            .returning(Message.id)
        )
        return result.scalar_one_or_none() is not None

    async def reset_stale_streaming(
        self, *, older_than_seconds: int = 300, limit: int = 100
    ) -> int:
        """把滞留的 `streaming` 消息标成 `interrupted`。

        ## 为什么必须有这个清理

        SSE 断连/进程崩溃时，没有机会把状态改成 failed——消息会永远停在
        `streaming`。表现为"历史里有一条永远在转圈的回答"。

        清理任务按 `created_at` 超时扫描（走 `ix_messages_streaming`
        部分索引，未完成的消息很少，扫描成本低）。
        """
        from sqlalchemy import text

        # **`UPDATE ... RETURNING ... LIMIT` 语法不合法**（PostgreSQL 的
        # LIMIT 只用于 SELECT）。实测报 `语法错误 在 "LIMIT" 或附近`。
        #
        # 正确做法：用子查询选出要更新的行（带 LIMIT），
        # 再按 id 列表更新。
        result = await self._session.execute(
            text(
                "UPDATE messages SET status = 'interrupted', updated_at = now() "
                "WHERE id IN ("
                "  SELECT id FROM messages "
                "  WHERE status = 'streaming' "
                "    AND created_at < now() - make_interval(secs => :secs) "
                "  ORDER BY created_at "
                "  LIMIT :lim "
                ") "
                "RETURNING id"
            ),
            {"secs": older_than_seconds, "lim": limit},
        )
        return len(list(result.scalars()))


def _active_only(stmt: Select[Any]) -> Select[Any]:
    """过滤软删除的会话（辅助，便于复用）。"""
    return stmt.where(Conversation.deleted_at.is_(None))


__all__ = ["ConversationRepository", "MessageRepository"]
