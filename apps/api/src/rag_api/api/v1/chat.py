"""问答接口：SSE 流式 chat + 会话管理。

## 端点

| 方法 | 路径 | 状态码 | 说明 |
|---|---|---|---|
| POST | `/chat` | 200 | SSE 流式问答 |
| GET | `/conversations` | 200 | 会话列表 |
| GET | `/conversations/{id}/messages` | 200 | 会话历史 |
| DELETE | `/conversations/{id}` | 204 | 删除会话（软删） |
| POST | `/messages/{id}/feedback` | 200 | 点赞/点踩 |

## 为什么 chat 返回 200 而不是 201/202

SSE 是一次"长连接流"，不是"创建了资源"。HTTP 语义上它是
"对现有集合的一次查询/处理"，用 200 最合适。

**但用户消息与助手消息确实被创建了**——它们的 ID 在 `done` 事件里回传，
前端拿到后再决定是否要单独拉详情。这是"创建发生在流内部"的合理妥协。

## SSE 的三个技术要点

### 1. 必须感知客户端断连

`StreamingResponse` 会在客户端断开时取消迭代器（抛 `CancelledError`）。
**必须让这个取消向上传播到 LLM 调用**，否则：
- LLM 继续生成到结束（用户已离开，纯浪费钱）；
- 消息状态永远停在 `streaming`（历史里一条永远转圈的回答）。

`ChatService.chat_stream` 里捕获 `CancelledError` 会先落库再 `raise`，
这样既保留了部分内容，也让取消传播到上游。

### 2. 必须禁用代理缓冲

Nginx 默认 `proxy_buffering on`，会把 SSE 攒成一批发出——
表现为"用户等了 5 秒，然后所有 token 一下子全出来"。

响应头里加 `X-Accel-Buffering: no` 是标准的关缓冲方式
（Nginx 认这个头，不需要改 nginx 配置）。

### 3. 不能设 `Content-Length`

SSE 是无限流，长度未知。由 chunked transfer 编码处理。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import Principal, get_current_principal, get_db
from rag_api.deps.pagination import PageQuery, page_query
from rag_api.openapi import (
    AUTHENTICATED_ENDPOINT,
    DELETE_ENDPOINT,
    TENANT_ENDPOINT,
    TENANT_WRITE_ENDPOINT,
)
from rag_api.services.chat import ChatEvent, ChatService
from rag_contracts.api.chat import ChatRequest, ConversationItem, MessageItem
from rag_contracts.api.pagination import Page, PageMeta
from rag_contracts.enums.conversations import FeedbackRating, FeedbackReason
from rag_core.logging import get_logger
from rag_core.settings import get_settings
from rag_db.models.conversation import Conversation, Message
from rag_db.repositories import ConversationRepository, MessageRepository

log = get_logger(__name__)

router = APIRouter(tags=["chat"])


async def enforce_chat_ratelimit(
    request: Request,
    principal: Principal = Depends(get_current_principal),
) -> None:
    """问答的双层限流（租户级 + 用户级）。

    ## 为什么两层都要

    - 只限租户：一个用户能把整个租户的额度吃光（其它同事全被限）；
    - 只限用户：大量小号注册能把全平台打爆。

    ## 为什么限额比普通接口低

    一次问答要调 3~4 次 LLM（改写 + 判答 + 生成），成本是普通接口的
    几十倍。用同一个限额会让"刷查询"变相成为"刷模型配额"。

    ## Redis 故障时放行而不是拒绝

    限流的目的是保护后端，不是拒绝用户。Redis 挂了的时候：
    - 拒绝所有请求 → 全站不可用（比被刷更糟）；
    - 放行 → 后端可能被打，但至少服务着。
    选后者，并记 warning（可观测到"限流失效了"）。
    """
    from rag_core.ratelimit import RateLimiter, RateLimitPolicy, tenant_key, user_key

    redis = request.app.state.redis
    metrics = getattr(request.app.state, "metrics", None)
    limiter = RateLimiter(redis.client)

    tenant_policy = RateLimitPolicy("chat_tenant", rate=20.0, burst=40)
    user_policy = RateLimitPolicy("chat_user", rate=2.0, burst=5)

    try:
        tenant_result = await limiter.check(
            policy=tenant_policy, key=tenant_key(principal.tenant_id)
        )
        user_result = await limiter.check(
            policy=user_policy,
            key=user_key(principal.tenant_id, principal.user_id),
        )
    except Exception as exc:
        log.warning(
            "ratelimit_unavailable",
            error_type=type(exc).__name__,
            note="Redis 不可用，本次请求放行（限流是保护而非拒绝）",
        )
        return

    blocked = None
    if not tenant_result.allowed:
        blocked = tenant_result
    elif not user_result.allowed:
        blocked = user_result

    if blocked is not None:
        if metrics is not None:
            from rag_core.metrics import record_ratelimit_denied

            record_ratelimit_denied(metrics, policy=blocked.limit_name)
        from rag_core.errors import RateLimitedError

        raise RateLimitedError(
            f"请求过于频繁，请 {blocked.retry_after_s:.1f} 秒后重试",
            details={"retry_after_s": round(blocked.retry_after_s, 1)},
        )


def _build_service(request: Request, session: AsyncSession) -> ChatService:
    """从 app.state 取已构造的 provider，装配服务。

    **provider 必须是进程级单例**（挂在 app.state）：
    - LLM provider 内部有 httpx 连接池（每请求新建会泄漏 fd）；
    - Embedding provider 持有 2GB 模型（每请求加载要 13 秒）。
    """
    # **惰性构造**（见 main.get_llm_provider 的说明）：
    # 未配置 LLM 时这里会抛 ConfigError，由全局异常处理器转成
    # 503（而不是让整个 API 起不来）。
    from rag_api.main import get_embedding_provider, get_llm_provider

    return ChatService(
        session,
        llm=get_llm_provider(request.app),
        embedder=get_embedding_provider(request.app),
        settings=get_settings(),
        metrics=getattr(request.app.state, "metrics", None),
    )


def _serialize(event: ChatEvent) -> str:
    """把一个事件序列化成 SSE 帧。

    格式（SSE 规范）：

        event: delta
        data: {"text": "住"}

        （空行分隔事件）

    **data 必须是单行**：SSE 用换行分隔字段，多行 JSON 会被拆断。
    `json.dumps` 默认不产生换行（除非字符串里有 `\\n`，而它会被转义成
    `\\n` 字面量），所以安全。
    """
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {payload}\n\n"


@router.post(
    "/chat",
    summary="流式问答（SSE）",
    response_class=StreamingResponse,
    # SSE 不能声明 JSON 响应模型，但**必须声明错误 responses**：
    # 校验失败（422）、鉴权失败（401）、会话冲突（409）都发生在
    # 流开始之前，走普通 JSON 响应。
    responses=TENANT_WRITE_ENDPOINT,
)
async def chat(
    payload: ChatRequest,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    service = _build_service(request, session)

    async def event_stream() -> AsyncIterator[str]:
        # 收集流里的事件，用于**流结束后写审计**。
        #
        # 为什么审计不能提前写：审计要记"这次问答成功了没有、引用了哪些文档"，
        # 这些只有流结束才知道。提前写会记录一个"正在进行"的状态，
        # 而流可能随后失败——审计内容就错了。
        final_event: ChatEvent | None = None
        try:
            async for event in service.chat_stream(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                query=payload.query,
                conversation_id=payload.conversation_id,
                knowledge_base_ids=payload.knowledge_base_ids or None,
            ):
                if event.event in {"done", "error"}:
                    final_event = event
                yield _serialize(event)
        except Exception as exc:
            # 流已经开始后出错，无法改 HTTP 状态码（响应头已发出）。
            # 只能发一个 error 事件让前端处理。
            #
            # **不能把异常细节发给客户端**：堆栈/SQL/路径都会泄漏。
            log.error(
                "chat_stream_failed",
                error_type=type(exc).__name__,
                exc_info=True,
            )
            yield _serialize(
                ChatEvent(
                    "error",
                    {
                        "code": 90000,
                        "code_name": "INTERNAL_UNEXPECTED",
                        "message": "服务内部错误，请稍后重试",
                    },
                )
            )
        finally:
            # **审计必须写**（合规硬要求）：记录谁在什么时候问了什么、
            # 引用了哪些文档。这是"知识库被如何使用"的唯一痕迹。
            #
            # 细节脱敏：`details` 里只放 ID 与计数，**不放问题原文**
            # （问题里可能有个人信息，而审计日志保留期很长）。
            await _audit_chat(
                session=session,
                principal=principal,
                payload=payload,
                final_event=final_event,
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # 关代理缓冲（见模块 docstring 第 2 点）
            "X-Accel-Buffering": "no",
            # 告诉中间层不要缓存
            "Cache-Control": "no-cache, no-transform",
            # 有些代理会为长连接设超时，这个头提示"是流"
            "Connection": "keep-alive",
        },
    )


@router.get(
    "/conversations",
    summary="会话列表",
    response_model=Page[ConversationItem],
    responses=TENANT_ENDPOINT,
)
async def list_conversations(
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db),
    page: PageQuery = Depends(page_query),
) -> Page[ConversationItem]:
    repo = ConversationRepository(session)
    rows, total = await repo.list_for_user(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        limit=page.limit,
        offset=page.offset,
        include_total=page.include_total,
    )
    items = [_to_conversation_item(row) for row in rows]
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
    "/conversations/{conversation_id}/messages",
    summary="会话历史",
    response_model=Page[MessageItem],
    responses=TENANT_ENDPOINT,
)
async def list_messages(
    conversation_id: uuid.UUID,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db),
    page: PageQuery = Depends(page_query),
) -> Page[MessageItem]:
    service = _build_service(request, session)
    rows, total = await service.list_messages(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        conversation_id=conversation_id,
        limit=page.limit,
        offset=page.offset,
        include_total=page.include_total,
    )
    items = [_to_message_item(row) for row in rows]
    return Page(
        items=items,
        meta=PageMeta(
            limit=page.limit,
            offset=page.offset,
            total=total,
            has_more=len(items) == page.limit,
        ),
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=204,
    summary="删除会话（软删）",
    responses=DELETE_ENDPOINT,
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db),
) -> None:
    """软删除。

    **为什么软删而不是物理删**：
    - 审计链需要"这个会话存在过"（合规）；
    - 消息里的 feedback 是评估集来源，物理删会丢训练数据。

    跨用户/跨租户访问返回 404（不泄漏存在性）。
    """
    from rag_core.errors import NotFoundError

    repo = ConversationRepository(session)
    conversation = await repo.get(
        tenant_id=principal.tenant_id,
        conversation_id=conversation_id,
        user_id=principal.user_id,
    )
    if conversation is None:
        raise NotFoundError("会话不存在")
    await repo.soft_delete(conversation_id=conversation_id)


@router.post(
    "/messages/{message_id}/feedback",
    summary="消息反馈（点赞/点踩）",
    response_model=dict,
    responses=AUTHENTICATED_ENDPOINT,
)
async def submit_feedback(
    message_id: uuid.UUID,
    rating: FeedbackRating,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db),
    reason: FeedbackReason | None = None,
    comment: str | None = None,
) -> dict[str, object]:
    """记录用户反馈。

    ## 为什么点踩原因用固定枚举

    自由文本无法聚合统计，而"哪类问题最常被踩"是优化 prompt 的
    关键输入。枚举让"本月 40% 的踩是 not_relevant"成为一条可查的结论。

    允许覆盖（用户改主意）：同一条消息再次提交会更新而不是报错。
    """
    from rag_core.errors import NotFoundError

    repo = MessageRepository(session)
    message = await repo.get(tenant_id=principal.tenant_id, message_id=message_id)
    if message is None:
        raise NotFoundError("消息不存在")

    # 只允许对**助手消息**反馈：给用户自己的提问点赞没有意义
    from rag_contracts.enums.conversations import MessageRole

    if message.role != MessageRole.ASSISTANT:
        from rag_core.errors import ValidationFailedError

        raise ValidationFailedError("只能对助手回答提交反馈")

    # 评论长度限制：反馈会进评估集，过长的自由文本可能是误粘贴
    if comment is not None and len(comment) > 1000:
        from rag_core.errors import ValidationFailedError

        raise ValidationFailedError("反馈评论不能超过 1000 字符")

    # 点踩才需要原因（点赞不需要解释）
    if rating == FeedbackRating.DISLIKE and reason is None:
        from rag_core.errors import ValidationFailedError

        raise ValidationFailedError("点踩时请选择原因")

    await repo.set_feedback(
        tenant_id=principal.tenant_id,
        message_id=message_id,
        rating=rating,
        reason=reason,
        comment=comment,
    )
    return {"ok": True}


async def _audit_chat(
    *,
    session: AsyncSession,
    principal: Principal,
    payload: ChatRequest,
    final_event: ChatEvent | None,
) -> None:
    """写问答审计记录。

    放在 `finally` 里：无论流正常结束、出错、还是客户端断连，
    都要留下"这次问答发生过"的痕迹。

    **审计失败不能影响响应**：审计是旁路记录，它的失败不该让
    用户看到一个已经完成的回答变成错误。所以整体 try/except。
    """
    from rag_contracts.enums import AuditAction, AuditOutcome, AuditTargetType
    from rag_db.repositories import AuditRepository

    try:
        details: dict[str, object] = {
            # **不记录问题原文**：日志/审计保留期长，且原文可能含 PII。
            # 只记长度便于判断"是不是异常长的输入"。
            "query_length": len(payload.query),
            "conversation_id": str(payload.conversation_id) if payload.conversation_id else None,
            "knowledge_base_count": len(payload.knowledge_base_ids),
        }

        outcome = AuditOutcome.SUCCESS
        error_code = None
        if final_event is not None:
            if final_event.event == "error":
                outcome = AuditOutcome.FAILED
                error_code = str(final_event.data.get("code_name", ""))
            else:
                details["refused"] = bool(final_event.data.get("refused"))
                details["citation_count"] = len(final_event.data.get("citations") or [])
                details["input_tokens"] = int(final_event.data.get("input_tokens") or 0)
                details["output_tokens"] = int(final_event.data.get("output_tokens") or 0)
                details["message_id"] = final_event.data.get("message_id")

        await AuditRepository(session).record(
            tenant_id=principal.tenant_id,
            action=AuditAction.CHAT_COMPLETION,
            outcome=outcome,
            actor_user_id=principal.user_id,
            target_type=AuditTargetType.CONVERSATION,
            target_id=payload.conversation_id,
            details=details,
            error_code=error_code,
        )
    except Exception as exc:
        log.error(
            "chat_audit_failed",
            error_type=type(exc).__name__,
            note="审计写入失败（不影响已完成的问答）",
        )


# ---------------------------------------------------------------- 序列化


def _to_conversation_item(row: Conversation) -> ConversationItem:
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


def _to_message_item(row: Message) -> MessageItem:
    from rag_contracts.api.citations import Citation

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
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cost_usd=row.cost_usd,
        latency_ms=row.latency_ms,
    )


__all__ = ["router"]
