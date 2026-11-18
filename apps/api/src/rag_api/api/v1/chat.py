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

## 业务逻辑的位置

问答编排在 `ChatService`，会话读取与轻量写在 `ConversationService`，
本模块只承担 HTTP 语义（SSE 帧格式、状态码、响应头）。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import nullcontext

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rag_adapters.retrieval.state import RetrievalConfig
from rag_api.deps import enforce_chat_ratelimit, get_db
from rag_api.deps.pagination import PageQuery, page_query
from rag_api.openapi import (
    AUTHENTICATED_ENDPOINT,
    DELETE_ENDPOINT,
    TENANT_ENDPOINT,
    TENANT_WRITE_ENDPOINT,
)
from rag_api.permissions import PermissionContext, get_permission_context
from rag_api.schemas.common import Page
from rag_api.schemas.conversations import (
    ChatRequest,
    ConversationItem,
    FeedbackRequest,
    MessageItem,
    to_conversation_page,
    to_message_page,
)
from rag_api.services.chat import ChatEvent, ChatService
from rag_api.services.conversations import ConversationService
from rag_core.logging import get_logger
from rag_core.settings import get_settings

log = get_logger(__name__)

router = APIRouter(tags=["chat"])


def _build_service(request: Request, session: AsyncSession) -> ChatService:
    """从 app.state 取已构造的 provider，装配服务。

    **provider 必须是进程级单例**（挂在 app.state）：
    - LLM provider 内部有 httpx 连接池（每请求新建会泄漏 fd）；
    - Embedding provider 持有 2GB 模型（每请求加载要 13 秒）；
    - Reranker 也是本地模型（只在实际开启重排时才构造）。

    检索配置在这里从 `Settings.retrieval` 映射成 `RetrievalConfig`：
    `ChatService` 不直接依赖 `Settings`（它只需要运行时参数），
    这样单测可以传任意配置而不必造一整份 Settings。
    """
    # **惰性构造**（见 main.get_llm_provider 的说明）：
    # 未配置 LLM 时这里会抛 ConfigError，由全局异常处理器转成
    # 503（而不是让整个 API 起不来）。
    from rag_api.main import (
        get_embedding_provider,
        get_llm_provider,
        get_reranker_provider,
    )

    settings = get_settings()
    retrieval = settings.retrieval
    # 开了 use_rerank 但 provider=none 或缺 key 时，会在请求期失败并降级为
    # RRF 顺序。这里在装配期就把它关掉（配置错误要“显式降级”而不是静默跑空），
    # 启动日志会给出对应的 warning。
    use_rerank = retrieval.use_rerank and settings.reranker.configured

    retrieval_config = RetrievalConfig(
        use_keyword=retrieval.use_keyword,
        use_rewrite=retrieval.use_rewrite,
        use_rerank=use_rerank,
        use_self_check=retrieval.use_self_check,
        self_check_refuse=retrieval.self_check_refuse,
        vector_limit=retrieval.vector_limit,
        keyword_limit=retrieval.keyword_limit,
        candidate_multiplier=retrieval.candidate_multiplier,
        final_limit=retrieval.final_limit,
        rerank_top_n=retrieval.rerank_top_n,
        rrf_k=retrieval.rrf_k,
        max_distance=retrieval.max_distance,
        max_keywords=retrieval.max_keywords,
    )

    return ChatService(
        session,
        llm=get_llm_provider(request.app),
        embedder=get_embedding_provider(request.app),
        reranker=get_reranker_provider(request.app) if use_rerank else None,
        settings=settings,
        retrieval_config=retrieval_config,
        metrics=getattr(request.app.state, "metrics", None),
        # 追踪器：与 LLM/Embedding 一样从 app.state 取（进程级单例）。
        # None 时所有 span 退化为 nullcontext，不影响业务。
        tracer=getattr(request.app.state, "tracer", None),
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
    # **限流必须在进入 SSE 之前**：流一旦开始就无法再返回 429。
    # 双层（租户 + 用户）配额见 `rag_api.deps.ratelimit`。
    dependencies=[Depends(enforce_chat_ratelimit)],
    # SSE 不能声明 JSON 响应模型，但**必须声明错误 responses**：
    # 校验失败（422）、鉴权失败（401）、会话冲突（409）都发生在
    # 流开始之前，走普通 JSON 响应。
    responses=TENANT_WRITE_ENDPOINT,
)
async def chat(
    payload: ChatRequest,
    request: Request,
    # **权限上下文必须注入**：检索返回的是文档正文，不过滤 ACL
    # 等于用户能通过提问把无权文档的内容问出来（AGENTS.md 4.8）。
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    service = _build_service(request, session)
    tracer = getattr(request.app.state, "tracer", None)

    async def event_stream() -> AsyncIterator[str]:
        # 收集流里的事件，用于**流结束后写审计**。
        #
        # 为什么审计不能提前写：审计要记"这次问答成功了没有、引用了哪些文档"，
        # 这些只有流结束才知道。提前写会记录一个"正在进行"的状态，
        # 而流可能随后失败——审计内容就错了。
        final_event: ChatEvent | None = None

        # 一轮问答 = 一条 trace。`session_id` 用会话 ID，
        # 这样 Langfuse 侧能把多轮追问串成同一条会话（同一条会话里的 trace
        # 共享 session，而不是各看各的）。新会话首轮没有 ID，此时不设 session，
        # 从第二轮开始就有。
        trace_cm = (
            tracer.atrace(
                "chat.turn",
                user_id=str(ctx.user_id),
                session_id=str(payload.conversation_id) if payload.conversation_id else None,
                tenant_id=str(ctx.tenant_id),
                query_chars=len(payload.query),
                knowledge_base_count=len(payload.knowledge_base_ids),
                conversation_id=payload.conversation_id,
            )
            if tracer is not None
            else nullcontext()
        )

        async with trace_cm:
            try:
                async for event in service.chat_stream(
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                    query=payload.query,
                    conversation_id=payload.conversation_id,
                    knowledge_base_ids=payload.knowledge_base_ids or None,
                    # 传 ACL 上下文：检索侧据此过滤越权文档
                    permission_ctx=ctx,
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
                await service.record_audit(
                    ctx=ctx,
                    query=payload.query,
                    conversation_id=payload.conversation_id,
                    knowledge_base_ids=payload.knowledge_base_ids,
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
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
    page: PageQuery = Depends(page_query),
) -> Page[ConversationItem]:
    rows, total = await ConversationService(session).list_for_user(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        limit=page.limit,
        offset=page.offset,
        include_total=page.include_total,
    )
    return to_conversation_page(rows, total=total, limit=page.limit, offset=page.offset)


@router.get(
    "/conversations/{conversation_id}/messages",
    summary="会话历史",
    response_model=Page[MessageItem],
    responses=TENANT_ENDPOINT,
)
async def list_messages(
    conversation_id: uuid.UUID,
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
    page: PageQuery = Depends(page_query),
) -> Page[MessageItem]:
    rows, total = await ConversationService(session).list_messages(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=conversation_id,
        limit=page.limit,
        offset=page.offset,
        include_total=page.include_total,
    )
    return to_message_page(rows, total=total, limit=page.limit, offset=page.offset)


@router.delete(
    "/conversations/{conversation_id}",
    status_code=204,
    summary="删除会话（软删）",
    responses=DELETE_ENDPOINT,
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> None:
    """软删除。

    **为什么软删而不是物理删**：
    - 审计链需要"这个会话存在过"（合规）；
    - 消息里的 feedback 是评估集来源，物理删会丢训练数据。

    跨用户/跨租户访问返回 404（不泄漏存在性）。
    """
    await ConversationService(session).delete(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=conversation_id,
    )


@router.post(
    "/messages/{message_id}/feedback",
    summary="消息反馈（点赞/点踩）",
    response_model=dict,
    responses=AUTHENTICATED_ENDPOINT,
)
async def submit_feedback(
    message_id: uuid.UUID,
    payload: FeedbackRequest,
    request: Request,
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """记录用户反馈。

    ## 为什么反馈是请求体而不是 query 参数

    见 `FeedbackRequest` 的 docstring：散装标量参数会被 FastAPI 按 query
    解析，前端按 REST 惯例发 JSON body 就会 422。这个 bug 单元测试里用
    TestClient 直接传参数是测不出的，只有端到端点一次才暴露。

    ## 为什么点踩原因用固定枚举

    自由文本无法聚合统计，而"哪类问题最常被踩"是优化 prompt 的
    关键输入。枚举让"本月 40% 的踩是 not_relevant"成为一条可查的结论。

    允许覆盖（用户改主意）：同一条消息再次提交会更新而不是报错。
    """
    await ConversationService(
        session, tracer=getattr(request.app.state, "tracer", None)
    ).submit_feedback(
        tenant_id=ctx.tenant_id,
        message_id=message_id,
        rating=payload.rating,
        reason=payload.reason,
        comment=payload.comment,
    )
    return {"ok": True}


__all__ = ["router"]
