"""问答服务：多轮 RAG 的编排核心。

## 一次问答的完整流程

    ┌─ 1. 校验会话归属（跨用户 → 404，不泄漏存在性）
    │
    ├─ 2. 锁会话（FOR UPDATE NOWAIT）→ 拿不到返回 409
    │     只覆盖"分配序号 + 读历史"，不覆盖 LLM 生成
    │
    ├─ 3. 读历史（最近 N 条，按 token 预算裁剪）
    │
    ├─ 4. 独立化改写（多轮必需）：query + history → standalone_query
    │
    ├─ 5. 检索（用 standalone_query，不是原问题）
    │
    ├─ 6. 能否回答？（材料存在性 + LLM 判断）
    │     ├─ 不能 → 拒答（存 message，refused=True）
    │     └─ 能   → 继续
    │
    ├─ 7. 流式生成（SSE 逐 token）
    │     ├─ 先发 retrieval 事件（前端可先渲染引用卡片）
    │     └─ 再发 delta 事件
    │
    ├─ 8. 组装引用（只含本轮检索来源）
    │
    ├─ 9. 落库（消息 + 成本明细 + 会话成本累加）
    │
    └─ 10. 发 done 事件（带 message_id 与用量）

## 三个容易做错的地方

### 1. 检索用独立化问题，生成用完整历史

    第 2 轮："那二线呢？"
      检索 → standalone: "二线城市住宿标准"   ← 有语义
      生成 → 历史 + 材料 + 原问题                ← 有上下文

反过来的两种错法：
- 检索用原问题 → 召回为零（"那二线呢"没有可检索语义）；
- 生成只用原问题 → 模型不知道在问什么。

### 2. 流式过程中必须能感知客户端断连

SSE 生成到一半用户关页面时，**必须取消上游 LLM 调用**：
不取消的话，LLM 会继续生成到结束（浪费钱），且消息状态永远停在
`streaming`（历史上一条永远转圈的回答）。

实现：`asyncio.CancelledError` 捕获后把 message 标成 `interrupted`，
并记录已有内容（对用户仍有用）。

### 3. 成本必须落库到两处

- `model_calls`：每次 LLM 调用的精确明细（对账以它为准）；
- `messages.cost_usd` + `conversations.total_cost_usd`：展示用聚合。

只记一处的话：只记 message 无法做"按任务类型拆解成本"，
只记 model_calls 则每次列表页都要 JOIN 聚合（慢）。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy import text

from rag_adapters.providers.embedding import EmbeddingProvider, EmbeddingTask
from rag_adapters.providers.llm import LLMMessage, LLMProvider, LLMRole
from rag_adapters.retrieval.citations_builder import (
    AssemblyStats,
    Material,
    assemble_citations,
)
from rag_adapters.retrieval.graph import (
    RetrievalGraph,
    build_answerability_prompt,
    build_generation_prompt,
    build_self_check_prompt,
)
from rag_adapters.retrieval.keyword import extract_keywords_async
from rag_adapters.retrieval.selfcheck import (
    REFUSAL_MESSAGE,
    SelfCheckResult,
    parse_self_check_output,
    should_refuse,
)
from rag_adapters.retrieval.state import RetrievalConfig, RetrievedChunk
from rag_api.permissions.acl import PermissionContext
from rag_contracts.api.citations import Citation
from rag_contracts.enums.audit import (
    AuditAction,
    AuditOutcome,
    AuditTargetType,
    ModelTaskType,
)
from rag_contracts.enums.conversations import (
    MessageRole,
    MessageStatus,
    RefusalReason,
)
from rag_core.errors import AppError, ConflictError, NotFoundError
from rag_core.logging import get_logger
from rag_db.models.conversation import Conversation, Message
from rag_db.repositories import (
    AuditRepository,
    ConversationRepository,
    DocumentRepository,
    MessageRepository,
)
from rag_db.repositories.audit import ModelCallRepository
from rag_db.repositories.chunk import ChunkRepository, IndexVersionRepository

log = get_logger(__name__)

# 历史裁剪：最多取最近 N 条消息拼进上下文。
#
# 为什么是 10（= 5 轮）：多轮对话的上下文价值随距离快速衰减。
# 5 轮足够消解"那它呢"这类指代，而 20 轮的 token 成本是 4 倍却几乎无收益。
# 更早的历史如果需要，应该走"摘要"而不是原样塞入（未实现，见遗留）。
MAX_HISTORY_MESSAGES = 10

# 历史 token 预算（粗估：中文 1 字 ≈ 1 token）。
# 超过就从头截断——保留最近的。
MAX_HISTORY_CHARS = 4000


@dataclass
class ChatEvent:
    """SSE 事件（服务层产出，路由层序列化）。"""

    event: str
    """retrieval / delta / done / error"""

    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatOutcome:
    """一次问答的结果（非流式场景用；流式由事件承载）。"""

    message: Message
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False


class ChatService:
    """问答编排。

    ## 为什么依赖注入而不是内部构造

    LLM/Embedding/检索图都要注入：
    - 测试用假实现（不烧钱、不依赖 GPU）；
    - 生产按配置构造（换模型不改这个类）。
    """

    def __init__(
        self,
        session: Any,
        *,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        settings: Any,
        retrieval_config: RetrievalConfig | None = None,
        reranker: Any = None,
        metrics: Any = None,
        tracer: Any = None,
    ) -> None:
        self._session = session
        self._llm = llm
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings
        # 追踪器（可选）：None 时所有 span 退化为 nullcontext。
        # 用 Any 而不是 `Tracer`，避免服务层 import core 的协议——
        # 服务层只需要它提供 aspan/ageneration/annotate/record_usage。
        self._tracer = tracer
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)
        self._documents = DocumentRepository(session)
        self._chunks = ChunkRepository(session)
        self._index_versions = IndexVersionRepository(session)
        self._model_calls = ModelCallRepository(session)
        self._config = retrieval_config or RetrievalConfig(use_rewrite=True)
        self._metrics = metrics

    # ------------------------------------------------------------ 会话管理

    async def get_or_create_conversation(
        self,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID | None,
        knowledge_base_id: uuid.UUID | None = None,
    ) -> Conversation:
        """取会话或新建。

        跨用户 / 跨租户访问 → `NotFoundError`（404，不泄漏存在性）。
        """
        if conversation_id is None:
            return await self._conversations.create(
                tenant_id=tenant_id,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
            )

        conversation = await self._conversations.get(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        if conversation is None:
            # 用 NotFound 而不是 Forbidden：不泄漏"这个会话是否存在"。
            # 否则攻击者能通过 403/404 的差异枚举别人的会话 ID。
            raise NotFoundError("会话不存在")
        return conversation

    # ------------------------------------------------------------ 问答主流程
    async def record_audit(
        self,
        *,
        ctx: PermissionContext,
        query: str,
        conversation_id: uuid.UUID | None,
        knowledge_base_ids: list[uuid.UUID],
        final_event: ChatEvent | None,
    ) -> None:
        """写问答审计记录。

        放在流的 `finally` 里调用：无论流正常结束、出错、还是客户端断连，
        都要留下"这次问答发生过"的痕迹。

        **审计失败不能影响响应**：审计是旁路记录，它的失败不该让
        用户看到一个已经完成的回答变成错误。所以整体 try/except。
        """
        try:
            details: dict[str, object] = {
                # **不记录问题原文**：日志/审计保留期长，且原文可能含 PII。
                # 只记长度便于判断"是不是异常长的输入"。
                "query_length": len(query),
                "conversation_id": str(conversation_id) if conversation_id else None,
                "knowledge_base_count": len(knowledge_base_ids),
            }

            outcome = AuditOutcome.SUCCESS
            error_code = None
            if final_event is not None:
                if final_event.event == "error":
                    outcome = AuditOutcome.FAILED
                    error_code = str(final_event.data.get("code_name", ""))
                else:
                    details["refused"] = bool(final_event.data.get("refused"))
                    details["refusal_reason"] = final_event.data.get("refusal_reason")
                    details["degraded"] = bool(final_event.data.get("degraded"))
                    details["citation_count"] = len(final_event.data.get("citations") or [])
                    details["input_tokens"] = int(final_event.data.get("input_tokens") or 0)
                    details["output_tokens"] = int(final_event.data.get("output_tokens") or 0)
                    details["message_id"] = final_event.data.get("message_id")

            await AuditRepository(self._session).record(
                tenant_id=ctx.tenant_id,
                action=AuditAction.CHAT_COMPLETION,
                outcome=outcome,
                actor_user_id=ctx.user_id,
                target_type=AuditTargetType.CONVERSATION,
                target_id=conversation_id,
                details=details,
                error_code=error_code,
            )
        except Exception as exc:
            log.error(
                "chat_audit_failed",
                error_type=type(exc).__name__,
                note="审计写入失败（不影响已完成的问答）",
            )

    # ------------------------------------------------------------ 问答主流程

    async def chat_stream(
        self,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        query: str,
        conversation_id: uuid.UUID | None = None,
        knowledge_base_ids: list[uuid.UUID] | None = None,
        permission_ctx: Any = None,
    ) -> AsyncIterator[ChatEvent]:
        """流式问答。产出 `ChatEvent` 序列。

        ## 事件顺序（前端依赖这个顺序）

            retrieval  ← 检索完成，引用卡片可先渲染
            delta × N  ← 逐 token
            done       ← 结束（含 message_id、用量、引用）

        出错时：
            error      ← 带 code/code_name/message

        **不在这里做事务提交**：事务边界由路由层（`get_db` 依赖）控制。
        服务层只 flush，不 commit——与 worker handler 的约定一致
        （见 AGENTS.md 4.11）。

        ## `permission_ctx` 为什么是必需的安全参数

        检索返回的是**文档正文**。不过滤 ACL 等于"用户可以通过提问
        把无权文档的内容问出来"——比直接返回文档更隐蔽，因为泄漏
        发生在答案里（AGENTS.md 4.8："检索是最危险的路径"）。

        参数类型是 `Any` 而不是 `PermissionContext`：`services` 层
        不 import `permissions`（避免 app 内部循环依赖）。缺省 `None`
        表示**不做过滤**，只允许在内部脚本/评估里用；
        路由层必须传，`test_acl_retrieval.py` 会盯着这一点。
        """
        started = time.perf_counter()

        conversation = await self.get_or_create_conversation(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        conversation_id = conversation.id

        # ---- 锁会话：同一会话不能并发提问 ----
        #
        # 两个请求同时基于"历史到第 4 条"改写，各写第 5、6 条会形成
        # 并列的历史（互不知情），下一轮追问时模型看到错乱对话。
        #
        # 用 NOWAIT 立即失败（不是等待）：等待会让用户以为界面卡住。
        locked = await self._conversations.lock_for_turn(
            tenant_id=tenant_id, conversation_id=conversation_id
        )
        if not locked:
            raise ConflictError("该会话正在处理另一条消息，请稍后再试")

        # ---- 分配序号（用户消息）----
        user_seq = await self._conversations.allocate_sequence(
            conversation_id=conversation_id, tenant_id=tenant_id
        )
        user_message = await self._messages.add(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            sequence=user_seq,
            role=MessageRole.USER,
            content=query,
            status=MessageStatus.COMPLETED,
        )

        # ---- 读历史（不含刚插入的这条）----
        history_rows = await self._messages.list_recent(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            limit=MAX_HISTORY_MESSAGES + 1,
        )
        # 排除刚插入的当前问题（用 id 而不是序号比较：
        # 序号可能因并发回收而重排，id 是稳定标识）
        prior = [m for m in history_rows if m.id != user_message.id]
        # `m.role` 是 String 列，读回来是 str 而不是枚举
        # （SQLAlchemy 不做自动转换，除非在模型上配 TypeDecorator）。
        # 直接 `.value` 会 AttributeError——实测踩到。
        history = _trim_history([(str(m.role), m.content) for m in prior])

        # ---- 分配序号（助手消息占位）----
        assistant_seq = await self._conversations.allocate_sequence(
            conversation_id=conversation_id, tenant_id=tenant_id
        )
        assistant = await self._messages.add(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            sequence=assistant_seq,
            role=MessageRole.ASSISTANT,
            content="",
            status=MessageStatus.STREAMING,
        )

        # 会话标题：首轮用问题前 30 字（用户可改）
        if conversation.message_count <= 2 and conversation.title == "新会话":
            await self._conversations.rename(
                conversation_id=conversation_id, title=_make_title(query)
            )

        # ---- 检索 ----
        try:
            retrieval_state, materials, visible_document_ids = await self._retrieve(
                tenant_id=tenant_id,
                query=query,
                history=history,
                knowledge_base_ids=knowledge_base_ids,
                conversation=conversation,
                permission_ctx=permission_ctx,
            )
        except AppError as exc:
            await self._messages.mark_failed(
                message_id=assistant.id,
                status=MessageStatus.FAILED,
                error_code=exc.code_name,
                error_message=_safe_message(exc),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            yield ChatEvent(
                "error",
                {
                    "code": int(exc.code),
                    "code_name": exc.code_name,
                    "message": exc.message,
                },
            )
            return

        # 先把独立化后的问题写回用户消息（便于审计"检索实际用了什么"）
        # 把独立化后的查询写回用户消息（审计"检索实际用了什么"）。
        # 直接 UPDATE 而不是走仓储：这是单字段的辅助写入，
        # 为它加一个仓储方法会让仓储接口膨胀（且没有复用价值）。
        await self._session.execute(
            text("UPDATE messages SET standalone_query = :sq WHERE id = :uid"),
            {
                "sq": retrieval_state.rewritten_query or query,
                "uid": user_message.id,
            },
        )

        # ---- 能否回答？----
        answerability = None
        if materials:
            answerability = await self._judge_answerability(
                tenant_id=tenant_id,
                query=retrieval_state.rewritten_query or query,
                materials=materials,
            )

        if should_refuse(materials=[m.content for m in materials], answerability=answerability):
            reason = (
                RefusalReason.NO_MATERIALS if not materials else RefusalReason.INSUFFICIENT_EVIDENCE
            )
            await self._messages.mark_failed(
                message_id=assistant.id,
                status=MessageStatus.COMPLETED,
                content=REFUSAL_MESSAGE,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            await self._session.execute(
                text("UPDATE messages SET refused = true, refusal_reason = :r WHERE id = :mid"),
                {"r": reason.value, "mid": assistant.id},
            )
            # 拒答也要发 retrieval（sources_found 为命中数，拒答时可能仍 >0）
            # + done。前端靠 `refused=true` 决定显示拒答 UI。
            yield ChatEvent(
                "retrieval",
                {"sources_found": len(materials), "refused": True},
            )
            yield ChatEvent("delta", {"text": REFUSAL_MESSAGE, "message_id": str(assistant.id)})
            yield ChatEvent(
                "done",
                {
                    "message_id": str(assistant.id),
                    "conversation_id": str(conversation_id),
                    "refused": True,
                    "refusal_reason": reason.value,
                    "citations": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "degraded": False,
                },
            )
            return

        # ---- 通知前端"检索完成" ----
        #
        # **只发数量，不发引用**。
        #
        # 初版在这里把全部检索命中当成 `Citation` 列表发给前端，于是：
        # 1. 答案还没生成，就显示「来源 1 2 3 4 5」；
        # 2. 编号按**材料顺序**编，而答案里的 `[n]` 是模型自己写的——
        #    模型先引用第 2 条材料时，答案写 `[1]` 而引用卡片 1 指向
        #    第 1 条材料，用户点开看到的是**错的那段原文**；
        # 3. 检索 5 条 ≠ 引用 5 条，另外 4 条是噪音却看起来像"依据"。
        #
        # 引用必须在生成结束后按答案里的标记组装（见下面 `assemble_citations`）。
        # 代价是生成期间看不到引用——这个取舍是对的：引用是"依据"，
        # 依据必须是真的。
        yield ChatEvent(
            "retrieval",
            {"sources_found": len(materials), "refused": False},
        )

        # ---- 流式生成 ----
        prompt = build_generation_prompt(
            # 检索用独立化问题（指代已消解），生成带上历史提供上下文。
            # 历史不是证据：提示词里已明确约束（见 build_generation_prompt）。
            query=retrieval_state.rewritten_query or query,
            materials=_as_chunks(materials),
            history=history,
        )
        messages = [LLMMessage(role=LLMRole.USER, content=prompt)]

        collected: list[str] = []
        gen_started = time.perf_counter()
        # **流式 usage 靠回调收集**：OpenAI 兼容协议只在最后一个 chunk
        # 给 usage，而 stream() 只 yield 文本。不收集的话流式路径的
        # 成本永远记 0 —— 对账会显示"有回答但没成本"，直接账不平。
        usage_holder: list[Any] = []

        def _on_usage(usage: Any) -> None:
            usage_holder.append(usage)

        usage: tuple[int, int] = (0, 0)
        # generation 类型让 Langfuse 把它归到"模型调用"而不是普通 span，
        # 这样才能按 provider/model 聚合 token 与延迟。
        # **只记长度不记 prompt 正文**（见 rag_core.tracing 的脱敏约定）。
        generation_cm = (
            self._tracer.ageneration(
                "llm.stream",
                provider=self._llm.name,
                model=self._llm.model_name,
                temperature=0.3,
                prompt_chars=len(prompt),
            )
            if self._tracer is not None
            else nullcontext()
        )

        try:
            async with generation_cm as generation_span:
                async for delta in self._llm.stream(
                    messages, temperature=0.3, usage_callback=_on_usage
                ):
                    collected.append(delta)
                    yield ChatEvent("delta", {"text": delta})

                # usage 在流结束后才齐。写进 span 让 trace 里能看到单次成本量级；
                # 金额的权威来源仍是 `model_calls`（见 `_record_cost`）。
                usage = _extract_usage(usage_holder)
                if self._tracer is not None:
                    self._tracer.record_usage(
                        generation_span,
                        input_tokens=usage[0],
                        output_tokens=usage[1],
                    )
        except asyncio.CancelledError:
            # **客户端断连**：必须把已有内容落库并标成 interrupted。
            # 不落库的话用户重新打开会话会看到"一条空白回答"。
            await self._mark_interrupted(
                assistant_id=assistant.id,
                content="".join(collected),
                started=started,
            )
            raise
        except Exception as exc:
            await self._messages.mark_failed(
                message_id=assistant.id,
                status=MessageStatus.FAILED,
                content="".join(collected),
                error_code=_error_code_of(exc),
                error_message=_safe_message(exc),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            yield ChatEvent(
                "error",
                {
                    "code": 80001,
                    "code_name": "PROVIDER_UNAVAILABLE",
                    "message": "模型服务暂时不可用，请重试",
                },
            )
            return

        answer = "".join(collected)
        gen_latency_ms = int((time.perf_counter() - gen_started) * 1000)

        # ---- 生成后自省：答案里的事实是否都有材料依据 ----
        #
        # ## 为什么这是独立的一次调用
        #
        # `answerability`（上面，生成前）判断"材料能否回答" → 决定拒答；
        # `support`（这里，生成后）判断"答案有没有编" → 拦幻觉。
        # 两者语义不同，混用会让"模型碰巧拒答"被当成"有支撑"而通过
        # （A11 实测，见 ADR 0003）。
        #
        # ## 流式带来的限制
        #
        # 答案已经逐 token 发给用户，无法"撤销"。所以默认动作是 `flag`：
        # 保留答案，`done.degraded=true` + 指标/审计供观察；
        # 配置为 `refuse` 时把库里的内容替换为拒答，并在 `done` 里
        # 返回 `refused=true`（`done` 是权威状态，前端据此覆盖已渲染文本）。
        support: SelfCheckResult | None = None
        self_check_failed = False
        if self._config.use_self_check and answer.strip():
            support = await self._judge_support(
                tenant_id=tenant_id,
                query=retrieval_state.rewritten_query or query,
                answer=answer,
                materials=materials,
            )
            if support is None:
                self._record_self_check_metric(result="error")
            elif not support.can_answer and support.raw_output:
                # raw_output 为空 = 解析失败（保守判 false）。解析失败不应
                # 触发拒答：一次 JSON 格式抖动不该让用户拿不到任何答案。
                self_check_failed = True
                log.warning(
                    "answer_not_supported_by_materials",
                    message_id=str(assistant.id),
                    reason=support.reason[:200],
                )
                self._record_self_check_metric(result="unsupported")
            else:
                self._record_self_check_metric(result="supported")

        self_check_refused = self_check_failed and self._config.self_check_refuse
        if self_check_refused:
            log.warning(
                "answer_replaced_by_refusal",
                message_id=str(assistant.id),
                note="self_check_action=refuse：生成后自省未通过",
            )
            answer = REFUSAL_MESSAGE

        # ---- 组装引用（生成结束后，按答案里的 [n] 标记）----
        #
        # ## 为什么必须等到生成结束
        #
        # 引用的定义是"答案的哪句话依据了哪段原文"。答案没定稿时，
        # 这个映射不存在——生成前能拿到的只有"检索命中了哪几段"，
        # 那是**候选**不是**依据**。
        #
        # 初版把候选当依据提前发给前端，导致三个错误（见 `retrieval`
        # 事件处的注释）：编号对不上、数量对不上、时机不对。
        #
        # `assemble_citations` 做对了这些：
        # - 按答案里 `[n]` **首次出现顺序**重新连续编号；
        # - 同一材料被引用多次只出一条（保证 index 连续）；
        # - 越界编号（模型引用了不存在的材料）**丢弃引用但保留答案**；
        # - 片段级定位（`locate_in_chunk`）：quote 是答案那句话对应的原文，
        #   不是整块 chunk 的前 300 字；
        # - `visible_document_ids` 二次校验，防越权引用混进答案。
        if self_check_refused:
            # 答案已被替换为拒答：它没有任何引用，不能保留原答案的引用
            # （否则用户看到"拒答 + 3 条来源"，语义自相矛盾）。
            citations: list[Citation] = []
            citation_stats = AssemblyStats()
        else:
            citations_result, citation_stats = assemble_citations(
                answer=answer,
                materials=materials,
                visible_document_ids=visible_document_ids,
            )
            citations = citations_result.citations

        # ---- 落库：消息 + 成本 ----
        # 注意：`usage` 已在生成 span 内解析完（见上面），这里不再重复提取。
        cost_usd = 0.0
        await self._record_cost(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=assistant.id,
            input_tokens=usage[0],
            output_tokens=usage[1],
            latency_ms=gen_latency_ms,
        )
        cost_usd = await self._accumulate_cost(
            conversation_id=conversation_id,
            input_tokens=usage[0],
            output_tokens=usage[1],
        )

        await self._messages.mark_completed(
            message_id=assistant.id,
            content=answer,
            input_tokens=usage[0],
            output_tokens=usage[1],
            cost_usd=cost_usd,
            latency_ms=int((time.perf_counter() - started) * 1000),
            citations=[c.model_dump(mode="json") for c in citations],
        )
        if self_check_refused:
            # 自省判"没有材料支撑"：与"材料不足"共用 insufficient_evidence，
            # 不新增枚举值（发布过的枚举不能改，前端会硬编码）。
            await self._session.execute(
                text("UPDATE messages SET refused = true, refusal_reason = :r WHERE id = :mid"),
                {
                    "r": RefusalReason.INSUFFICIENT_EVIDENCE.value,
                    "mid": assistant.id,
                },
            )
        await self._conversations.add_cost(conversation_id=conversation_id, cost_usd=cost_usd)

        # 检索元信息（B4 链路可视化用）
        await self._session.execute(
            text("UPDATE messages SET retrieval_meta = CAST(:m AS jsonb) WHERE id = :mid"),
            {
                "m": json.dumps(
                    {
                        "timings": {k: round(v, 1) for k, v in retrieval_state.timings.items()},
                        "candidate_count": len(retrieval_state.candidates),
                        "final_count": len(retrieval_state.final),
                        "rewritten_query": retrieval_state.rewritten_query,
                        # 双路召回的真实贡献：融合后看不出"关键词路到底有没有命中"，
                        # 必须单独记各路原始条数（ADR 0003 的教训）。
                        "source_counts": retrieval_state.source_counts,
                        "reranked": retrieval_state.reranked,
                        "strategy": self._retrieval_strategy(),
                        # 生成后自省：`supported=None` 且 enabled=true 表示调用失败
                        # （不是"支持"）。管理端要能区分这两种情况。
                        "self_check": {
                            "enabled": self._config.use_self_check,
                            # raw_output 为空 = 解析失败：记为 None（“没判断出来”），
                            # 不能写成 false（那会被误读成“确认无支撑”）。
                            "supported": (
                                None
                                if support is None or not support.raw_output
                                else support.can_answer
                            ),
                            "failed": self_check_failed,
                            "action": "refuse" if self_check_refused else "flag",
                        },
                        # 引用质量指标：让管理端能看出"模型是不是经常
                        # 引用不存在的编号"（out_of_range）与"引用了但
                        # 定位不到原文"（locate_failed）。两者都不报错，
                        # 只能靠埋点观察。
                        "citation_markers": citation_stats.markers_found,
                        "citation_built": citation_stats.citations_built,
                        "citation_out_of_range": citation_stats.out_of_range,
                        "citation_locate_failed": citation_stats.locate_failed,
                    },
                    ensure_ascii=False,
                ),
                "mid": assistant.id,
            },
        )

        # ---- 指标（A14 接点）----
        metrics = getattr(self._settings, "_metrics", None)
        if metrics is None:
            # 从 app.state 拿不到时跳过（服务层不持有 app 引用）。
            # 路由层会通过 `attach_metrics` 注入。
            metrics = self._metrics
        if metrics is not None:
            from rag_core.metrics import (
                record_citation,
                record_llm_call,
                record_retrieval,
            )

            record_retrieval(
                metrics,
                strategy=self._retrieval_strategy(),
                duration_s=time.perf_counter() - started,
                result_count=len(materials),
            )
            # 引用质量：locate_failed 已包含在 citations_built 里，
            # 不能对同一条既记 located 又记 failed（会双计）。
            located_count = citation_stats.citations_built - citation_stats.locate_failed
            for _ in range(located_count):
                record_citation(metrics, located=True)
            for _ in range(citation_stats.locate_failed):
                record_citation(metrics, located=False)
            record_llm_call(
                metrics,
                task=ModelTaskType.CHAT.value,
                provider=self._llm.name,
                model=self._llm.model_name,
                status="success",
                duration_s=gen_latency_ms / 1000,
                input_tokens=usage[0],
                output_tokens=usage[1],
                cost_usd=cost_usd,
            )

        yield ChatEvent(
            "done",
            {
                "message_id": str(assistant.id),
                "conversation_id": str(conversation_id),
                "refused": self_check_refused,
                "refusal_reason": (
                    RefusalReason.INSUFFICIENT_EVIDENCE.value if self_check_refused else None
                ),
                "citations": [c.model_dump(mode="json") for c in citations],
                "input_tokens": usage[0],
                "output_tokens": usage[1],
                "cost_usd": round(cost_usd, 8),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                # flag 模式：答案保留但标记"有材料不支撑的事实"，前端可展示警示；
                # refuse 模式不是降级（是安全动作），因此为 false。
                "degraded": self_check_failed and not self_check_refused,
            },
        )

    # ------------------------------------------------------------ 内部

    async def _retrieve(
        self,
        *,
        tenant_id: uuid.UUID,
        query: str,
        history: list[tuple[str, str]],
        knowledge_base_ids: list[uuid.UUID] | None,
        conversation: Conversation,
        permission_ctx: Any = None,
    ) -> tuple[Any, list[Material], set[uuid.UUID]]:
        """执行检索（向量 + 关键词 → RRF → 可选 rerank），回填成 `Material`。

        ## 职责边界

        这个方法的职责是**把检索图接到数据库与权限上**：
        - 构造向量/关键词/改写/重排的闭包（它们需要 tenant、ACL、索引版本）；
        - 跑 `RetrievalGraph`（融合与重排的算法在 adapters 里，可单测）；
        - 按最终排名回填位置信息与文档标题（引用高亮需要）。

        ## 多轮的关键：先用历史做独立化改写

        "那二线呢？" 单独检索召回为零。必须先用历史消解指代：

            history + "那二线呢？" → "二线城市住宿标准是多少"

        **只有多轮（history 非空）才改写**：首轮问题是完整的，
        改写反而可能引入偏差（且多花一次 LLM 调用）。

        ## 返回的 `Material.relevance_score` 是**向量距离**

        RRF 分与 rerank 分都是"越大越相关"，而引用契约是"余弦距离，
        0 最相关"。这里用 `vector_distance_by_id` 单独留存向量距离，
        关键词独有的 chunk 则为 None（"不知道距离"比"方向相反的分数"好）。
        """
        # 检索范围：请求指定 > 会话绑定 > 租户内全部
        kb_ids = knowledge_base_ids or (
            [conversation.knowledge_base_id] if conversation.knowledge_base_id else None
        )

        # ---- ACL 过滤条件 ----
        #
        # **这是安全边界，不是优化**。检索返回的是文档正文，不过滤等于
        # 用户能通过提问把无权文档的内容问出来。
        #
        # ## 为什么用 `confidentiality_filter_for_chunks`
        #
        # 它用 `Chunk.document_id.in_(SELECT documents.id WHERE ...)`，
        # 是纯子查询，塞进 `select(Chunk).where(...)` 不会污染 FROM 子句。
        # 而 `document_access_filter` 直接引用 `Document.xxx` 列，会生成
        # `FROM chunks, documents` 的**笛卡尔积**（无 join 条件），
        # SECRET 内容照样漏出来，且不报错。
        #
        # 向量路与关键词路**必须共用同一个 filter**：漏一路等于全漏，
        # 而且 RRF 只看排名，越权项会被融合抬到更前面。
        access_filter: Any = None
        if permission_ctx is not None:
            from rag_api.permissions.acl import confidentiality_filter_for_chunks

            access_filter = confidentiality_filter_for_chunks(permission_ctx)

        # 开了重排但没有 provider：显式降级到融合顺序（写日志，不静默）。
        config = self._config
        if config.use_rerank and self._reranker is None:
            log.warning(
                "rerank_enabled_without_provider",
                note="use_rerank=true 但未装配 reranker，本次退化为 RRF 顺序",
            )
            config = replace(config, use_rerank=False)

        # **向量距离必须单独留存**：RRF 之后 `state.candidates[].score`
        # 变成 RRF 分（越大越相关），rerank 之后又变成模型分。
        # 而 `Citation.relevance_score` 的契约是"余弦距离，0 最相关"，
        # 直接把图中的 score 写进去会让前端显示方向相反的分数。
        vector_distance_by_id: dict[str, float] = {}

        async def vector_search(query: str, *, limit: int) -> list[tuple[str, str, float]]:
            """向量路：按知识库分别取 active 索引版本后检索。"""
            started = time.perf_counter()
            query_vector = await self._embedder.embed_one(query, task=EmbeddingTask.QUERY)

            hits: list[tuple[Any, float]] = []
            if kb_ids:
                for kb_id in kb_ids:
                    # **必须带 active 索引版本**：重建期间库里有两套向量，
                    # 不带版本过滤会返回重复内容且距离不可比。
                    version = await self._index_versions.get_active(knowledge_base_id=kb_id)
                    if version is None:
                        continue
                    page = await self._chunks.search_vector(
                        tenant_id=tenant_id,
                        query_embedding=query_vector,
                        knowledge_base_ids=[kb_id],
                        limit=limit,
                        max_distance=config.max_distance,
                        index_version_ids=[version.id],
                        access_filter=access_filter,
                        candidate_multiplier=config.candidate_multiplier,
                    )
                    hits.extend(page)
                hits.sort(key=lambda item: item[1])
                hits = hits[:limit]
            else:
                # 未指定库：租户内全部（不带版本过滤，取全局最新）
                hits = await self._chunks.search_vector(
                    tenant_id=tenant_id,
                    query_embedding=query_vector,
                    limit=limit,
                    max_distance=config.max_distance,
                    access_filter=access_filter,
                    candidate_multiplier=config.candidate_multiplier,
                )

            for chunk, distance in hits:
                vector_distance_by_id[str(chunk.id)] = distance

            log.info(
                "vector_search_completed",
                limit=limit,
                hits=len(hits),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return [(str(chunk.id), chunk.content, distance) for chunk, distance in hits]

        async def keyword_search(query: str, *, limit: int) -> list[tuple[str, str, float]]:
            """关键词路：jieba 提词后走 PostgreSQL 词项检索（ts_rank）。"""
            keywords = await extract_keywords_async(query, max_keywords=config.max_keywords)
            if not keywords:
                return []
            hits = await self._chunks.search_keyword(
                tenant_id=tenant_id,
                keywords=keywords,
                knowledge_base_ids=kb_ids,
                limit=limit,
                access_filter=access_filter,
            )
            return [(str(chunk.id), chunk.content, score) for chunk, score in hits]

        async def rewrite(query: str, *, history: list[tuple[str, str]]) -> str:
            """首轮不改写：完整问题改写只会多一次 LLM 调用并可能引入偏差。"""
            if not history:
                return ""
            return await self._standalone_query(tenant_id=tenant_id, query=query, history=history)

        async def rerank(query: str, documents: Sequence[str]) -> list[float]:
            """重排失败时退回融合顺序（降级链：Reranker 挂 → 只用 RRF）。

            返回全 0 分而不是抛异常：图里是稳定排序，全 0 等于保留
            入参顺序，即融合顺序。抛异常会让整个问答失败，
            而"排序退化"比"没有答案"好得多。
            """
            if self._reranker is None:  # pragma: no cover - 构造时已保证
                return [0.0] * len(documents)
            try:
                result = await self._reranker.rerank(query, list(documents))
            except Exception as exc:
                log.warning(
                    "rerank_failed_fallback_to_fusion",
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                )
                return [0.0] * len(documents)
            return list(result.scores)

        graph = RetrievalGraph(
            vector_searcher=vector_search,
            # 关键词路总是提供实现，是否执行由 config.use_keyword 控制；
            # 图会在构造时校验"开了开关必须有实现"。
            keyword_searcher=keyword_search,
            reranker=rerank if config.use_rerank else None,
            rewriter=rewrite if config.use_rewrite else None,
            tracer=self._tracer,
            # 生成后自省不属于检索图（图不负责生成）。图里保留这个开关
            # 只用于构造期一致性校验；真正执行在 `chat_stream` 里，
            # 所以这里把它置 false，避免"开了却没有 checker"的误报。
            config=replace(config, use_self_check=False),
        )
        # `retrieval.pipeline` 是各节点的父 span：父 span 的耗时就是"检索总共花了多久"，
        # 子 span 回答"花在哪一步"。两者都必要（只看子节点无法知道总量，
        # 只看总量无法定位瓶颈）。
        pipeline_cm = (
            self._tracer.aspan(
                "retrieval.pipeline",
                query_chars=len(query),
                knowledge_base_count=len(kb_ids or []),
                strategy=self._retrieval_strategy(),
            )
            if self._tracer is not None
            else nullcontext()
        )
        async with pipeline_cm as pipeline_span:
            state = await graph.run(query, history=history)
            if self._tracer is not None:
                self._tracer.annotate(
                    pipeline_span,
                    candidate_count=len(state.candidates),
                    final_count=len(state.final),
                    source_counts=dict(state.source_counts),
                )

        # ---- 把检索结果回填成完整材料 ----
        #
        # 检索图只负责"选出哪些 chunk"（ID + 内容 + 分数）。引用需要
        # 位置信息（页码/偏移/bbox）与文档标题，这些必须从库里回填。
        final_ids: list[uuid.UUID] = []
        for item in state.final:
            try:
                final_ids.append(uuid.UUID(item.chunk_id))
            except ValueError:  # pragma: no cover - 检索器只返回数据库 ID
                continue

        chunks = await self._chunks.get_current_by_ids(
            tenant_id=tenant_id,
            chunk_ids=final_ids,
            access_filter=access_filter,
        )
        chunk_by_id = {str(chunk.id): chunk for chunk in chunks}

        # 标题批量查（逐条查是 N+1；`_document_title` 已删除）
        titles = await self._documents.get_titles(
            tenant_id=tenant_id,
            document_ids=[chunk.document_id for chunk in chunks],
        )

        final_materials: list[Material] = []
        for item in state.final:
            chunk = chunk_by_id.get(item.chunk_id)
            if chunk is None:
                # 检索后被删除/换成旧版本：跳过而不是拼进 prompt。
                # 这会让 materials 少于 final_limit，是正确行为。
                log.info(
                    "retrieved_chunk_missing_on_reload",
                    chunk_id=item.chunk_id,
                    note="检索后被删除或不再是当前版本",
                )
                continue
            final_materials.append(
                Material(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=titles.get(chunk.document_id, "未知文档"),
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    page_no=chunk.page_no,
                    bbox=chunk.bbox,
                    section_path=tuple(chunk.section_path or ()),
                    # **只用向量距离**：RRF/rerank 的分数语义不同，
                    # 写进 Citation 会让前端把"越大越相关"当成
                    # "越大越不相关"（契约是余弦距离）。
                    relevance_score=vector_distance_by_id.get(item.chunk_id),
                    pages=(),
                )
            )

        # 可见文档集：由**检索结果自身**推导。
        #
        # 为什么这是"第二道校验"而不是多余：
        # `search_vector` 已经用 `confidentiality_filter_for_chunks` 过滤过，
        # 理论上 materials 里只会有可见文档。但 AGENTS.md 4.15 要求
        # 生成后**再校验一次引用**，因为这些都可能引入未过滤内容：
        # - 模型"记错"来源（把 A 文档的内容归到 B）；
        # - 父块回填 / 上下文压缩带进了别的 chunk；
        # - 未来加多查询融合时某个子查询漏了过滤条件。
        #
        # 引用越界是最隐蔽的泄漏：它嵌在答案里，用户看不出异常。
        visible_document_ids = {material.document_id for material in final_materials}
        return state, final_materials, visible_document_ids

    async def _standalone_query(
        self, *, tenant_id: uuid.UUID, query: str, history: list[tuple[str, str]]
    ) -> str:
        """把当前问题改写为不依赖历史的独立问题（指代消解）。

        ## 为什么这是多轮 RAG 的必要环节

        用户第 2 轮问"那二线呢？"——这句话：
        - 拿去检索 → 召回为零（不含可检索语义）；
        - 拿去问 LLM → 模型也不知道在问什么（如果没带历史）。

        改写后："二线城市住宿标准是多少" —— 检索和生成都能用。

        ## 提示词设计的三条约束

        1. **要求只输出改写结果**，不要解释（否则会把说明文字一起送进检索）；
        2. **明确"若已经完整则原样返回"** —— 防止把完整问题改成画蛇添足的形式；
        3. **禁止引入历史里没有的信息** —— 避免改写虚构出用户没问的内容
           （那会让检索跑偏，且用户无法察觉）。

        失败时返回空串（调用方回退到原问题）——不改写比改错好。
        """
        # 只取最近 3 轮拼进改写提示词。
        #
        # 为什么不是全部历史：改写只需要"指代对象"在附近的上下文。
        # 3 轮足够，更多会稀释指令（模型容易开始总结整段对话）。
        recent = history[-6:]
        transcript = "\n".join(
            f"{'用户' if role == 'user' else '助手'}：{content[:200]}" for role, content in recent
        )

        prompt = (
            "任务：把「当前问题」改写成不依赖对话历史的独立问题。\n\n"
            "规则：\n"
            "1. 只输出改写后的问题本身，不要解释、不要加引号；\n"
            "2. 如果当前问题已经完整（不含「那」「这个」「上面说的」等指代），"
            "原样返回；\n"
            "3. **禁止引入对话历史里没有的信息**；\n"
            "4. 保持原问题的语言（中文问中文答）。\n\n"
            f"对话历史：\n{transcript}\n\n"
            f"当前问题：{query}\n\n"
            "改写后的问题："
        )

        started = time.perf_counter()
        try:
            response = await self._llm.complete(
                [LLMMessage(role=LLMRole.USER, content=prompt)],
                temperature=0.0,
                max_tokens=200,
            )
        except Exception as exc:
            # 改写失败不阻断问答：用原问题检索（召回可能差，但至少能跑）
            log.warning(
                "standalone_rewrite_failed",
                error_type=type(exc).__name__,
                note="回退到原问题检索",
            )
            return ""

        # 改写也产生成本，必须记账（否则账不平）
        await self._record_cost(
            tenant_id=tenant_id,
            user_id=None,
            conversation_id=None,
            message_id=None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            task_type=ModelTaskType.QUERY_REWRITE,
        )

        rewritten = response.content.strip().strip('"').strip("「」").strip()
        # 防御：改写结果为空或过长都视为失败（回退原问题）
        if not rewritten or len(rewritten) > len(query) * 4 + 100:
            return ""
        return rewritten

    async def _judge_answerability(
        self, *, tenant_id: uuid.UUID, query: str, materials: list[Material]
    ) -> Any:
        """用 LLM 判断"材料能否回答"（A11 的拒答机制）。"""
        prompt = build_answerability_prompt(query=query, materials=[m.content for m in materials])
        started = time.perf_counter()
        # 独立 span：拒答是用户可感知的行为，阀值与误判必须能在 trace 里
        # 和"检索命中了几条"放在一起看，否则无法判断是"没检索到"还是"检到但判错"。
        span_cm = (
            self._tracer.aspan("answerability", material_count=len(materials))
            if self._tracer is not None
            else nullcontext()
        )
        async with span_cm as span:
            try:
                response = await self._llm.complete(
                    [LLMMessage(role=LLMRole.USER, content=prompt)],
                    temperature=0.0,
                    json_mode=True,
                    max_tokens=200,
                )
            except Exception as exc:
                # 判断失败不能拦死问答：保守地"往下走"（让生成模型自己判断），
                # 但记录日志——拒答机制失效是重要信号。
                log.warning(
                    "answerability_check_failed",
                    error_type=type(exc).__name__,
                )
                if self._tracer is not None:
                    self._tracer.annotate(span, error_type=type(exc).__name__)
                return None

            await self._record_cost(
                tenant_id=tenant_id,
                user_id=None,
                conversation_id=None,
                message_id=None,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
                task_type=ModelTaskType.CHAT,
            )
            result = parse_self_check_output(response.content)
            if self._tracer is not None:
                self._tracer.annotate(
                    span,
                    # parsed=False 表示模型输出不是合法 JSON（调用失败与此不同）
                    parsed=bool(result.raw_output),
                    can_answer=result.can_answer if result.raw_output else None,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            return result

    async def _judge_support(
        self, *, tenant_id: uuid.UUID, query: str, answer: str, materials: list[Material]
    ) -> SelfCheckResult | None:
        """生成后判断"答案中的事实是否都有材料依据"。

        与 `_judge_answerability` 的区别（AGENTS.md 4.14）：
        - answerability（生成前）→ 决定拒答；
        - support（生成后）→ 拦幻觉，**不参与拒答决策**。

        失败时返回 None 而不是"判为不支撑"：自省服务抖动不该
        让所有答案都被标记为幻觉（那会让指标与告警全部失真）。
        调用方区分"None"（没跑 / 失败）与"can_answer=False"（真的不支撑）。
        """
        prompt = build_self_check_prompt(
            query=query,
            answer=answer,
            materials=[m.content for m in materials],
        )
        started = time.perf_counter()
        # 与 `answerability` 分开埋点：两个判断的误判代价不同
        # （前者误判=少答，后者误判=幻觉被标/漏标），混在一个 span 里无法分开评估。
        span_cm = (
            self._tracer.aspan("support_check", material_count=len(materials))
            if self._tracer is not None
            else nullcontext()
        )
        async with span_cm as span:
            try:
                response = await self._llm.complete(
                    [LLMMessage(role=LLMRole.USER, content=prompt)],
                    temperature=0.0,
                    json_mode=True,
                    max_tokens=300,
                )
            except Exception as exc:
                log.warning(
                    "support_check_failed",
                    error_type=type(exc).__name__,
                    note="自省调用失败，本次不标记答案",
                )
                if self._tracer is not None:
                    self._tracer.annotate(span, error_type=type(exc).__name__)
                return None

            await self._record_cost(
                tenant_id=tenant_id,
                user_id=None,
                conversation_id=None,
                message_id=None,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
                task_type=ModelTaskType.EVALUATION,
            )
            result = parse_self_check_output(response.content)
            if self._tracer is not None:
                self._tracer.annotate(
                    span,
                    parsed=bool(result.raw_output),
                    supported=result.can_answer if result.raw_output else None,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            return result

    def _retrieval_strategy(self) -> str:
        """指标/埋点里的策略名（低基数，可安全做 label）。"""
        base = "hybrid" if self._config.use_keyword else "vector"
        if self._config.use_rerank and self._reranker is not None:
            return f"{base}+rerank"
        return base

    def _record_self_check_metric(self, *, result: str) -> None:
        """记录自省结果（supported / unsupported / error）。"""
        metrics = self._metrics
        if metrics is None:
            metrics = getattr(self._settings, "_metrics", None)
        if metrics is None:
            return
        from rag_core.metrics import record_self_check

        record_self_check(metrics, result=result)

    async def _mark_interrupted(
        self, *, assistant_id: uuid.UUID, content: str, started: float
    ) -> None:
        await self._messages.mark_failed(
            message_id=assistant_id,
            status=MessageStatus.INTERRUPTED,
            content=content,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        log.info(
            "chat_interrupted",
            message_id=str(assistant_id),
            partial_chars=len(content),
            note="客户端断连，已保留部分内容",
        )

    async def _record_cost(
        self,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        message_id: uuid.UUID | None,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        task_type: ModelTaskType = ModelTaskType.CHAT,
    ) -> None:
        """把这次 LLM 调用写进 `model_calls`（对账的唯一真相）。"""
        if input_tokens == 0 and output_tokens == 0:
            return
        price_in, price_cached, price_out = self._pricing_for(self._llm.name)
        await self._model_calls.record(
            tenant_id=tenant_id,
            task_type=task_type,
            provider=self._llm.name,
            model=self._llm.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unit_price_input=price_in,
            unit_price_cached=price_cached,
            unit_price_output=price_out,
            latency_ms=latency_ms,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            # ## 为什么这两列必须填
            #
            # 只有接口返回的 request_id，排障时能从日志找到 HTTP 请求，
            # 但找不到"这次调用在 trace 里的那个 LLM span"；
            # 只有 trace_id，又看不到对应的访问日志。
            # 实测初版这两列全空（count(request_id)=count(trace_id)=0），
            # 于是"成本涨了但不知道是哪一步"只能翻 Langfuse 手动对时间。
            request_id=_current_request_id(),
            trace_id=self._tracer.current_trace_id() if self._tracer is not None else None,
        )

    def _pricing_for(self, provider: str) -> tuple[Any, Any, Any]:
        """取当前时刻适用的单价快照 `(input, cached_input, output)`。

        ## 必须用 `price_at()` 而不是直接读字段（实测踩到的坑）

        `ModelPricing` 有 **peak / off_peak 两档**（DeepSeek 峰时价是
        谷时的 2 倍）。直接读 `.input` 会 AttributeError（字段在
        `TokenPricing` 上，不在 `ModelPricing` 上）。

        正确做法：`pricing.price_at(now)` → `TokenPricing`，再取三个单价。

        ## 缓存单价必须一起取

        `input_cached` 通常是常规输入的 1/30（DeepSeek 实测 0.022 vs 0.66）。
        不取它会让命中 prompt cache 的调用被高估 30 倍——
        A13 的对账会因为"记录成本远大于重算成本"而报错。

        取不到时返回 (None, None, None)：仓储会记成本 0 但保留 token
        用量，等补上定价后可重算历史（A3 的设计）。
        """
        from datetime import UTC, datetime

        model = self._llm.model_name
        try:
            pricing = self._settings.llm.pricing_for(model)
            if pricing is not None:
                tier = pricing.price_at(datetime.now(UTC))
                return tier.input, tier.input_cached, tier.output
        except Exception as exc:
            # 取不到单价不阻断问答，但要留痕：成本会记 0，
            # 对账时会出现"有调用无成本"的差异，需要能追到这里。
            log.warning(
                "pricing_lookup_failed",
                model=model,
                error_type=type(exc).__name__,
                error=str(exc)[:100],
            )
        return None, None, None

    async def _accumulate_cost(
        self, *, conversation_id: uuid.UUID, input_tokens: int, output_tokens: int
    ) -> float:
        """算出本次成本（用于消息与会话的展示字段）。"""
        price_in, _price_cached, price_out = self._pricing_for(self._llm.name)
        if price_in is None and price_out is None:
            return 0.0
        from decimal import Decimal

        total = Decimal("0")
        if price_in is not None:
            total += Decimal(input_tokens) / Decimal(1_000_000) * price_in
        if price_out is not None:
            total += Decimal(output_tokens) / Decimal(1_000_000) * price_out
        return float(total)


# ---------------------------------------------------------------- 辅助


def _trim_history(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """按 token 预算裁剪历史（保留最近的）。

    ## 为什么从**最旧**开始丢

    多轮对话里最近的几轮才与当前问题相关。20 轮前的"你好"占着预算
    却没有信息量。

    ## 为什么成对丢弃（不是逐条）

    `user`/`assistant` 必须成对出现——丢掉一条 `user` 只留 `assistant`
    会让模型看到"凭空出现的回答"。所以从头部开始，每次丢两条。
    """
    trimmed = list(history)
    total_chars = sum(len(content) for _role, content in trimmed)

    while trimmed and total_chars > MAX_HISTORY_CHARS:
        # 成对丢弃（2 条 = 1 轮）
        removed = trimmed[:2]
        trimmed = trimmed[2:]
        total_chars -= sum(len(content) for _role, content in removed)

    return trimmed


def _make_title(query: str) -> str:
    """用首轮问题生成会话标题（取前 30 字）。"""
    cleaned = " ".join(query.split())
    return cleaned[:30] or "新会话"


def _as_chunks(materials: list[Material]) -> list[RetrievedChunk]:
    """`Material` → `RetrievedChunk`（提示词构造需要后者）。"""
    return [
        RetrievedChunk(
            chunk_id=str(m.chunk_id),
            content=m.content,
            score=m.relevance_score or 0.0,
        )
        for m in materials
    ]


def _extract_usage(usage_holder: list[Any]) -> tuple[int, int]:
    """从 usage 回调收集器里取 (input_tokens, output_tokens)。

    回调是 `stream(usage_callback=...)` 在流结束时触发的
    （见 `LangChainLLMProvider.stream`）。取不到时返回 (0, 0) 并记日志
    ——**不静默**：成本记 0 会让账不平，必须可观测。
    """
    if not usage_holder:
        log.warning(
            "stream_usage_missing",
            note="流式未返回 usage，本条消息成本记 0（对账会显示差异）",
        )
        return (0, 0)
    usage = usage_holder[-1]
    return (int(getattr(usage, "input_tokens", 0)), int(getattr(usage, "output_tokens", 0)))


def _error_code_of(exc: BaseException) -> str:
    if isinstance(exc, AppError):
        return exc.code_name
    mapping = {
        "LLMProviderError": "PROVIDER_UNAVAILABLE",
        "TimeoutError": "PROVIDER_TIMEOUT",
        "ConnectionError": "PROVIDER_UNAVAILABLE",
    }
    return mapping.get(type(exc).__name__, f"UNHANDLED_{type(exc).__name__}")


def _current_request_id() -> str | None:
    """取当前请求的 request_id（由 `RequestContextMiddleware` 绑进 contextvars）。

    为什么不在函数签名里一路传：`_record_cost` 被改写/判断/生成四处调用，
    多一个参数就要四处加，而它与业务语义无关，只是日志关联。
    取不到（worker/脚本/单测）时返回 None，不编造。
    """
    try:
        from structlog.contextvars import get_contextvars

        value = get_contextvars().get("request_id")
    except Exception:  # pragma: no cover - structlog 内部异常
        return None
    return str(value) if value else None


def _safe_message(exc: Exception) -> str:
    """安全的错误描述（脱敏 + 截断）。"""
    from rag_core.pii import redact_text

    return redact_text(str(exc)[:200])


__all__ = [
    "MAX_HISTORY_CHARS",
    "MAX_HISTORY_MESSAGES",
    "ChatEvent",
    "ChatOutcome",
    "ChatService",
]
