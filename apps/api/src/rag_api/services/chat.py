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
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from rag_adapters.providers.embedding import EmbeddingProvider, EmbeddingTask
from rag_adapters.providers.llm import LLMMessage, LLMProvider, LLMRole
from rag_adapters.retrieval.citations_builder import Material, assemble_citations
from rag_adapters.retrieval.graph import (
    build_answerability_prompt,
    build_generation_prompt,
)
from rag_adapters.retrieval.selfcheck import (
    REFUSAL_MESSAGE,
    parse_self_check_output,
    should_refuse,
)
from rag_adapters.retrieval.state import RetrievalConfig, RetrievedChunk
from rag_contracts.api.citations import Citation
from rag_contracts.enums.audit import ModelTaskType
from rag_contracts.enums.conversations import (
    MessageRole,
    MessageStatus,
    RefusalReason,
)
from rag_core.errors import AppError, ConflictError, NotFoundError
from rag_core.logging import get_logger
from rag_db.models.conversation import Conversation, Message
from rag_db.repositories import (
    ConversationRepository,
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
        metrics: Any = None,
    ) -> None:
        self._session = session
        self._llm = llm
        self._embedder = embedder
        self._settings = settings
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)
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

    async def list_messages(
        self,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        include_total: bool = False,
    ) -> tuple[list[Message], int | None]:
        # 先校验归属：会话不属于该用户时返回 404
        await self.get_or_create_conversation(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        rows, total = await self._messages.page(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
            include_total=include_total,
        )
        return list(rows), total

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
            query=retrieval_state.rewritten_query or query,
            materials=_as_chunks(materials),
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

        try:
            async for delta in self._llm.stream(
                messages, temperature=0.3, usage_callback=_on_usage
            ):
                collected.append(delta)
                yield ChatEvent("delta", {"text": delta})
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
        citations_result, citation_stats = assemble_citations(
            answer=answer,
            materials=materials,
            visible_document_ids=visible_document_ids,
        )
        citations = citations_result.citations

        # ---- 落库：消息 + 成本 ----
        usage = _extract_usage(usage_holder)
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
            from rag_core.metrics import record_llm_call, record_retrieval

            record_retrieval(
                metrics,
                strategy="chat",
                duration_s=time.perf_counter() - started,
                result_count=len(materials),
            )
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
                "refused": False,
                "refusal_reason": None,
                "citations": [c.model_dump(mode="json") for c in citations],
                "input_tokens": usage[0],
                "output_tokens": usage[1],
                "cost_usd": round(cost_usd, 8),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "degraded": False,
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
        """执行检索，把结果转成 `Material`（含页信息）。

        ## 多轮的关键：先用历史做独立化改写

        "那二线呢？" 单独检索召回为零。必须先用历史消解指代：

            history + "那二线呢？" → "二线城市住宿标准是多少"

        **只有多轮（history 非空）才改写**：首轮问题是完整的，
        改写反而可能引入偏差（且多花一次 LLM 调用）。
        """
        # ---- 独立化改写（多轮必需）----
        retrieval_query = query
        if history and self._config.use_rewrite:
            standalone = await self._standalone_query(
                tenant_id=tenant_id, query=query, history=history
            )
            if standalone:
                retrieval_query = standalone
                log.info(
                    "query_standalone_rewritten",
                    original_length=len(query),
                    standalone_length=len(standalone),
                    history_turns=len(history) // 2,
                )

        query_vector = await self._embedder.embed_one(retrieval_query, task=EmbeddingTask.QUERY)

        # 检索范围：请求指定 > 会话绑定 > 租户内全部
        kb_ids = knowledge_base_ids or (
            [conversation.knowledge_base_id] if conversation.knowledge_base_id else None
        )

        # ---- ACL 过滤条件 ----
        #
        # **这是安全边界，不是优化**。检索返回的是文档正文，不过滤等于
        # 用户能通过提问把无权文档的内容问出来。
        #
        # ## 为什么用 `confidentiality_filter_for_chunks` 而不是
        #    `document_access_filter`
        #
        # 两者语义相同（未删除 + 密级 ≤ 上限 + 库可见），但**生成的 SQL 不同**：
        #
        # - `document_access_filter` 直接引用 `Document.xxx` 列。把它塞进
        #   `select(Chunk).where(...)` 时，SQLAlchemy 会把 `documents` 表加进
        #   FROM 且**没有 join 条件** —— 变成 `FROM chunks, documents` 的
        #   笛卡尔积。实测就是这样：过滤条件对着"所有文档"求值，
        #   而不是"这条 chunk 所属的文档"，SECRET 内容照样漏出来。
        #
        # - `confidentiality_filter_for_chunks` 用
        #   `Chunk.document_id.in_(SELECT documents.id WHERE ...)`，
        #   是纯子查询，不会污染 FROM 子句。**chunk 级查询必须用它。**
        #
        # 这个差异不报错、不告警，只有"逐角色对比检索结果"的测试能发现
        # （见 tests/test_acl_retrieval.py）。
        access_filter: Any = None
        if permission_ctx is not None:
            from rag_api.permissions.acl import confidentiality_filter_for_chunks

            access_filter = confidentiality_filter_for_chunks(permission_ctx)

        hits: list[tuple[Any, float]] = []
        if kb_ids:
            for kb_id in kb_ids:
                version = await self._index_versions.get_active(knowledge_base_id=kb_id)
                if version is None:
                    continue
                page = await self._chunks.search_vector(
                    tenant_id=tenant_id,
                    query_embedding=query_vector,
                    knowledge_base_ids=[kb_id],
                    limit=self._config.vector_limit,
                    index_version_ids=[version.id],
                    access_filter=access_filter,
                )
                hits.extend(page)
            hits.sort(key=lambda item: item[1])
            hits = hits[: self._config.vector_limit]
        else:
            # 未指定库：租户内全部（也不带版本过滤，取全局最新）
            hits = await self._chunks.search_vector(
                tenant_id=tenant_id,
                query_embedding=query_vector,
                limit=self._config.vector_limit,
                access_filter=access_filter,
            )

        materials: list[Material] = []
        for chunk, distance in hits:
            materials.append(
                Material(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=await self._document_title(tenant_id, chunk.document_id),
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    page_no=chunk.page_no,
                    bbox=chunk.bbox,
                    section_path=tuple(chunk.section_path or ()),
                    relevance_score=distance,
                    pages=(),
                )
            )

        # 构造一个轻量 state 供上层取改写结果与耗时
        from rag_adapters.retrieval.state import RetrievalState

        state = RetrievalState(
            query=query,
            history=history,
            # 只有真的改写成功才记，否则 None（表明"用的原问题"）
            rewritten_query=retrieval_query if retrieval_query != query else None,
        )
        state.candidates = [
            RetrievedChunk(
                chunk_id=str(m.chunk_id),
                content=m.content,
                score=m.relevance_score or 0.0,
            )
            for m in materials
        ]
        state.final = state.candidates[: self._config.final_limit]
        final_materials = materials[: self._config.final_limit]

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

    async def _document_title(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> str:
        """取文档标题（引用展示用）。

        每次检索都查一次标题会有 N+1：这里用单条查询批量拿
        （见 `_build_citations` 的实际用法）。为了简单先逐条查，
        但**限制在 final_limit（默认 5）以内**，影响可控。
        """
        from sqlalchemy import text

        row = await self._session.execute(
            text("SELECT title FROM documents WHERE id = :id AND tenant_id = :tid LIMIT 1"),
            {"id": document_id, "tid": tenant_id},
        )
        value = row.scalar_one_or_none()
        return str(value) if value is not None else "未知文档"

    async def _judge_answerability(
        self, *, tenant_id: uuid.UUID, query: str, materials: list[Material]
    ) -> Any:
        """用 LLM 判断"材料能否回答"（A11 的拒答机制）。"""
        prompt = build_answerability_prompt(query=query, materials=[m.content for m in materials])
        started = time.perf_counter()
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
        return parse_self_check_output(response.content)

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


def _extract_usage(holder: list[Any]) -> tuple[int, int]:
    """从 usage 回调收集器里取 (input_tokens, output_tokens)。

    回调是 `stream(usage_callback=...)` 在流结束时触发的
    （见 `LangChainLLMProvider.stream`）。取不到时返回 (0, 0) 并记日志
    ——**不静默**：成本记 0 会让账不平，必须可观测。
    """
    if not holder:
        log.warning(
            "stream_usage_missing",
            note="流式未返回 usage，本条消息成本记 0（对账会显示差异）",
        )
        return (0, 0)
    usage = holder[-1]
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


def _safe_message(exc: BaseException) -> str:
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
