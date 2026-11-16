"""对话接口契约。

## 三个端点的形状

| 端点 | 请求 | 响应 |
|---|---|---|
| `POST /chat` | `ChatRequest` | **SSE 事件流**（不是 JSON） |
| `GET /conversations` | 分页参数 | `Page[ConversationItem]` |
| `GET /conversations/{id}/messages` | 分页参数 | `Page[MessageItem]` |

## 为什么 chat 用 SSE 而不是 WebSocket

| | SSE | WebSocket |
|---|---|---|
| 协议 | 纯 HTTP | 需升级握手 |
| 代理/网关 | 原生支持 | 常被拦或需额外配置 |
| 断线重连 | 浏览器原生 `EventSource` 支持 | 要自己实现 |
| 双向通信 | ❌（本场景不需要） | ✅ |

问答是"一问一答"的单向流，SSE 足够。用 WebSocket 会引入
握手、心跳、重连、状态同步的全部复杂度，而换不来任何东西。

**代价**：SSE 只能单向。用户想"中途打断"必须发新请求（DELETE 或
新 POST）——但主流做法是客户端直接断开连接，服务端感知后取消上游调用。

## SSE 事件类型

流里不是纯文本 delta，而是结构化事件：

    event: retrieval         ← 检索完成（只知道"找到几段"，还不知道会引用哪几条）
    data: {"sources_found": 5, "refused": false}

    event: delta             ← 逐 token 增量
    data: {"text": "住宿"}

    event: done              ← 结束（带最终消息 ID、用量、**真正的引用**）
    data: {"message_id": "...", "citations": [...], "usage": {...}}

    event: error             ← 出错（前端显示重试按钮）
    data: {"code": 80001, "code_name": "...", "message": "..."}

**为什么先发 retrieval 再发 delta**：LLM 首字延迟常 1~3 秒。检索一完成就
告诉前端"找到 5 段资料"，等待期就不是空白页了。

### `retrieval` 事件**不带 citations**（这是修过的设计错误）

初版把检索到的**全部**材料当成 `Citation` 列表塞进 `retrieval` 事件，
于是前端在答案还没生成时就显示「来源 1 2 3 4 5」。

三个后果，每个都是错的：

**编号对不上**：事件里的 `index` 按材料顺序编（1..N），而答案里的 `[1]`
是模型自己写的。模型先引用第 2 条材料时，答案写 `[1]`，引用卡片 1 却
指向第 1 条材料——看的人会跳到错误的原文。

**数量对不上**：检索 5 条 ≠ 引用 5 条。答案只用了 1 条时，另外 4 条是
噪音，且会让人误以为答案有 5 个依据。

**时机不对**：生成前根本不知道模型会引用什么，此时给出的"来源"只是
候选，不是依据。

正确做法：**引用在生成结束后按答案里的 `[n]` 标记组装**
（`rag_adapters.retrieval.citations_builder.assemble_citations`）：
按答案首次出现顺序编号、去重、越界丢弃、片段级定位、越权文档二次校验。

代价是生成期间看不到引用。这个取舍是明确的：**引用是"依据"，依据必须是
真的**。宁可等 3 秒，也不要在等待期显示一堆可能对不上的编号。

## 为什么 `refused` 与 `citations 为空` 分开

拒答（材料不足）与"回答了但没有引用"是两件事：
- 拒答 → 前端显示"未能从知识库找到答案" + 建议换个问法；
- 无引用的普通回答 → 正常显示，只是没有引用卡片。

合在一起会让前端无法区分该显示哪种 UI。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_contracts.api.citations import Citation
from rag_contracts.enums.conversations import (
    ConversationStatus,
    FeedbackRating,
    FeedbackReason,
    MessageRole,
    MessageStatus,
    RefusalReason,
)


class ChatRequest(BaseModel):
    """发起一次问答。

    ## 为什么 `conversation_id` 可选

    - 空：新建会话（首轮问答）；
    - 非空：在该会话内追问（服务端取历史做独立化改写）。

    服务端会校验该会话属于当前用户——**跨用户访问返回 404**
    （不泄漏"这个 ID 是否存在"）。
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=4000,
        description="用户问题。上限 4000 字符——超过这个长度的问题本身不合法（应拆成多次提问）",
    )
    conversation_id: uuid.UUID | None = Field(default=None, description="会话 ID。空则新建会话")
    knowledge_base_ids: list[uuid.UUID] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "限定的知识库。空表示租户内全部可见库。上限 20：太多库会让检索范围过宽、召回质量下降"
        ),
    )

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        """去掉首尾空白；全空白视为非法。

        `min_length=1` 拦不住 `"   "`——它在 Python 里长度是 3。
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("问题不能为空")
        return stripped


class ConversationItem(BaseModel):
    """会话列表项。"""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    title: str
    status: ConversationStatus
    knowledge_base_id: uuid.UUID | None = None
    message_count: int = Field(ge=0)
    last_message_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    total_cost_usd: float = Field(ge=0, description="本会话累计成本（展示用）")


class MessageItem(BaseModel):
    """消息项。"""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    conversation_id: uuid.UUID
    sequence: int = Field(ge=1)
    role: MessageRole
    content: str
    status: MessageStatus
    refused: bool = False
    refusal_reason: RefusalReason | None = None
    citations: list[Citation] = Field(
        default_factory=list, description="引用列表（只含本轮检索来源）"
    )
    created_at: datetime
    # 用户反馈（刷新页面后要能回显"已点赞"，否则用户会重复点）
    rating: FeedbackRating | None = Field(
        default=None, description="用户反馈：like / dislike"
    )
    feedback_reason: FeedbackReason | None = Field(
        default=None, description="点踩原因"
    )

    # 用量（仅有助手消息）
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)


class FeedbackRequest(BaseModel):
    """消息反馈请求体。

    ## 为什么必须是请求体模型，不能是三个散装参数

    初版在路由签名里直接写了 `rating: FeedbackRating`、`reason`、`comment`
    三个标量参数。FastAPI 对**标量参数默认按 query 解析**，于是前端按 REST
    惯例 POST 一个 JSON body 时，服务端去 query 里找 `rating` → 422：

        {"fields":[{"location":["query","rating"],"message":"Field required"}]}

    实测现象：前端点赞点了没反应（422 被吞在 `.then` 里），图标永远不亮。

    请求体模型既修正了解析位置，也让契约能导出给前端做类型约束。
    """

    model_config = ConfigDict(extra="forbid")

    rating: FeedbackRating = Field(description="like / dislike")
    reason: FeedbackReason | None = Field(
        default=None,
        description="点踩原因（点踩必填、点赞不需要）。自由文本无法聚合统计",
    )
    comment: str | None = Field(
        default=None,
        max_length=1000,
        description="补充说明。上限 1000 字符——过长的自由文本可能是误粘贴",
    )


class ChatRetrievalPayload(BaseModel):
    """SSE `retrieval` 事件的 payload。

    ## 只有"找到几段"，没有引用

    见模块 docstring 的说明：生成前不知道模型会引用哪几条材料，
    提前发引用会让编号与答案里的 `[n]` 对不上。

    前端用它做两件事：
    1. 从"检索中"切到"生成中"（`sources_found` 出现即表示检索完成）；
    2. 显示"已检索到 N 段资料"，让首 token 的等待期可感知。

    `sources_found == 0` 通常意味着材料不足，紧接着会收到拒答
    （`refused=true` 与一条 `delta` 拒答文案）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sources_found: int = Field(
        ge=0, description="检索命中的材料段数。**不是引用数**——答案可能只用到其中几条"
    )
    refused: bool = Field(
        default=False,
        description="是否已判定拒答（材料不足）。为 true 时不会再有正常生成",
    )


class ChatDonePayload(BaseModel):
    """SSE `done` 事件的 payload。

    ## `citations` 是"答案实际引用的"，不是"检索到的"

    由 `assemble_citations` 在生成结束后按答案里的 `[n]` 标记组装：
    按首次出现顺序连续编号、同一材料去重、越界编号丢弃、
    引用的文档必须通过可见性二次校验。

    **`index` 从 1 连续、不跳号**：前端按 `index` 找引用条目，
    跳号会让它拿到错条目且不报错。
    """

    model_config = ConfigDict(extra="forbid")

    message_id: uuid.UUID
    conversation_id: uuid.UUID
    refused: bool = False
    refusal_reason: RefusalReason | None = None
    citations: list[Citation] = Field(default_factory=list)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    degraded: bool = Field(default=False, description="是否走了降级路径（供应商降级/rerank 挂）")


__all__ = [
    "ChatDonePayload",
    "ChatRequest",
    "ChatRetrievalPayload",
    "ConversationItem",
    "FeedbackRequest",
    "MessageItem",
]
