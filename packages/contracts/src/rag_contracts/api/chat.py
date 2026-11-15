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

    event: retrieval         ← 检索完成（前端可先渲染引用卡片）
    data: {"citations": [...]}

    event: delta             ← 逐 token 增量
    data: {"text": "住宿"}

    event: done              ← 结束（带最终消息 ID 与用量）
    data: {"message_id": "...", "usage": {...}}

    event: error             ← 出错（前端显示重试按钮）
    data: {"code": 80001, "code_name": "...", "message": "..."}

**为什么先发 retrieval 再发 delta**：用户等待首 token 时（LLM 首字延迟
常 1~3 秒），引用卡片已经可以先显示——这让"等待"变得可感知。
只发 delta 的话，用户盯着空白页 3 秒。

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
    # 用量（仅有助手消息）
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)


class ChatDonePayload(BaseModel):
    """SSE `done` 事件的 payload。"""

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
    "ConversationItem",
    "MessageItem",
]
