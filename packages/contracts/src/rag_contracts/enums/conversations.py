"""对话域枚举。

对应模型：`rag_db.models.conversation`（`conversations` / `messages`）。

## 为什么对话域需要独立枚举文件

对话不是"文档的附属物"——它有自己完整的状态与生命周期：

- **会话级**：进行中 / 已归档（归档后只读，可恢复）
- **消息级**：用户在问 / 助手在答 / 答完了 / 出错 / 被拒答
- **反馈级**：点赞 / 点踩（点踩要能带原因，进评估集）

把它们塞进 `documents.py` 会让该文件的语义变成"文档 + 对话"，
而两者除了都带 tenant_id 之外没有任何共同点。

## 与 `TaskStatus` 的关系

不要混用。`TaskStatus` 描述"异步任务执行到哪了"，
`MessageStatus` 描述"这条消息本身处于什么状态"——
一次问答可能产生 3 个任务（检索/生成/自省），而只有 1 条 message。
"""

from __future__ import annotations

from enum import StrEnum


class ConversationStatus(StrEnum):
    """会话状态。"""

    ACTIVE = "active"
    """正常可继续追问。"""

    ARCHIVED = "archived"
    """已归档：**只读**（不能再追加消息），但可读取历史与恢复。

    为什么用归档而不是删除：企业场景里"员工离职前的对话记录"
    需要保留供审计；用户主动"删除会话"应走软删（`deleted_at`），
    而不是物理删除——审计链需要它存在。
    """


class MessageRole(StrEnum):
    """消息角色。

    与 `LLMRole`（provider 层）分开：这里是**业务角色**，
    将来可能要加 `tool`（工具返回）等业务概念，而 provider 层
    只认识 system/user/assistant 三种。
    """

    USER = "user"
    ASSISTANT = "assistant"


class MessageStatus(StrEnum):
    """消息状态。

    ## 为什么助手消息需要 5 种状态

    流式回答不是原子的：可能生成到一半断连、可能被自省判为无依据、
    可能供应商报错。用 `success`/`failed` 两态无法表达
    "用户看到了部分内容但没写完"——而那恰恰是最需要记录的情况
    （前端要显示"回答中断"，且这条消息不该进评估集）。
    """

    PENDING = "pending"
    """已创建、还没有内容（流式开始前的占位）。"""

    STREAMING = "streaming"
    """正在流式生成。

    **这个状态可能因客户端断连而滞留**：进程崩溃/连接断开时
    没有机会把状态改成 failed。清理任务按 `created_at` 超时把它
    标成 `interrupted`（见 `INTERRUPTED`）。
    """

    COMPLETED = "completed"
    """正常生成完毕。"""

    INTERRUPTED = "interrupted"
    """生成中途停止（客户端断连 / 超时 / 进程重启）。

    **与 FAILED 的区别**：这里有部分内容（可能有用），
    FAILED 通常没有内容可展示。
    """

    FAILED = "failed"
    """生成失败，内容为空或不完整。

    **与 INTERRUPTED 的区别**：这是明确收到了错误（供应商 5xx、
    内容审核拦截），不是"莫名其妙停了"。
    """


class RefusalReason(StrEnum):
    """拒答原因。

    `docs/PROMPT.md` 要求"无依据拒答"。拒答必须记录原因，
    否则无法回答"为什么这个月拒答率涨了 3 倍"。
    """

    NO_MATERIALS = "no_materials"
    """检索没有返回任何材料。"""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    """有材料，但自省判定不足以回答（A11 的 answerability 判断）。"""

    PROMPT_INJECTION_SUSPECTED = "prompt_injection_suspected"
    """输出侧检测到疑似注入成功（复述了系统提示）。"""


class FeedbackRating(StrEnum):
    """用户反馈。"""

    LIKE = "like"
    DISLIKE = "dislike"


class FeedbackReason(StrEnum):
    """点踩原因（可选，用于构建改进数据集）。

    固定枚举而不是自由文本：自由文本无法聚合统计，
    而"哪类问题最常被踩"是优化 prompt 的关键输入。
    """

    NOT_RELEVANT = "not_relevant"
    """答非所问。"""

    WRONG_FACT = "wrong_fact"
    """事实错误。"""

    INCOMPLETE = "incomplete"
    """回答不完整。"""

    NO_CITATION = "no_citation"
    """没有给出引用，或引用无法支撑结论。"""

    TOO_VERBOSE = "too_verbose"
    """太啰嗦。"""

    OTHER = "other"


__all__ = [
    "ConversationStatus",
    "FeedbackRating",
    "FeedbackReason",
    "MessageRole",
    "MessageStatus",
    "RefusalReason",
]
