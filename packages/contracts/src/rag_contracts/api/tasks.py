"""异步任务查询契约。

## 为什么需要这个接口（它是断链，不是新功能）

`TaskAccepted.poll_url` 的文档写着"前端靠它轮询 `GET /api/v1/tasks/{id}`"，
`POST /documents` 也返回 `202 + task_id`。但**这个查询端点从未存在**：

    POST /documents → 202 {task_id}
                        │
                        └─▶ GET /api/v1/tasks/{id}   ← 404

前端拿到 task_id 之后无处可查，只能反复拉文档详情猜进度。
这是"契约写了、实现没有"的典型断链——比缺功能更糟，因为调用方
按文档写代码，联调时才发现。

## 为什么任务查询属于**用户面**

任务由用户自己的操作触发（上传文档、重试）。任务 ID 不可枚举
（UUID），且查询时按租户 + 触发者过滤，跨用户访问返回 404。
把进度查询塞进管理面会让普通用户看不到自己的上传进度。

## 为什么不返回 payload

`tasks.payload` 里是 handler 的入参（含 document_version_id 等内部 ID）。
对用户没有价值，且会扩大攻击面（攻击者从中推断内部结构）。
契约只暴露"这个任务在做什么、做到哪、错在哪"。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from rag_contracts.enums.uploads import TaskStatus, TaskType


class TaskItem(BaseModel):
    """一个任务的状态快照。

    ## `progress=100` 的语义

    只由 worker 的 `complete()` 设置。handler 上报上限 99 —— 否则
    "仍在跑（甚至马上失败）"的任务会显示已完成，用户看到 100 却等不到结果。
    """

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    task_type: TaskType
    status: TaskStatus
    progress: int = Field(ge=0, le=100, description="0-100。100 表示任务已成功结束")
    retry_count: int = Field(ge=0, description="已重试次数")
    max_retries: int = Field(ge=0)
    document_id: uuid.UUID | None = Field(default=None, description="关联文档（如有）")
    knowledge_base_id: uuid.UUID | None = Field(default=None, description="关联知识库（如有）")
    error_code: str | None = Field(default=None, description="失败错误码（仅 failed/dead 有值）")
    error_message: str | None = Field(default=None, description="失败原因（已脱敏，不含文档正文）")
    queued_at: datetime = Field(description="入队时间（= created_at）")
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RetryTaskResponse(BaseModel):
    """死信任务重试的响应（202 语义）。

    ## 为什么重试是"新建任务"而不是"复活旧任务"

    复用同一行会抹掉 `error_message` 与 `retry_count` 历史——而管理端
    恰恰要看"这个任务失败过 3 次、每次为什么"。所以重试插入一条新任务，
    旧任务保持 `dead` 状态供追溯。

    幂等键用 `retry:{task_id}:{uuid4}`：同一死信可以被重试多次，
    每次都是独立的执行记录（人工重试是显式动作，不需要去重）。
    """

    model_config = ConfigDict(frozen=True)

    task_id: uuid.UUID = Field(description="新任务 ID")
    original_task_id: uuid.UUID = Field(description="被重试的原死信任务")
    status: TaskStatus = Field(description="新任务状态，正常为 pending")
    poll_url: str = Field(description="轮询地址，前端可直接 GET")


__all__ = ["RetryTaskResponse", "TaskItem"]
