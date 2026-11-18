"""任务相关的响应实体与映射。

契约形状来自 `rag_contracts.api.tasks`；本模块只做 re-export 与映射。
"""

from __future__ import annotations

from rag_contracts.api.tasks import RetryTaskResponse, TaskItem
from rag_contracts.enums import TaskStatus, TaskType
from rag_db.models.task import Task


def to_task_item(task: Task) -> TaskItem:
    """ORM → 用户面任务项。

    `task_type` / `status` 存的是 String 列，读回来是 `str`。
    直接传给 `TaskItem` 时 Pydantic 会按枚举校验（值合法就转换），
    非法值会抛 ValidationError → 500。用 `TaskType(...)` 显式转换
    能在数据脏掉时给出可定位的错误（哪个任务、什么值）。
    """
    return TaskItem(
        id=task.id,
        task_type=TaskType(str(task.task_type)),
        status=TaskStatus(str(task.status)),
        progress=task.progress,
        retry_count=task.retry_count,
        max_retries=task.max_retries,
        document_id=task.document_id,
        knowledge_base_id=task.knowledge_base_id,
        error_code=task.error_code,
        error_message=task.error_message,
        queued_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )


def to_retry_response(
    *, task_id: object, original_task_id: object, poll_url: str
) -> RetryTaskResponse:
    """死信重试结果 → 202 响应实体。"""
    import uuid

    return RetryTaskResponse(
        task_id=uuid.UUID(str(task_id)),
        original_task_id=uuid.UUID(str(original_task_id)),
        status=TaskStatus.PENDING,
        poll_url=poll_url,
    )


__all__ = ["RetryTaskResponse", "TaskItem", "to_retry_response", "to_task_item"]
