"""任务进度查询路由（用户面）。

## 这个文件补的是一条断链

`POST /documents` 返回 `202 {task_id}`，`TaskAccepted.poll_url` 的契约
文档写着"前端靠它轮询 `GET /api/v1/tasks/{id}`"。但**端点从未存在**：

    前端拿到 task_id → GET /api/v1/tasks/{id} → 404

表现是"上传后进度条永远不动"——用户以为卡住了，其实是没人告诉他
"该问哪里"。这类缺口比缺功能更难发现：接口文档、响应契约、
前端代码全都对，只是中间那一步不存在。

## 为什么任务查询在用户面而不是管理面

任务由用户自己的操作触发。进度查询按 `tenant_id` 过滤（跨租户 404），
且**不返回 payload**（里面是内部 ID 与 handler 入参）。
普通用户看自己的上传进度是基本需求；把它放进管理面会让
"上传后看进度"变成只有管理员才能做的事。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import get_db
from rag_api.openapi import TENANT_ENDPOINT
from rag_api.permissions import PermissionContext, get_permission_context
from rag_api.schemas.tasks import TaskItem, to_task_item
from rag_core.errors import NotFoundError

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get(
    "/{task_id}",
    response_model=TaskItem,
    summary="查询任务进度",
    description=(
        "返回任务的当前状态与进度（0-100）。\n\n"
        "**`progress=100` 只由 worker 在成功结束时设置**：handler 上报上限 99，"
        "所以看到 100 就说明任务真的成功了，不是「快好了」。\n\n"
        "任务状态与文档状态分离：一次文档处理会产生多个任务"
        "（解析→切分→向量化），`documents.status` 是业务状态，"
        "本接口返回的是**某一次执行**的状态。\n\n"
        "跨租户或不存在都返回 404（不泄漏任务是否存在）。"
    ),
    responses=TENANT_ENDPOINT,
)
async def get_task(
    task_id: uuid.UUID,
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> TaskItem:
    from rag_db.repositories import TaskRepository

    task = await TaskRepository(session).get_for_tenant(tenant_id=ctx.tenant_id, task_id=task_id)
    if task is None:
        raise NotFoundError("任务不存在")
    return to_task_item(task)


@router.post(
    "/{task_id}/cancel",
    response_model=TaskItem,
    summary="取消任务",
    description=(
        "取消 pending / running 任务。\n\n"
        "**running 的取消是「协作式」的**：worker 已经认领并开始执行，"
        "本操作只把数据库状态置为 cancelled。worker 在下一个进度回写点"
        "会发现认领已失效（`update_progress` 返回 False）并停止。\n\n"
        "已结束（succeeded / failed / dead）的任务返回 409。"
    ),
    responses=TENANT_ENDPOINT,
)
async def cancel_task(
    task_id: uuid.UUID,
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> TaskItem:
    from rag_core.errors import ConflictError
    from rag_db.repositories import TaskRepository

    repo = TaskRepository(session)
    task = await repo.get_for_tenant(tenant_id=ctx.tenant_id, task_id=task_id)
    if task is None:
        raise NotFoundError("任务不存在")

    cancelled = await repo.cancel(tenant_id=ctx.tenant_id, task_id=task_id)
    if not cancelled:
        raise ConflictError(
            f"任务当前状态不可取消（{task.status}）",
            details={"status": str(task.status)},
        )

    # 重新读一次拿到最新状态：cancel 只返回 bool，
    # 直接在内存里改 ORM 对象不会触发刷新（session 里还是旧值）
    await session.refresh(task)
    return to_task_item(task)


__all__ = ["router"]
