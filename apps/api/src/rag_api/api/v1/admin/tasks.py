"""管理面：任务台账与死信重试。

## 为什么管理端需要独立的任务台账

用户面 `/tasks/{id}` 只能查单个任务（自己触发的）。运营要回答的是
另一类问题：

- "现在积压了多少任务？"（`status=pending` 计数）
- "那些死信是什么原因？"（跨文档、跨用户看失败原因）
- "这个文档为什么卡住？"（按 `document_id` 过滤全部任务）

所以本模块提供**列表** + **按维度过滤**，且带 `tenant_id` /
`locked_by`（排障必需：哪个 worker 认领了、卡了多久）。

## 重试死信为什么是「新建任务」

复用旧行会抹掉 `retry_count` 与 `error_message` 历史——而管理端恰恰
要看「失败过几次、每次为什么」。所以重试插入一条新任务，旧行保持
`dead` 供追溯。幂等键带随机后缀，因为人工重试是显式动作，
同一死信可以被重试多次（每次都是独立的执行记录）。

## 为什么重试走 `OPERATE` 能力而不是 `ADMIN`

重试是运维动作（「再跑一次」），不改变权限或配置。
平台运营（`platform_operator`）应该能做；而审计员不能。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps.pagination import PageQuery, page_query
from rag_api.openapi import ADMIN_ENDPOINT
from rag_api.permissions import (
    AdminCapability,
    AdminContext,
    get_admin_context,
    get_admin_db,
)
from rag_api.permissions.admin import require_admin_capability
from rag_api.schemas.admin import to_admin_task
from rag_api.schemas.common import Page
from rag_api.schemas.tasks import RetryTaskResponse
from rag_contracts.api.admin import AdminTaskItem
from rag_contracts.api.pagination import PageMeta
from rag_contracts.enums import AuditAction, AuditOutcome, AuditTargetType, TaskStatus
from rag_core.errors import ConflictError, NotFoundError
from rag_db.repositories import AuditRepository, TaskRepository

router = APIRouter(prefix="/tasks", tags=["admin"])


@router.get(
    "",
    response_model=Page[AdminTaskItem],
    summary="任务台账",
    description=(
        "按状态 / 类型 / 文档过滤任务，用于排查积压与失败。\n\n"
        "**`total` 默认不返回**：任务表在活跃系统上增长很快，"
        "`COUNT(*)` 很慢。管理端需要「共 N 条」时显式传 `include_total=true`。\n\n"
        "`status=dead` 等价于死信列表。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.READ))],
    responses=ADMIN_ENDPOINT,
)
async def list_tasks(
    ctx: AdminContext = Depends(get_admin_context),
    session: AsyncSession = Depends(get_admin_db),
    page: PageQuery = Depends(page_query),
    status: str | None = Query(
        default=None, description="pending/running/succeeded/failed/dead/cancelled"
    ),
    task_type: str | None = Query(default=None, description="parse_document/chunk_document/…"),
    document_id: uuid.UUID | None = Query(default=None, description="按文档过滤"),
) -> Page[AdminTaskItem]:
    rows, total = await TaskRepository(session).list_filtered(
        tenant_id=ctx.data_tenant_id,
        limit=page.limit,
        offset=page.offset,
        status=status,
        task_type=task_type,
        document_id=document_id,
        include_total=page.include_total,
    )
    items = [to_admin_task(row) for row in rows]
    return Page(
        items=items,
        meta=PageMeta(
            limit=page.limit,
            offset=page.offset,
            total=total,
            has_more=len(items) == page.limit,
        ),
    )


@router.post(
    "/{task_id}/retry",
    response_model=RetryTaskResponse,
    status_code=202,
    summary="重试死信任务",
    description=(
        "为一条 `dead` 任务创建新的执行记录并重新入队。\n\n"
        "**原任务保持 dead 状态**：排障需要「失败过几次、每次为什么」的完整历史，"
        "复用旧行会抹掉它。\n\n"
        "非 dead 状态返回 409（failed 会自动重试，running/pending 本来就在队列里）。\n\n"
        "重试会写审计（`admin.task_retry`），记录谁在何时重试了哪个任务。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.OPERATE))],
    responses=ADMIN_ENDPOINT,
)
async def retry_task(
    task_id: uuid.UUID,
    ctx: AdminContext = Depends(get_admin_context),
    session: AsyncSession = Depends(get_admin_db),
) -> RetryTaskResponse:
    repo = TaskRepository(session)
    task = await repo.get_for_tenant(tenant_id=ctx.data_tenant_id, task_id=task_id)
    if task is None:
        raise NotFoundError("任务不存在")
    if str(task.status) != TaskStatus.DEAD.value:
        raise ConflictError(
            f"只有死信任务可以手动重试（当前状态：{task.status}）",
            details={"status": str(task.status)},
        )

    # 新任务沿用原任务的类型与 payload（payload 是 handler 入参，
    # 原样复用；这里不解析它，避免与 handler 的 schema 耦合）
    new_task, _created = await repo.enqueue(
        tenant_id=ctx.data_tenant_id,
        task_type=str(task.task_type),
        idempotency_key=f"admin-retry:{task.id}:{uuid.uuid4().hex[:8]}",
        payload=dict(task.payload or {}),
        document_id=task.document_id,
        document_version_id=task.document_version_id,
        knowledge_base_id=task.knowledge_base_id,
        max_retries=task.max_retries,
    )

    await AuditRepository(session).record(
        tenant_id=ctx.data_tenant_id,
        action=AuditAction.ADMIN_TASK_RETRY,
        outcome=AuditOutcome.SUCCESS,
        actor_user_id=ctx.user_id,
        target_type=AuditTargetType.TASK,
        target_id=task.id,
        details={
            "new_task_id": str(new_task.id),
            "task_type": str(task.task_type),
            "original_retry_count": task.retry_count,
            "platform_role": ctx.platform_role.value,
        },
    )

    return RetryTaskResponse(
        task_id=new_task.id,
        original_task_id=task.id,
        status=TaskStatus.PENDING,
        poll_url=f"/api/v1/tasks/{new_task.id}",
    )


__all__ = ["router"]
