"""审计查询接口（管理面）。

## 谁能查

- `auditor`：安全审计员，**只读**，能查所有租户的审计；
- `super_admin`：平台超管，能查所有；
- 租户 `owner`/`admin`：只能查**自己租户**的审计。

`docs/PROMPT.md` 要求"管理端自身分权：超管 / 平台运营 / 知识管理员 / 审计员"。

## 为什么审计查询本身也要审计

查审计日志是**敏感操作**：它能暴露"谁在什么时候看了什么文档"。
如果查审计不被记录，内部人员可以悄悄窥探他人行为。

所以每次审计查询都会写一条 `ADMIN_AUDIT_QUERY`——"查审计要留痕"
是合规硬要求，不能省。

## 业务逻辑的位置

权限判定、过滤参数校验、查询留痕、ORM → 契约映射都在
`rag_api.services.audit.AuditQueryService`，本模块只做 HTTP 语义。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import get_db
from rag_api.deps.pagination import page_query
from rag_api.openapi import ADMIN_ENDPOINT
from rag_api.permissions import PermissionContext, get_permission_context
from rag_api.schemas.audit import AuditQueryResponse, to_response
from rag_api.schemas.common import PageRequest
from rag_api.services.audit import AuditQueryService

router = APIRouter(prefix="/admin/audit", tags=["admin"])


@router.get(
    "",
    response_model=AuditQueryResponse,
    summary="查询审计日志",
    description=(
        "支持按动作、目标类型、目标 ID、结果过滤。\n\n"
        "**权限**：`auditor`/`super_admin` 可查全部租户；"
        "租户 `owner`/`admin` 只能查自己租户。\n\n"
        "**查询本身会被审计**：查看他人行为记录是敏感操作。\n\n"
        "分页：`total` 默认不返回（审计表很大，COUNT(*) 很慢）。"
    ),
    responses=ADMIN_ENDPOINT,
)
async def query_audit_logs(
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
    page: PageRequest = Depends(page_query),
    action: str | None = Query(default=None, description="动作类型过滤"),
    target_type: str | None = Query(default=None, description="目标对象类型过滤"),
    target_id: uuid.UUID | None = Query(default=None, description="目标对象 ID 过滤"),
    only_denied: bool = Query(default=False, description="只看被拒绝的操作"),
    since: datetime | None = Query(default=None, description="起始时间（含）"),
    until: datetime | None = Query(default=None, description="结束时间（不含）"),
) -> AuditQueryResponse:
    records = await AuditQueryService(session).query(
        ctx=ctx,
        limit=page.limit,
        action=action,
        target_type=target_type,
        target_id=target_id,
        only_denied=only_denied,
        since=since,
        until=until,
    )
    return to_response(records, limit=page.limit, offset=page.offset)
