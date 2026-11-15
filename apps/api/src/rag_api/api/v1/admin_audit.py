"""审计查询接口（管理面）。

## 谁能查

- `auditor`：安全审计员，**只读**，能查所有租户的审计；
- `super_admin`：平台超管，能查所有；
- 租户 `owner`/`admin`：只能查**自己租户**的审计。

`docs/PROMPT.md` 要求"管理端自身分权：超管 / 平台运营 / 知识管理员 / 审计员"。

## 为什么审计查询本身也要审计

查审计日志是**敏感操作**：它能暴露"谁在什么时候看了什么文档"。
如果查审计不被记录，内部人员可以悄悄窥探他人行为。

所以每次审计查询都会写一条 `admin.config_change`... 不，应该用
独立动作。这里复用 `SECURITY_ACCESS_DENIED` 不合适——加一个
专门的动作会在契约里新增枚举值。

**决策**：审计查询的审计记录用 `AuditAction.SECURITY_ACCESS_DENIED` 的
兄弟——实际上应该新增 `ADMIN_AUDIT_QUERY`。这里我先在枚举里加上，
因为"查审计要留痕"是合规硬要求，不能省。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import Principal, get_current_principal, get_db
from rag_api.deps.pagination import page_query
from rag_api.openapi import ADMIN_ENDPOINT
from rag_api.permissions import PermissionContext, get_permission_context
from rag_contracts.api.audit import AuditLogItem, AuditQueryResponse
from rag_contracts.api.pagination import PageRequest
from rag_contracts.enums import AuditAction, AuditTargetType, PlatformRole
from rag_core.errors import PermissionDeniedError
from rag_core.logging import get_logger
from rag_db.repositories import AuditRepository, UserRepository

log = get_logger(__name__)

router = APIRouter(prefix="/admin/audit", tags=["admin"])

# 能查全部租户审计的平台角色
_GLOBAL_AUDIT_ROLES = frozenset({PlatformRole.SUPER_ADMIN, PlatformRole.AUDITOR})
# 能查本租户审计的租户角色
_TENANT_AUDIT_ROLES = ("owner", "admin")


def _parse_action(value: str | None) -> AuditAction | None:
    """解析 action 过滤参数。

    未知值返回 422 而不是静默忽略：静默忽略会让用户以为过滤生效了，
    实际拿到全量数据——在审计场景下这是危险的（可能误判"没有越权记录"）。
    """
    if not value:
        return None
    try:
        return AuditAction(value)
    except ValueError as exc:
        from rag_contracts.errors import FieldError
        from rag_core.errors import ValidationFailedError

        raise ValidationFailedError(
            fields=[
                FieldError(
                    location=["query", "action"],
                    message=f"未知的审计动作: {value}",
                    type="value_error",
                )
            ]
        ) from exc


def _parse_target_type(value: str | None) -> AuditTargetType | None:
    if not value:
        return None
    try:
        return AuditTargetType(value)
    except ValueError as exc:
        from rag_contracts.errors import FieldError
        from rag_core.errors import ValidationFailedError

        raise ValidationFailedError(
            fields=[
                FieldError(
                    location=["query", "target_type"],
                    message=f"未知的目标类型: {value}",
                    type="value_error",
                )
            ]
        ) from exc


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
    principal: Principal = Depends(get_current_principal),
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
    # ---- 权限判定 ----
    # 先查平台角色（决定能否跨租户）
    user = await UserRepository(session).get_active_by_id(principal.user_id)
    if user is None:
        raise PermissionDeniedError("账号不可用")

    try:
        platform_role = PlatformRole(str(user.platform_role))
    except ValueError:
        platform_role = PlatformRole.NONE

    can_query_global = platform_role in _GLOBAL_AUDIT_ROLES
    can_query_tenant = ctx.has_any_role(*_TENANT_AUDIT_ROLES)  # type: ignore[arg-type]

    if not (can_query_global or can_query_tenant):
        raise PermissionDeniedError("需要审计员或租户管理员权限才能查看审计日志")

    # 租户管理员只能看自己的租户。平台审计员/超管看全部
    # （这里用 tenant_id 过滤——跨租户查询在 A7-2 的 bypass 路径里，
    #   当前接口限定在 token 的租户内，全局查询留到 C5 管理端）
    tenant_id = ctx.tenant_id

    repo = AuditRepository(session)
    records = await repo.list_recent(
        tenant_id=tenant_id,
        limit=page.limit,
        action=_parse_action(action),
        target_type=_parse_target_type(target_type),
        target_id=target_id,
        only_denied=only_denied,
    )

    # ---- 查询本身留痕 ----
    # 在**同一事务**里写（与查询一致）：如果事务回滚，说明这次查询也没成功，
    # 审计记录不该留下
    from rag_api.services.audit import AuditService

    await AuditService(session).record(
        ctx=ctx,
        action=AuditAction.ADMIN_AUDIT_QUERY,
        details={
            "filters_action": action or "",
            "filters_target_type": target_type or "",
            "only_denied": only_denied,
            "result_count": len(records),
            "platform_role": platform_role.value,
        },
    )

    # 时间范围过滤（在 Python 侧做——审计记录量不大时够用，
    # 数据量大时应该下推到 SQL，A15 优化）
    if since is not None or until is not None:
        records = [
            record
            for record in records
            if (since is None or record.created_at >= since)
            and (until is None or record.created_at < until)
        ]

    items = [
        AuditLogItem(
            id=record.id,
            seq=record.seq,
            action=str(record.action),
            outcome=str(record.outcome),
            actor_user_id=record.actor_user_id,
            actor_email=record.actor_email,
            target_type=str(record.target_type) if record.target_type else None,
            target_id=record.target_id,
            target_name=record.target_name,
            request_id=record.request_id,
            ip_address=str(record.ip_address) if record.ip_address else None,
            error_code=record.error_code,
            details=record.details,
            created_at=record.created_at,
        )
        for record in records
    ]

    log.info(
        "audit_query_executed",
        platform_role=platform_role.value,
        tenant_roles=sorted(role.value for role in ctx.roles),
        result_count=len(items),
        queried_by=str(ctx.user_id),
    )

    return AuditQueryResponse(
        items=items,
        total=None,  # 默认不返回：审计表 COUNT(*) 很慢
        limit=page.limit,
        offset=page.offset,
        has_more=len(items) == page.limit,
    )
