"""管理面：安全审计查询与审计链验证。

## 谁能查

| 角色 | 权限 |
|---|---|
| 平台 `super_admin` | 全部租户 |
| 平台 `auditor` | 全部租户，**只读** |
| 平台 `platform_operator` | **不能查审计**（管运维不等于能看别人的行为） |
| 租户 `owner` / `admin` | 只能查自己租户 |

这条口径由 `AdminCapability.AUDIT` 表达（见 `permissions/admin.py`）：
`platform_operator` 有能力集 `{READ, OPERATE}`，不含 `AUDIT`。
把它放在能力表里而不是每个接口里 if，是为了让「运营能不能查审计」
这问题的答案只有一处。

## 为什么审计查询本身也要审计

查审计日志能暴露「谁在什么时候看了什么文档」。如果查审计不被记录，
内部人员可以悄悄窥探他人行为。所以每次查询写一条 `admin.audit_query`。

## 为什么链验证是独立接口

`AuditRepository.verify_chain()` 重算每条记录的哈希，是 O(n) 的
CPU 操作（n = 该租户记录数）。混进列表接口会让每次翻页都全表重算。
独立接口 + 显式 `limit`，让调用方知道「我在验多少条」，
返回里的 `is_complete` 明确区分「全量验证通过」与"前 N 条通过"。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query

from rag_api.deps import get_database
from rag_api.deps.pagination import page_query
from rag_api.openapi import ADMIN_ENDPOINT
from rag_api.permissions import (
    AdminCapability,
    AdminContext,
    get_admin_context,
)
from rag_api.permissions.admin import require_admin_capability, scoped_admin_session
from rag_api.schemas.admin import to_chain_verification
from rag_api.schemas.audit import AuditQueryResponse, to_response
from rag_api.schemas.common import PageRequest
from rag_contracts.api.admin import AuditChainVerification
from rag_contracts.enums import AuditAction, AuditOutcome, AuditTargetType
from rag_core.db import Database
from rag_db.repositories import AuditRepository

router = APIRouter(prefix="/audit", tags=["admin"])


@router.get(
    "",
    response_model=AuditQueryResponse,
    summary="查询审计日志",
    description=(
        "支持按动作、目标类型、目标 ID、结果过滤。\n\n"
        "**权限**：`super_admin` / `auditor` 可查全部租户；"
        "租户 `owner` / `admin` 只能查自己租户；`platform_operator` **不可查**。\n\n"
        "**查询本身会被审计**：查看他人行为记录是敏感操作。\n\n"
        "分页：`total` 默认不返回（审计表很大，COUNT(*) 很慢）。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.AUDIT))],
    responses=ADMIN_ENDPOINT,
)
async def query_audit_logs(
    ctx: AdminContext = Depends(get_admin_context),
    database: Database = Depends(get_database),
    page: PageRequest = Depends(page_query),
    tenant_id: uuid.UUID | None = Query(
        default=None, description="目标租户（仅平台角色可指定；租户管理员忽略此参数）"
    ),
    action: str | None = Query(default=None, description="动作类型过滤"),
    target_type: str | None = Query(default=None, description="目标对象类型过滤"),
    target_id: uuid.UUID | None = Query(default=None, description="目标对象 ID 过滤"),
    only_denied: bool = Query(default=False, description="只看被拒绝的操作"),
    since: datetime | None = Query(default=None, description="起始时间（含）"),
    until: datetime | None = Query(default=None, description="结束时间（不含）"),
) -> AuditQueryResponse:

    # 数据范围收敛：平台角色可指定租户，租户管理员只能看自己
    scoped = ctx.restrict_to(tenant_id) if tenant_id is not None else ctx.data_tenant_id

    # 会话按 scoped 打开：RLS 上下文与 WHERE 条件必须同范围，
    # 否则平台角色查别的租户时会被 RLS 过滤成空。
    async with scoped_admin_session(ctx, database, requested_tenant_id=scoped) as session:
        repo = AuditRepository(session)
        records = list(
            await repo.list_recent(
                tenant_id=scoped,
                limit=page.limit,
                action=_parse_action(action),
                target_type=_parse_target_type(target_type),
                target_id=target_id,
                only_denied=only_denied,
            )
        )

        # 时间范围过滤
        if since is not None or until is not None:
            records = [
                record
                for record in records
                if (since is None or record.created_at >= since)
                and (until is None or record.created_at < until)
            ]

        # ---- 查询本身留痕（与查询同事务）----
        # 事务回滚说明这次查询也没成功，审计不该留下。
        await repo.record(
            tenant_id=scoped,
            action=AuditAction.ADMIN_AUDIT_QUERY,
            outcome=AuditOutcome.SUCCESS,
            actor_user_id=ctx.user_id,
            details={
                "filters_action": action or "",
                "filters_target_type": target_type or "",
                "only_denied": only_denied,
                "result_count": len(records),
                "platform_role": ctx.platform_role.value,
            },
        )

        return to_response(records, limit=page.limit, offset=page.offset)


@router.get(
    "/verify",
    response_model=AuditChainVerification,
    summary="验证审计链完整性",
    description=(
        "重算哈希链，检查是否有记录被篡改或删除。\n\n"
        "**这是审计链存在的理由**：只写不验的链等于「相信没人改过数据库」，"
        "而威胁模型里恰恰包含内部人员。建议定期跑（或监管前跑）。\n\n"
        "`limit` 控制验证条数（O(n) 的 CPU 操作）：\n"
        "- `is_complete=true` 表示该租户全部记录都验过；\n"
        "- `is_complete=false` 表示只验了最早的 N 条 —— **不能读成「全表通过」**。\n\n"
        "验证本身写 `admin.audit_verify` 审计（谁在何时验了哪个租户，结果如何）。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.AUDIT))],
    responses=ADMIN_ENDPOINT,
)
async def verify_audit_chain(
    ctx: AdminContext = Depends(get_admin_context),
    database: Database = Depends(get_database),
    tenant_id: uuid.UUID | None = Query(default=None, description="目标租户（仅平台角色可指定）"),
    limit: int = Query(default=1000, ge=1, le=100_000, description="最多验证多少条（从 seq=1 起）"),
) -> AuditChainVerification:
    scoped = ctx.restrict_to(tenant_id) if tenant_id is not None else ctx.data_tenant_id

    async with scoped_admin_session(ctx, database, requested_tenant_id=scoped) as session:
        repo = AuditRepository(session)
        verified, first_broken_seq = await repo.verify_chain(tenant_id=scoped, limit=limit)
        total = await repo.count_total(tenant_id=scoped)

        # 验证动作也要留痕：谁在什么时候验了哪个租户、结果如何
        await repo.record(
            tenant_id=scoped,
            action=AuditAction.ADMIN_AUDIT_VERIFY,
            outcome=AuditOutcome.SUCCESS if verified else AuditOutcome.FAILED,
            actor_user_id=ctx.user_id,
            target_type=AuditTargetType.TENANT,
            target_id=scoped,
            details={
                "checked_limit": limit,
                "total_records": total,
                "verified": verified,
                "first_broken_seq": first_broken_seq if first_broken_seq is not None else 0,
                "platform_role": ctx.platform_role.value,
            },
        )

    # 注意：上面的 record() 会插入一条新记录，所以实际「已验证条数」
    # 是 limit 与验证时刻总数的最小值。用 min 而不是直接 limit：
    # 总量小于 limit 时 checked 应等于总量（否则 is_complete 永远 false）。
    checked = min(limit, total)
    return to_chain_verification(
        verified=verified,
        checked=checked,
        total=total,
        first_broken_seq=first_broken_seq,
    )


def _parse_action(value: str | None) -> AuditAction | None:
    """解析 action 过滤参数。

    未知值返回 422 而不是静默忽略：静默忽略会让用户以为过滤生效了，
    实际拿到全量数据——在审计场景下这是危险的（可能误判「没有越权记录」）。
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
    """解析 target_type 过滤参数（未知值 422）。"""
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


__all__ = ["router"]
