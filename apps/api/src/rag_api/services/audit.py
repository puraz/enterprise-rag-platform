"""审计服务：把审计写入封装成易用的接口。

## 两种事务模式（A3-5 已设计，这里是实际用法）

| 场景 | 用法 | 为什么 |
|---|---|---|
| 操作成功（上传了文档） | `audit.success(...)` 与业务同事务 | 业务回滚则审计该消失 |
| 安全事件（越权被拒） | `audit.security(...)` 独立事务 | 业务回滚后审计**必须留下** |

实现上，`AuditService` 持有 session，由调用方决定事务边界：

    # 同事务
    async with db.transaction(tenant_id=t) as session:
        await upload_service.complete(...)
        await AuditService(session).success(action=..., ...)

    # 独立事务（异常路径）
    async with db.transaction(tenant_id=t) as session:
        try:
            ...
        except PermissionDeniedError:
            raise
    # 捕获后另开事务写审计

## 为什么不在中间件里自动审计

自动审计（比如"每个请求记一条"）会导致：
- 审计表被无意义的读操作灌满（每天几十万条 GET）；
- 关键事件（越权、删除、权限变更）淹没在噪音里；
- 业务语义丢失（中间件不知道"这个请求是在删文档还是查列表"）。

所以审计是**显式调用**的，只在有语义价值的地方写。
`docs/PROMPT.md` 要求的是"谁/何时/查了什么/命中哪些文档/是否越权被拒"——
这需要业务上下文，中间件给不了。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.permissions.acl import PermissionContext
from rag_contracts.enums import (
    AuditAction,
    AuditOutcome,
    AuditTargetType,
    PlatformRole,
    TenantRole,
)
from rag_core.errors import PermissionDeniedError, ValidationFailedError
from rag_core.logging import get_logger
from rag_db.models.audit import AuditLog
from rag_db.repositories import AuditRepository, UserRepository

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AuditContext:
    """审计的请求侧上下文（由路由层提供）。

    集中成一个对象而不是几十个参数：新增字段（如 `user_agent`）
    时不用改动所有调用点。
    """

    request_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None


class AuditService:
    """审计写入的薄封装。

    职责只有两件：
    1. 把 `PermissionContext` + `AuditContext` 填进 `AuditLog` 的字段；
    2. 保证 details 里**不含用户数据**。
    """

    def __init__(self, session: AsyncSession) -> None:
        self._repo = AuditRepository(session)

    async def record(
        self,
        *,
        ctx: PermissionContext,
        action: AuditAction,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        request: AuditContext | None = None,
        target_type: AuditTargetType | None = None,
        target_id: uuid.UUID | None = None,
        target_name: str | None = None,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        actor_email: str | None = None,
        actor_tenant_role: str | None = None,
    ) -> None:
        """写一条审计。

        **本方法必须与业务在同一事务里调用**（见模块 docstring）。
        """
        await self._repo.record(
            tenant_id=ctx.tenant_id,
            action=action,
            outcome=outcome,
            actor_user_id=ctx.user_id,
            actor_email=actor_email,
            actor_tenant_role=actor_tenant_role,
            target_type=target_type,
            target_id=target_id,
            target_name=target_name,
            request_id=request.request_id if request else None,
            ip_address=request.ip_address if request else None,
            user_agent=request.user_agent if request else None,
            details=_safe_details(details),
            error_code=error_code,
        )

    async def denied(
        self,
        *,
        ctx: PermissionContext,
        action: AuditAction,
        reason: str,
        request: AuditContext | None = None,
        target_type: AuditTargetType | None = None,
        target_id: uuid.UUID | None = None,
        error_code: str | None = None,
    ) -> None:
        """记录一次被拒绝的操作（安全事件）。

        **调用方必须用独立事务**：这个调用发生在业务异常路径上，
        原事务即将回滚。
        """
        await self.record(
            ctx=ctx,
            action=action,
            outcome=AuditOutcome.DENIED,
            request=request,
            target_type=target_type,
            target_id=target_id,
            details={"reason": reason},
            error_code=error_code,
        )
        # 同时打日志：审计表的"被拒"记录是异步消费的，
        # 而越权尝试需要即时告警
        log.warning(
            "access_denied",
            action=action.value,
            user_id=str(ctx.user_id),
            tenant_id=str(ctx.tenant_id),
            reason=reason,
            target_id=str(target_id) if target_id else None,
        )


# 能查全部租户审计的平台角色
_GLOBAL_AUDIT_ROLES = frozenset({PlatformRole.SUPER_ADMIN, PlatformRole.AUDITOR})
# 能查本租户审计的租户角色
_TENANT_AUDIT_ROLES: tuple[TenantRole, ...] = (TenantRole.OWNER, TenantRole.ADMIN)


class AuditQueryService:
    """审计查询（管理面）。

    职责：权限判定、过滤参数校验、查询留痕、ORM → 契约映射。
    路由只负责把查询参数传进来并返回结果。

    ## 为什么审计查询本身也要审计

    查审计日志是**敏感操作**：它能暴露"谁在什么时候看了什么文档"。
    如果查审计不被记录，内部人员可以悄悄窥探他人行为。
    记录在**同一事务**里：事务回滚说明这次查询也没成功，审计不该留下。
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = AuditRepository(session)
        self._users = UserRepository(session)

    async def query(
        self,
        *,
        ctx: PermissionContext,
        limit: int,
        action: str | None = None,
        target_type: str | None = None,
        target_id: uuid.UUID | None = None,
        only_denied: bool = False,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[AuditLog]:
        """查询当前用户有权看到的审计日志（按时间倒序）。"""
        platform_role = await self._require_query_permission(ctx)

        records = list(
            await self._repo.list_recent(
                tenant_id=ctx.tenant_id,
                limit=limit,
                action=_parse_action(action),
                target_type=_parse_target_type(target_type),
                target_id=target_id,
                only_denied=only_denied,
            )
        )

        # ---- 查询本身留痕 ----
        await AuditService(self._session).record(
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

        # 时间范围过滤（审计记录量不大时够用；数据量大时应下推到 SQL）
        if since is not None or until is not None:
            records = [
                record
                for record in records
                if (since is None or record.created_at >= since)
                and (until is None or record.created_at < until)
            ]

        items = list(records)

        log.info(
            "audit_query_executed",
            platform_role=platform_role.value,
            tenant_roles=sorted(role.value for role in ctx.roles),
            result_count=len(items),
            queried_by=str(ctx.user_id),
        )

        return items

    async def _require_query_permission(self, ctx: PermissionContext) -> PlatformRole:
        """校验查询权限，返回平台角色（用于审计留痕）。"""
        user = await self._users.get_active_by_id(ctx.user_id)
        if user is None:
            raise PermissionDeniedError("账号不可用")

        try:
            platform_role = PlatformRole(str(user.platform_role))
        except ValueError:
            platform_role = PlatformRole.NONE

        can_query_global = platform_role in _GLOBAL_AUDIT_ROLES
        can_query_tenant = ctx.has_any_role(*_TENANT_AUDIT_ROLES)

        if not (can_query_global or can_query_tenant):
            raise PermissionDeniedError("需要审计员或租户管理员权限才能查看审计日志")
        return platform_role


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

        raise ValidationFailedError(
            fields=[
                FieldError(
                    location=["query", "target_type"],
                    message=f"未知的目标类型: {value}",
                    type="value_error",
                )
            ]
        ) from exc


def _safe_details(details: dict[str, Any] | None) -> dict[str, Any]:
    """过滤审计 details，只保留可公开的结构化信息。

    **审计 details 不得含用户数据**：文档正文、prompt、邮箱、
    用户输入的原值都不该进审计表。审计表会被长期保留（180 天），
    里面的用户数据是合规风险。

    允许的类型：int/float/bool/str（短字符串）+ 它们的列表。
    超长字符串（可能是文档片段）一律截断。
    """
    if not details:
        return {}

    from rag_api.security.pii import redact_text

    safe: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, (bool, int, float)):
            safe[key] = value
        elif isinstance(value, str):
            # 截断长字符串后再脱敏：先截断避免对超长文本跑正则，
            # 再脱敏保证 PII 不进审计表（保留 180 天，合规风险）
            safe[key] = redact_text(value[:200])
        elif isinstance(value, (list, tuple)):
            safe[key] = [
                item if isinstance(item, (bool, int, float)) else str(item)[:100]
                for item in value[:50]  # 列表也限长
            ]
        else:
            # 复杂对象不记录（可能是 ORM 对象或文档内容）
            safe[key] = f"<{type(value).__name__}>"
    return safe


def extract_request_context(request: object) -> AuditContext:
    """从 FastAPI Request 提取审计上下文。

    用 `object` 类型而不是 `Request`：让本模块不依赖 starlette
    （便于 worker 复用）。
    """
    headers = getattr(request, "headers", {})
    client = getattr(request, "client", None)
    state = getattr(request, "scope", {}).get("state", {}) if request else {}

    return AuditContext(
        request_id=str(state.get("request_id")) if state.get("request_id") else None,
        # 客户端 IP：注意 X-Forwarded-For 是**不可信**的（客户端可伪造）。
        # 生产应该用可信代理链（proxy headers middleware）或者只用
        # 直接的 client.host。这里用 client.host，交由网关层处理真实 IP。
        ip_address=str(client.host) if client and client.host else None,
        user_agent=headers.get("user-agent", "")[:512] or None,
    )
