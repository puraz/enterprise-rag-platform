"""管理看板：业务计数 + 成本拆解。

## 为什么看板只放「数据库能回答」的指标

`docs/PROMPT.md` 的 C2 要求看板展示"QPS、P95、错误率、队列积压、
成本与预算水位、越权拒绝数"。这些指标分两类：

| 类别 | 来源 | 为什么不能混 |
|---|---|---|
| QPS / P95 / 错误率 | Prometheus | 进程内计数，重启归零；多副本时单实例数字无意义 |
| 队列积压 / 成本 / 越权数 | **数据库** | 全局事实，与副本无关 |

本模块只做后者。前者的正确用法是 Grafana 直接查 Prometheus，
而不是让 API 去 Prometheus 拉一次再转手——那样多一层故障点，
且图表的采样窗口会被 API 的请求间隔限制。

**例外**：`AppMetricsMiddleware` 已经在 `/health/metrics` 暴露了
Prometheus 格式，Grafana 直接抓那个端点即可。

## 为什么成本拆解按 task_type 而不是 provider

运营问的第一个问题通常是「钱花在哪了」——答案是「生成占 80%，改写占 5%」，
而不是「DeepSeek 花了多少、OpenAI 花了多少」（后者只有在比价时才看）。
按功能拆解直接对应可以优化的动作（比如「把改写换成小模型」）。
按 provider/model 的拆解在 `GET /admin/cost` 里给。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query

from rag_api.deps import get_database
from rag_api.openapi import ADMIN_ENDPOINT
from rag_api.permissions import AdminCapability, AdminContext, get_admin_context
from rag_api.permissions.admin import require_admin_capability, scoped_admin_session
from rag_api.schemas.admin import to_dashboard_response
from rag_contracts.api.admin import DashboardResponse
from rag_core.db import Database
from rag_core.errors import ValidationFailedError
from rag_db.repositories import (
    AuditRepository,
    DocumentRepository,
    KnowledgeBaseRepository,
    ModelCallRepository,
    TaskRepository,
)

router = APIRouter(prefix="/dashboard", tags=["admin"])

_DEFAULT_WINDOW_HOURS = 24
_MAX_WINDOW_HOURS = 24 * 90


@router.get(
    "",
    response_model=DashboardResponse,
    summary="运营看板（业务计数 + 成本拆解）",
    description=(
        "返回当前租户的业务计数与成本拆解。\n\n"
        "**不包含 QPS / P95 / 错误率**：那些是进程内指标，多副本下"
        "单实例数字无意义，应由 Grafana 直接抓 `/health/metrics`。\n\n"
        "**租户范围**：租户管理员只能看自己的租户；平台角色默认看"
        "token 里选中的租户，`tenant_id` 参数可显式指定别的租户。\n\n"
        "统计窗口默认最近 24 小时（成本与越权计数按窗口聚合；"
        "文档/任务计数是当前快照，不受窗口影响）。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.READ))],
    responses=ADMIN_ENDPOINT,
)
async def get_dashboard(
    ctx: AdminContext = Depends(get_admin_context),
    database: Database = Depends(get_database),
    tenant_id: str | None = Query(
        default=None, description="目标租户（平台角色可指定；租户管理员只能填自己）"
    ),
    window_hours: int = Query(
        default=_DEFAULT_WINDOW_HOURS,
        ge=1,
        le=_MAX_WINDOW_HOURS,
        description="成本与安全计数的统计窗口（小时）",
    ),
) -> DashboardResponse:
    import uuid as _uuid

    # 数据范围收敛：平台角色可用参数指定，租户管理员只能看自己
    if tenant_id is not None:
        try:
            requested = _uuid.UUID(tenant_id)
        except ValueError as exc:
            raise ValidationFailedError("tenant_id 不是合法 UUID") from exc
        scoped = ctx.restrict_to(requested)
    else:
        scoped = ctx.data_tenant_id

    since = datetime.now(UTC) - timedelta(hours=window_hours)

    # **会话必须按 scoped 打开**（不是 ctx.data_tenant_id）：
    # 平台角色指定了别的租户时，RLS 上下文必须跟着走，
    # 否则 RLS 会过滤掉目标租户的数据（返回 0 行，不报错）。
    async with scoped_admin_session(ctx, database, requested_tenant_id=scoped) as session:
        kb_count = await KnowledgeBaseRepository(session).count_active(tenant_id=scoped)

        docs = DocumentRepository(session)
        document_count = await docs.count_active_documents(tenant_id=scoped)
        documents_by_status = await docs.count_by_status(tenant_id=scoped)

        tasks = TaskRepository(session)
        tasks_by_status = await tasks.count_by_status(tenant_id=scoped)
        dead_letter_count = await tasks.count_dead(tenant_id=scoped)

        audit = AuditRepository(session)
        denied_count = await audit.count_denied(tenant_id=scoped, since=since)

        model_calls = ModelCallRepository(session)
        totals = await model_calls.totals(tenant_id=scoped, since=since)
        breakdown = await model_calls.breakdown(tenant_id=scoped, since=since)

    return to_dashboard_response(
        tenant_id=scoped,
        since=since,
        counters={
            "knowledge_base_count": kb_count,
            "document_count": document_count,
            "documents_by_status": documents_by_status,
            "tasks_by_status": tasks_by_status,
            "dead_letter_count": dead_letter_count,
            "denied_audit_count": denied_count,
            "total_cost_usd": float(totals["cost_usd"]),
        },
        breakdown=breakdown,
    )


__all__ = ["router"]
