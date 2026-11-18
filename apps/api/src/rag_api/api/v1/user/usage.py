"""个人中心路由：我的用量、我的审计记录。

## 为什么这两个接口属于用户面

`docs/PROMPT.md` 的 B6 明确要求 Web 端有"个人中心：配额用量、我的审计记录"。
两者的共同点是**只看自己**：

- 用量：`user_id` 过滤（不是租户汇总），员工想知道"我为公司花了多少"；
- 审计：`actor_user_id` 过滤，员工有权知道"系统记录了我哪些操作"
  （这本身是透明性要求，也让员工能发现账号被盗用）。

## 与管理面的区别（不要合并）

| 接口 | 过滤条件 | 谁看 | 数据范围 |
|---|---|---|---|
| `GET /me/usage` | `user_id` | 任何登录用户 | 只有自己 |
| `GET /admin/cost` | `tenant_id` | 管理面 | 整个租户 |
| `GET /me/audit` | `actor_user_id` | 任何登录用户 | 只有自己的动作 |
| `GET /admin/audit` | `tenant_id` | 管理面 | 租户内所有人 |

合并成一个接口会带来"忘记传 user_id 就变成租户全量"的风险，
而这个错误的表现是 200 + 数据更多——**看起来一切正常**。
两个接口各自签名固定，误用会被 review 抓住。

## 审计查询自身要不要留痕

**不要**。管理面查审计要留痕是因为"窥探他人行为"是敏感操作。
查自己的记录不存在这个问题；留痕反而会把审计表灌满噪音
（每个人每次刷新个人中心都写一条）。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import get_db
from rag_api.deps.pagination import PageQuery, page_query
from rag_api.openapi import TENANT_ENDPOINT
from rag_api.permissions import PermissionContext, get_permission_context
from rag_api.schemas.audit import to_item
from rag_api.schemas.common import Page
from rag_api.schemas.usage import to_usage_summary
from rag_contracts.api.audit import AuditLogItem
from rag_contracts.api.pagination import PageMeta
from rag_contracts.api.usage import UsageSummary
from rag_contracts.enums import ModelTaskType
from rag_db.models.audit import AuditLog
from rag_db.models.model_call import ModelCall

router = APIRouter(tags=["me"])

# 用量查询的默认窗口：30 天。
# 不设默认窗口会让"个人用量"变成全表扫描（model_calls 是最大的表之一）。
_DEFAULT_USAGE_DAYS = 30
# 允许的最大窗口，防止一次请求扫三年数据。
_MAX_USAGE_DAYS = 366


@router.get(
    "/me/usage",
    response_model=UsageSummary,
    summary="我的模型用量与花费",
    description=(
        "按功能（chat / query_rewrite / embedding / rerank…）拆解**我自己**的调用量、"
        "token 与花费。\n\n"
        "默认最近 30 天。数据源是 `model_calls` 明细表，按 `user_id` 过滤——"
        "**不是租户汇总**（那是管理端的 `GET /admin/cost`）。\n\n"
        "`cost_usd` 用的是调用当时的单价快照，供应商调价后历史数字不变。"
    ),
    responses=TENANT_ENDPOINT,
)
async def get_my_usage(
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
    days: int = Query(
        default=_DEFAULT_USAGE_DAYS,
        ge=1,
        le=_MAX_USAGE_DAYS,
        description="统计最近多少天",
    ),
) -> UsageSummary:
    since = datetime.now(UTC) - timedelta(days=days)

    # 按 task_type 聚合（个人视图不需要 provider/model 维度：
    # 员工关心"问答花了多少"，不关心后端调的是哪个模型）
    stmt = (
        select(
            ModelCall.task_type,
            func.count().label("call_count"),
            func.coalesce(func.sum(ModelCall.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(ModelCall.output_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(ModelCall.cached_tokens), 0).label("cached_tokens"),
            func.coalesce(func.sum(ModelCall.cost_usd), 0).label("cost_usd"),
        )
        .where(
            ModelCall.tenant_id == ctx.tenant_id,
            # **user_id 是这里的核心过滤条件**：去掉它就变成租户汇总
            ModelCall.user_id == ctx.user_id,
            ModelCall.created_at >= since,
        )
        .group_by(ModelCall.task_type)
        .order_by(func.sum(ModelCall.cost_usd).desc())
    )
    rows = list((await session.execute(stmt)).all())

    hit_rate = await _user_cache_hit_rate(
        session, tenant_id=ctx.tenant_id, user_id=ctx.user_id, since=since
    )
    return to_usage_summary(
        rows,
        since=since,
        until=None,
        cache_hit_rate=hit_rate,
        task_type_factory=ModelTaskType,
    )


@router.get(
    "/me/audit",
    response_model=Page[AuditLogItem],
    summary="我的操作记录",
    description=(
        "返回**我自己**产生的审计记录（按时间倒序）。\n\n"
        "只按 `actor_user_id` 过滤，不看别人。想看租户内全部审计需要管理权限"
        "（`GET /admin/audit`）。\n\n"
        "**查询本身不留痕**：查自己的记录不是敏感操作，留痕只会给审计表加噪音。"
    ),
    responses=TENANT_ENDPOINT,
)
async def list_my_audit(
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
    page: PageQuery = Depends(page_query),
) -> Page[AuditLogItem]:
    conditions = [
        AuditLog.tenant_id == ctx.tenant_id,
        AuditLog.actor_user_id == ctx.user_id,
    ]

    stmt = (
        select(AuditLog)
        .where(*conditions)
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        .limit(page.limit)
    )
    if page.offset > 0:
        stmt = stmt.offset(page.offset)

    records = list((await session.execute(stmt)).scalars())

    total: int | None = None
    if page.include_total:
        count_stmt = select(func.count()).select_from(AuditLog).where(*conditions)
        total = int((await session.execute(count_stmt)).scalar_one())

    items = [to_item(record) for record in records]
    return Page(
        items=items,
        meta=PageMeta(
            limit=page.limit,
            offset=page.offset,
            total=total,
            has_more=len(items) == page.limit,
        ),
    )


async def _user_cache_hit_rate(
    session: AsyncSession, *, tenant_id: uuid.UUID, user_id: uuid.UUID, since: datetime
) -> float:
    """该用户的缓存命中率。

    与管理面的 `ModelCallRepository.cache_hit_rate` 分开：
    那个按租户，这个按用户——差别只有一个 WHERE 条件，但
    复用 `cache_hit_rate(tenant_id=...)` 会让个人页显示租户的数字。
    """
    result = await session.execute(
        select(
            func.count().label("total"),
            func.count().filter(ModelCall.cache_hit.is_(True)).label("hits"),
        ).where(
            ModelCall.tenant_id == tenant_id,
            ModelCall.user_id == user_id,
            ModelCall.created_at >= since,
        )
    )
    total, hits = result.one()
    return float(hits) / float(total) if total else 0.0


__all__ = ["router"]
