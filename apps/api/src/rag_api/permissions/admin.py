"""管理面权限：统一的"谁能进管理面、能做什么"判定。

## 为什么必须有这一层（而不是每个接口自己判）

A4 只有 `/admin/audit` 一个管理接口，权限判定写在 `AuditQueryService`
内部，看起来没问题。但管理面要长到十几个接口时，每个接口自己写：

    platform_role in {SUPER_ADMIN, AUDITOR} or ctx.has_any_role(OWNER, ADMIN)

会立刻出现三类错误，而且**都不报错**：

1. **漏判**：新接口忘了写这行 → 任何登录用户都能调（最严重）；
2. **判错**：某个接口把 `AUDITOR` 写成"能改"（审计员本应只读）；
3. **口径漂移**：A 接口允许 `platform_operator`，B 接口不允许，
   前端无法解释"为什么同一个运营账号，一个页面能进一个不能"。

所以把"管理面 = 平台角色 ∪ 本租户 owner/admin"做成一个**依赖**：
路由只需要声明"我需要哪种能力"，判定逻辑只有一处。

## 能力划分（对应 docs/PROMPT.md 的 C 阶段）

| 能力 | 含义 | 允许的平台角色 | 允许的租户角色 |
|---|---|---|---|
| `read` | 看板、台账、成本、审计查询 | super_admin / platform_operator / auditor | owner / admin |
| `operate` | 重试任务、切索引版本等运维动作 | super_admin / platform_operator | owner / admin |
| `audit` | 查审计、验证审计链 | super_admin / auditor | owner / admin |
| `admin` | 改权限、改配置等治理动作 | **仅 super_admin** | owner / admin |

`audit` 与 `read` 分开的原因：`platform_operator` 管运维复本，
但**不该能查别人的行为审计**（那能暴露"谁看了什么文档"）。
而 `auditor` 恰恰相反：能查审计，不能改任何东西。

## 一个租户 owner 能不能看别的租户

不能。`AdminContext` 里带 `scope_tenant_id`：
- 平台角色（非 NONE）→ `scope_tenant_id=None`，表示可跨租户（由接口
  自行决定是否真的跨：看板默认只看当前请求租户，跨租户汇总需显式开关）；
- 租户 owner/admin → `scope_tenant_id=token.tenant_id`，接口必须用它过滤。

**这个字段是安全边界**：所有管理查询都必须带上它（或显式跨租户），
不能只看 `ctx.tenant_id`——后者对平台角色来说只是"登录时选中的租户"，
不代表数据范围。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import Principal, get_current_principal, get_database
from rag_api.permissions.acl import PermissionContext
from rag_contracts.enums import PlatformRole, TenantRole
from rag_core.db import Database
from rag_core.errors import PermissionDeniedError
from rag_core.logging import get_logger
from rag_db.repositories import MembershipRepository, UserRepository

log = get_logger(__name__)


class AdminCapability(StrEnum):
    """管理面能力。"""

    READ = "read"
    """只读：看板、台账、成本、索引版本列表。"""

    OPERATE = "operate"
    """运维动作：重试死信任务、切换索引版本。**会产生副作用**。"""

    AUDIT = "audit"
    """审计读取与链验证。"""

    ADMIN = "admin"
    """治理动作：改权限、改配置。仅平台超管 / 租户 owner-admin。"""


# 平台角色 → 允许的能力
_PLATFORM_CAPABILITIES: dict[PlatformRole, frozenset[AdminCapability]] = {
    PlatformRole.SUPER_ADMIN: frozenset(AdminCapability),
    PlatformRole.PLATFORM_OPERATOR: frozenset({AdminCapability.READ, AdminCapability.OPERATE}),
    PlatformRole.AUDITOR: frozenset({AdminCapability.READ, AdminCapability.AUDIT}),
    PlatformRole.NONE: frozenset(),
}

# 租户内角色 → 允许的能力。
#
# **owner 与 admin 拿到全部能力，包括 ADMIN**：租户管理员本来就能改
# 自己租户的配置与成员角色。这不是"平台治理权限"，作用域被
# `scope_tenant_id` 限制在本租户内。
_TENANT_CAPABILITIES: dict[TenantRole, frozenset[AdminCapability]] = {
    TenantRole.OWNER: frozenset(AdminCapability),
    TenantRole.ADMIN: frozenset(AdminCapability),
}


@dataclass(frozen=True, slots=True)
class AdminContext:
    """管理面请求上下文。

    字段：
    - `user_id`：操作者（审计留痕用）；
    - `platform_role`：平台角色；
    - `tenant_id`：token 里选中的租户（平台角色也可能有）；
    - `tenant_roles`：本租户内的角色（可能为空——平台超管未必是租户成员）；
    - `scope_tenant_id`：**数据范围**。None 表示可跨租户；
      非 None 表示只能看这个租户的数据。
    - `capabilities`：允许的能力集合。

    `scope_tenant_id` 与 `tenant_id` 是两个不同的概念，混用会泄漏：
    - `tenant_id` 回答"我登录时选了哪个租户"；
    - `scope_tenant_id` 回答"我能看哪些租户的数据"。

    一个平台超管登录在租户 A，但他有权看 A 和 B 的数据；
    一个租户 owner 登录在 A，他只能看 A。
    """

    user_id: uuid.UUID
    platform_role: PlatformRole
    tenant_id: uuid.UUID
    tenant_roles: frozenset[TenantRole]
    scope_tenant_id: uuid.UUID | None
    capabilities: frozenset[AdminCapability]

    @property
    def is_platform(self) -> bool:
        """是否为平台角色（可跨租户）。"""
        return self.scope_tenant_id is None and self.platform_role is not PlatformRole.NONE

    def require(self, capability: AdminCapability) -> None:
        """要求具备某项能力，否则 403。"""
        if capability not in self.capabilities:
            log.info(
                "admin_capability_denied",
                user_id=str(self.user_id),
                platform_role=self.platform_role.value,
                tenant_roles=sorted(role.value for role in self.tenant_roles),
                required=capability.value,
            )
            raise PermissionDeniedError(f"需要 {capability.value} 管理权限")

    def restrict_to(self, tenant_id: uuid.UUID) -> uuid.UUID:
        """把一次查询的租户范围收敛到"用户可访问 ∪ 请求的租户"。

        ## 为什么收敛逻辑必须集中在这里

        接口拿到一个 `tenant_id` 参数（或从路径里取）时，必须回答：
        "这个用户能看这个租户的数据吗？"三种情况：

        - 平台角色：可以看任何租户 → 返回请求的租户；
        - 租户 owner/admin：只能看自己的租户 → 请求别的租户要拒绝；
        - 都不可能（没有能力）：`require()` 已经拦住了，走不到这里。

        写成方法而不是让每个接口自己 if：
        漏写一次就是跨租户越权，而且接口返回 200 完全看不出来。
        """
        if self.scope_tenant_id is None and self.platform_role is not PlatformRole.NONE:
            return tenant_id
        if tenant_id != self.scope_tenant_id:
            log.warning(
                "admin_cross_tenant_denied",
                user_id=str(self.user_id),
                requested_tenant=str(tenant_id),
                scope_tenant=str(self.scope_tenant_id),
            )
            # 403 而不是 404：管理面是显式的租户参数，
            # 不存在"资源是否存在"的探测风险（参数本身就是租户 ID）
            raise PermissionDeniedError("无权访问该租户的数据")
        return tenant_id

    @property
    def data_tenant_id(self) -> uuid.UUID:
        """默认数据范围（未显式指定租户时用）。

        平台角色没有固定范围时用 token 里的租户：管理看板默认展示
        "我当前登录租户"的数据，而不是全平台汇总——
        全量汇总需要显式开关（`all_tenants=true`），避免误看。
        """
        return self.scope_tenant_id or self.tenant_id


async def get_admin_context(
    principal: Principal = Depends(get_current_principal),  # noqa: B008
    database: Database = Depends(get_database),  # noqa: B008
) -> AdminContext:
    """构造管理面上下文。

    ## 为什么用两段短事务，而不是一个长期会话

    角色查询需要两种上下文，**RLS 决定了不能用同一个会话**：

    | 查什么 | 表在 RLS 里？ | 需要的上下文 |
    |---|---|---|
    | 平台角色（`users.platform_role`） | ❌ 不在 | 无租户（普通会话） |
    | 租户角色（`tenant_memberships`） | ✅ 在 | **必须**设置 `app.tenant_id` |

    `tenant_memberships` 在 `RLS_TABLES` 里。如果用无租户上下文的会话查它，
    RLS 策略里的 `tenant_id = NULLIF(current_setting(...), '')::uuid` 会
    判为 NULL → **返回 0 行，且不报错**。表现是"租户 owner 调管理接口返回 403"，
    而根因是查不到自己的成员关系——这个 bug 在超级用户（BYPASSRLS）环境下
    测不出来，只在真实应用角色下暴露。

    所以这里用两个已经关闭的短事务（与 `_load_platform_role` 同样的做法），
    上下文用完即弃，不把会话交给请求处理器。

    ## 安全前提

    本依赖不提供会话。所有管理查询必须通过 `get_admin_db`（它按数据范围
    设置租户上下文）或显式写 `tenant_id` 条件。这一约束由
    `AdminContext.restrict_to()` + 管理面仓储方法的签名共同保证。
    """
    # ---- 平台角色：users 表不在 RLS 里，普通会话即可 ----
    async with database.transaction() as session:
        user = await UserRepository(session).get_active_by_id(principal.user_id)
        if user is None:
            # token 有效但账号被禁用：管理面比用户面更严格，直接 403
            raise PermissionDeniedError("账号不可用")
        try:
            platform_role = PlatformRole(str(user.platform_role))
        except ValueError:
            log.error("unknown_platform_role", value=str(user.platform_role))
            platform_role = PlatformRole.NONE

    # ---- 租户角色：tenant_memberships **在 RLS 里**，必须带租户上下文 ----
    #
    # 用 `transaction(tenant_id=...)` 让 RLS 生效。写成无上下文会话的话，
    # 查询会静默返回 0 行（详见 docstring），表现为"租户 owner 进不了管理面"。
    tenant_roles: frozenset[TenantRole] = frozenset()
    async with database.transaction(tenant_id=principal.tenant_id) as session:
        # 这条查询**必须显式写 tenant_id**：RLS 是兜底不是唯一防线，
        # 且脚本/测试可能用 BYPASSRLS 角色跑，那时 RLS 不生效。
        membership = await MembershipRepository(session).get_active(
            tenant_id=principal.tenant_id, user_id=principal.user_id
        )
        if membership is not None:
            try:
                tenant_roles = frozenset({TenantRole(str(membership.role))})
            except ValueError:
                log.error("unknown_tenant_role", value=str(membership.role))

    capabilities: set[AdminCapability] = set(_PLATFORM_CAPABILITIES[platform_role])
    for role in tenant_roles:
        capabilities |= _TENANT_CAPABILITIES.get(role, frozenset())

    # 数据范围：
    # - 有平台角色（非 NONE）→ None（可跨租户，由接口决定看不看）
    # - 否则锁定在 token 的租户
    scope_tenant_id = None if platform_role is not PlatformRole.NONE else principal.tenant_id

    ctx = AdminContext(
        user_id=principal.user_id,
        platform_role=platform_role,
        tenant_id=principal.tenant_id,
        tenant_roles=tenant_roles,
        scope_tenant_id=scope_tenant_id,
        capabilities=frozenset(capabilities),
    )

    if not capabilities:
        log.info(
            "admin_access_denied",
            user_id=str(principal.user_id),
            tenant_id=str(principal.tenant_id),
            platform_role=platform_role.value,
            tenant_roles=sorted(role.value for role in tenant_roles),
        )
        raise PermissionDeniedError("需要平台管理权限或租户管理员权限")

    return ctx


def require_admin_capability(capability: AdminCapability) -> Callable[..., Awaitable[None]]:
    """生成"要求某项管理能力"的依赖。

    用法：

        @router.get(
            "/admin/dashboard",
            dependencies=[Depends(require_admin_capability(AdminCapability.READ))],
        )

    与直接 `Depends(get_admin_context)` 的区别：
    - 本函数同时保证"是管理面用户"与"有这项能力"；
    - 只依赖 `get_admin_context` 的接口需要自己调 `ctx.require(...)`，
      而漏调就变成"任何管理面用户都能调"。
    """

    async def _check(ctx: AdminContext = Depends(get_admin_context)) -> None:  # noqa: B008
        ctx.require(capability)

    return _check


def to_permission_context(admin: AdminContext) -> PermissionContext:
    """把管理上下文降级成普通权限上下文（需要复用 ACL 逻辑时用）。

    用途：管理面的 chunk 预览要判断"管理员本身能否读这份文档内容"。
    这里刻意**只取租户角色、不放大平台角色**：
    平台超管看文档正文仍然需要显式审计（见 chunk 预览接口）。
    """
    return PermissionContext(
        user_id=admin.user_id,
        tenant_id=admin.data_tenant_id,
        roles=admin.tenant_roles,
        department_ids=frozenset(),
    )


async def get_admin_db(
    ctx: AdminContext = Depends(get_admin_context),  # noqa: B008
    database: Database = Depends(get_database),  # noqa: B008
) -> AsyncIterator[AsyncSession]:
    """按数据范围设置租户上下文的会话（管理面默认用这个）。

    ## 为什么不能直接用 `get_session` / `deps.get_admin_db`

    | 依赖 | 租户上下文 | 管理面的问题 |
    |---|---|---|
    | `get_session` | ❌ | RLS 表（documents/tasks/model_calls…）**静默返回 0 行** |
    | `deps.get_admin_db` | ❌ + BYPASSRLS | 绕过租户隔离，靠 WHERE 自觉——漏一个条件就是跨租户 |
    | **本函数** | ✅ `ctx.data_tenant_id` | RLS 生效 + 查询仍显式带 tenant_id，双保险 |

    关键点：管理面的绝大部分查询**本来就是单租户查询**
    （"看这个租户的队列积压"）。用 `SET LOCAL app.tenant_id` 打开 RLS
    是最贴合语义的做法——它同时提供正确性（不会漏过滤）与安全性
    （SQL 漏写条件时数据库兜底）。

    代价：**跨租户汇总不能用本依赖**（RLS 会把它压成单租户）。
    需要全平台视图时显式用 `transaction_bypass_rls`，且那必须是一个
    有审计的专门接口，而不是顺手开个 `all_tenants=true` 开关。

    ## 平台角色指定了别的租户怎么办

    路由可能带 `tenant_id` 查询参数（平台角色运维别的租户）。
    本依赖**不能**读查询参数：`ctx.restrict_to()` 的收敛逻辑必须在
    路由里显式调用（它带审计日志），依赖里偷偷读参数会让"权限判定"
    散在两个地方。

    所以约定是：**需要跨租户参数的路由，在函数体里显式开新事务**
    （`database.transaction(tenant_id=scoped)`），而不是依赖本函数。
    本函数只服务"范围就是 `ctx.data_tenant_id`"的路由。

    ## 使用约定

    管理面的查询仍然要显式写 `tenant_id` 条件（见各仓储方法的签名）。
    RLS 是最后一道防线——**不能因为它存在就省掉 WHERE**：
    超级用户 / 迁移角色 / 未来某个 BYPASSRLS 路径都会让 RLS 静默失效。
    """
    async with database.transaction(tenant_id=ctx.data_tenant_id) as session:
        yield session


@asynccontextmanager
async def scoped_admin_session(
    ctx: AdminContext, database: Database, *, requested_tenant_id: uuid.UUID | None
) -> AsyncIterator[AsyncSession]:
    """按"请求指定的租户（经权限收敛）"开会话。

    ## 为什么这个函数不能做成 FastAPI 依赖

    它需要 `ctx` 与 `requested_tenant_id` 两个输入，而后者来自
    路由的查询参数。写成依赖就必须在依赖里读 `Request.query_params`
    然后重复解析 UUID、重复错误处理——而权限收敛（`restrict_to`）
    是安全逻辑，只应有一处实现，且必须在有审计上下文的路由里调用。

    所以它是**普通异步上下文管理器**，由路由显式使用：

        scoped = ctx.restrict_to(requested) if requested else ctx.data_tenant_id
        async with scoped_admin_session(ctx, database, requested_tenant_id=scoped) as session:
            ...

    返回的会话已设置 `app.tenant_id = scoped`，RLS 与显式 WHERE 条件
    使用同一个范围——不会出现"WHERE 用 A、RLS 用 B"的错位。
    """
    resolved = (
        ctx.restrict_to(requested_tenant_id)
        if requested_tenant_id is not None
        else ctx.data_tenant_id
    )
    async with database.transaction(tenant_id=resolved) as session:
        yield session


__all__ = [
    "AdminCapability",
    "AdminContext",
    "get_admin_context",
    "get_admin_db",
    "require_admin_capability",
    "scoped_admin_session",
    "to_permission_context",
]
