"""权限上下文依赖：从 token + 数据库构造 `PermissionContext`。

## 为什么角色要从数据库查，不能放 token

放 token 里的问题：
1. **角色变更不即时生效**：管理员把某人降级后，对方要等 token 过期
   （30 分钟）才失去权限；
2. **token 体积膨胀**：角色、部门、密级都塞进去会让 header 超长；
3. **撤销困难**：无法在不改动 token 的情况下调整权限。

放数据库、每次请求查一次，代价是一次索引查找（`(tenant_id, user_id)` 有
复合索引，A3-1 已建）。这是正确的取舍。

**缓存在 A13 做**：Redis 缓存 `(tenant_id, user_id) → roles`，
带版本号失效。现在每次都查库，保证正确性优先。
"""

from __future__ import annotations

import uuid

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import Principal, get_current_principal, get_db
from rag_api.permissions.acl import PermissionContext
from rag_contracts.enums import MembershipStatus, TenantRole
from rag_core.logging import get_logger
from rag_db.repositories import MembershipRepository, UserRepository

log = get_logger(__name__)


async def get_permission_context(
    principal: Principal = Depends(get_current_principal),  # noqa: B008
    session: AsyncSession = Depends(get_db),  # noqa: B008
) -> PermissionContext:
    """构造当前请求的权限上下文。

    **必须用 `get_db`**（带租户上下文的会话）：成员关系表
    `tenant_memberships` 受 RLS 保护，没有租户上下文时查不到。

    返回的 `roles` 是**用户在该租户内的所有 active 角色**——
    通常只有一个，但 A3-1 的唯一约束是 `(tenant_id, user_id)`，
    所以理论上就一条。取 set 是为了将来支持"多角色"时不用改签名。
    """
    # 确认用户仍可用（token 有效但账号被禁用的场景）
    user = await UserRepository(session).get_active_by_id(principal.user_id)
    if user is None:
        from rag_core.errors import AuthenticationError

        raise AuthenticationError("账号不可用，请重新登录")

    membership = await MembershipRepository(session).get_active(
        tenant_id=principal.tenant_id, user_id=principal.user_id
    )

    roles: frozenset[TenantRole] = frozenset()
    if membership is not None and membership.status == MembershipStatus.ACTIVE.value:
        try:
            roles = frozenset({TenantRole(str(membership.role))})
        except ValueError:
            # 未知角色（数据库里有脏数据）：不给任何权限，并告警
            log.error(
                "unknown_tenant_role",
                role=str(membership.role),
                user_id=str(principal.user_id),
                tenant_id=str(principal.tenant_id),
            )

    if not roles:
        # 不属于该租户：这是越权信号（token 里的 tenant 不属于用户）
        # 但**不在这里抛异常**：有些接口（如 /me）允许无角色访问。
        # 需要权限的接口自己调 `require_role()`。
        log.info(
            "no_active_membership",
            user_id=str(principal.user_id),
            tenant_id=str(principal.tenant_id),
        )

    return PermissionContext(
        user_id=principal.user_id,
        tenant_id=principal.tenant_id,
        roles=roles,
        # 部门归属：departments 表未建模（A7 后续），先为空。
        # 空集合让 `visibility=department` 的库不可见（安全默认）。
        department_ids=frozenset(),
    )


def require_tenant_role(*roles: TenantRole, action: str = "此操作") -> object:
    """生成校验租户角色的依赖。

    用法：

        @router.post(
            "/knowledge-bases",
            dependencies=[Depends(require_tenant_role(TenantRole.ADMIN))],
        )

    与 `require_platform_role`（A4）的区别：
    - 本函数校验**租户内**角色（owner/admin/knowledge_admin/...）；
    - `require_platform_role` 校验**平台**角色（super_admin/auditor/...）。

    两者正交：平台超管未必是任何租户的成员。
    """

    async def _check(
        ctx: PermissionContext = Depends(get_permission_context),  # noqa: B008
    ) -> None:
        ctx.require_role(*roles, action=action)

    return _check


def require_kb_write() -> object:
    """要求能写知识库内容（owner/admin/knowledge_admin）。"""

    async def _check(
        ctx: PermissionContext = Depends(get_permission_context),  # noqa: B008
    ) -> None:
        ctx.require_kb_write()

    return _check


def require_member_manage() -> object:
    """要求能管理租户成员（owner/admin）。"""

    async def _check(
        ctx: PermissionContext = Depends(get_permission_context),  # noqa: B008
    ) -> None:
        ctx.require_member_manage()

    return _check


def ensure_own_resource(*, resource_user_id: uuid.UUID, ctx: PermissionContext) -> None:
    """校验资源属于当前用户（或用户有管理权限）。

    用于"我的上传会话""我的会话记录"这类只有本人可访问的资源。
    """
    from rag_contracts.enums import TenantRole
    from rag_core.errors import NotFoundError

    if resource_user_id == ctx.user_id:
        return
    # 管理员可以访问他人的资源（如排查问题）
    if ctx.has_any_role(TenantRole.OWNER, TenantRole.ADMIN):
        return
    # 非本人且非管理员：返回 404（不泄漏资源存在性）
    raise NotFoundError("资源不存在")
