"""数据库会话依赖：带租户上下文的会话（RLS 生效的前提）。

## 三个会话依赖的区别

| 依赖 | 角色 | 租户上下文 | 用途 |
|---|---|---|---|
| `get_db` | rag_app | ✅ set_config | **99% 的业务接口** |
| `get_session` | rag_app | ❌ | 不需要租户上下文的读（如查 users 表） |
| `get_admin_db` | rag_admin (BYPASSRLS) | ❌ | 跨租户操作（首登开通、管理看板、对账） |

**`get_db` 是默认选择**。误用 `get_admin_db` 的后果是绕过了租户隔离，
所以它要求显式声明平台角色，且调用点应能被审计追到。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps.app import get_database
from rag_api.deps.auth import Principal, get_current_principal
from rag_contracts.enums import PlatformRole
from rag_core.db import Database
from rag_core.errors import PermissionDeniedError
from rag_db.repositories import UserRepository


async def get_db(
    principal: Principal = Depends(get_current_principal),  # noqa: B008
    database: Database = Depends(get_database),  # noqa: B008
) -> AsyncIterator[AsyncSession]:
    """带租户上下文的会话（业务接口默认用这个）。

    在事务开始后立即执行 `set_config('app.tenant_id', :t, true)`，
    使 RLS 策略生效。即使某个仓储方法漏写了 `WHERE tenant_id`，
    数据库也会兜住。

    事务在请求结束时提交；异常时回滚。业务代码不需要自己 commit。
    """
    async with database.transaction(tenant_id=principal.tenant_id) as session:
        yield session


async def get_session(
    database: Database = Depends(get_database),  # noqa: B008
) -> AsyncIterator[AsyncSession]:
    """不带租户上下文的会话。

    适用：只读全局事实的表（`users`、`identity_links`、`tenants`），
    这些表不在 RLS_TABLES 里，不需要 app.tenant_id。

    **不要用它查租户内表**：没有租户上下文时 RLS 会返回 0 行，
    表现为"查不到数据"而不是报错——很难排查。
    """
    async with database.transaction() as session:
        yield session


async def get_admin_db(
    principal: Principal = Depends(get_current_principal),  # noqa: B008
    database: Database = Depends(get_database),  # noqa: B008
) -> AsyncIterator[AsyncSession]:
    """跨租户会话（BYPASSRLS 管理连接）。

    仅用于：管理面看板、首登开通租户/成员、后台对账。

    **会先校验平台角色**：没有平台权限的普通用户即使调到管理接口也拿不到
    这个会话。这样即使某个管理路由忘了加 `require_platform_role`，
    这里还有一道检查。

    但内部服务（首登开通）不属于"平台管理员操作"，它们用
    `Database.transaction_bypass_rls()` 直接拿会话，不经过这个依赖。
    """
    async with database.transaction() as session:
        user = await UserRepository(session).get_active_by_id(principal.user_id)
        if user is None or PlatformRole(str(user.platform_role)) is PlatformRole.NONE:
            raise PermissionDeniedError("需要平台管理权限")

    async with database.transaction_bypass_rls() as session:
        yield session
