"""FastAPI 依赖注入。

## 依赖关系的顺序问题（本步最关键的设计）

请求处理的正确顺序：

    1. 解析 token → 得到 (user_id, tenant_id)
    2. 用 tenant_id 开会话 → 触发 set_config('app.tenant_id')
    3. 业务查询 → 受 RLS 约束

如果顺序错了（先开会话再解析 token），事务里没有 `app.tenant_id`，
**所有查询会返回 0 行** —— 表现为"登录成功但列表永远是空的"。

因此 `get_db` 接受租户参数，且 `get_current_principal` 不依赖数据库
（只验签），保证"先解析、后开库"。

## Application state 的用法

`Database` / `TokenService` / `IdentityProvider` 这些**进程级单例**
挂在 `app.state` 上，由 lifespan 初始化。依赖函数从 `request.app.state` 取。

为什么不用模块级全局变量：测试需要创建多个 app 实例（不同配置），
全局变量会在测试之间互相污染。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.identity.tokens import TokenClaims, TokenService, bearer_token
from rag_core.db import Database
from rag_core.errors import PermissionDeniedError
from rag_core.redis import RedisClient


@dataclass(frozen=True, slots=True)
class Principal:
    """当前请求的身份主体。

    只包含**从 token 解析出的可信信息**。业务需要更多信息
    （如用户邮箱、租户角色）时，应通过仓储查询而不是往这里加字段——
    那会诱导开发者把不可信的外部输入塞进来。
    """

    user_id: uuid.UUID
    tenant_id: uuid.UUID
    token_id: str


# ---------------------------------------------------------------- 应用单例


def get_database(request: Request) -> Database:
    """进程级数据库资源。"""
    database: Database = request.app.state.database
    return database


def get_redis(request: Request) -> RedisClient:
    """进程级 Redis 资源。"""
    redis: RedisClient = request.app.state.redis
    return redis


def get_token_service(request: Request) -> TokenService:
    return request.app.state.token_service  # type: ignore[no-any-return]


# ---------------------------------------------------------------- 认证


async def get_current_principal(
    request: Request,
    token_service: TokenService = Depends(get_token_service),  # noqa: B008
) -> Principal:
    """解析并校验 access token。

    **这一步不访问数据库**：只做签名与声明校验。
    好处是非法请求在触库前就被拒绝（省一次连接），
    且保证后续 `get_db` 一定能拿到 tenant_id。
    """
    credential = bearer_token(request.headers.get("authorization"))
    claims: TokenClaims = token_service.verify(credential)
    return Principal(
        user_id=claims.user_id,
        tenant_id=claims.tenant_id,
        token_id=claims.token_id,
    )


# ---------------------------------------------------------------- 数据库会话


async def get_db(
    principal: Principal = Depends(get_current_principal),  # noqa: B008
    database: Database = Depends(get_database),  # noqa: B008
) -> AsyncIterator[AsyncSession]:
    """带租户上下文的数据库会话。

    **自动设置 `app.tenant_id`**：这是 RLS 生效的前提。
    即使某个仓储方法忘了写 `WHERE tenant_id`，数据库也会兜住。

    事务在请求结束时提交；异常时回滚（`Database.transaction` 的语义）。
    业务代码不需要自己 commit。
    """
    async with database.transaction(tenant_id=principal.tenant_id) as session:
        yield session


async def get_admin_db(
    request: Request,
    token_service: TokenService = Depends(get_token_service),  # noqa: B008
) -> AsyncIterator[AsyncSession]:
    """管理面专用：不设租户上下文。

    管理接口需要跨租户（看板、台账），因此走 `transaction_bypass_rls`。
    但这只对带 `BYPASSRLS` 的角色真正有效——普通角色会看到 0 行，
    这是有意的安全默认。

    **调用方必须校验平台角色**（`require_platform_role`），
    否则普通用户能通过管理接口枚举所有租户数据。
    """
    principal = await get_current_principal(request, token_service)
    database: Database = request.app.state.database

    from rag_db.repositories import UserRepository

    # 校验平台角色必须在开事务后做（要查用户表），
    # 且用高权限连接（普通角色读 users 表不受 RLS 限制，
    # 因为 users 不在 RLS_TABLES 里）
    async with database.transaction() as session:
        user = await UserRepository(session).get_active_by_id(principal.user_id)
        if user is None or user.platform_role == "none":
            raise PermissionDeniedError("需要平台管理权限")

    async with database.transaction_bypass_rls() as session:
        yield session


# ---------------------------------------------------------------- 权限


def require_platform_role(*allowed: str) -> object:
    """生成一个校验平台角色的依赖。

    用法：
        @router.get(
            "/admin/dashboard",
            dependencies=[Depends(require_platform_role("super_admin"))],
        )

    返回 Callable 而不是直接 Depends：调用方需要指定允许的角色集合。
    """

    async def _check(
        request: Request,
        token_service: TokenService = Depends(get_token_service),  # noqa: B008
    ) -> None:
        from rag_db.repositories import UserRepository

        principal = await get_current_principal(request, token_service)
        database: Database = request.app.state.database
        async with database.transaction() as session:
            user = await UserRepository(session).get_active_by_id(principal.user_id)
            if user is None or user.platform_role not in allowed:
                raise PermissionDeniedError("权限不足")

    return _check
