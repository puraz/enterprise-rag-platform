"""认证与授权依赖。

## 依赖链的顺序约束（最关键）

    get_current_principal  ← 只验签，不访问数据库
        ↓
    get_db                  ← 用 principal.tenant_id 开事务并 set_config

**必须先解析 token 再开数据库事务**。反过来的话事务里没有
`app.tenant_id`，RLS 策略会让所有查询返回 0 行 ——
症状是"登录成功但列表永远为空"。

`get_current_principal` 不访问数据库是有意设计：
1. 非法请求在触库前就被拒绝（省连接）；
2. 保证 `get_db` 一定能拿到 tenant_id。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import Depends, Request

from rag_api.deps.app import get_database, get_token_service
from rag_api.identity.tokens import TokenService, bearer_token
from rag_contracts.enums import PlatformRole
from rag_core.db import Database
from rag_core.errors import PermissionDeniedError
from rag_db.repositories import UserRepository


@dataclass(frozen=True, slots=True)
class Principal:
    """当前请求的身份主体。

    只包含**从 token 解析出的可信信息**。业务需要更多信息
    （邮箱、租户角色）时通过仓储查询，不要往这里加字段——
    那会诱导开发者把不可信的外部输入塞进来。
    """

    user_id: uuid.UUID
    tenant_id: uuid.UUID
    token_id: str


async def get_current_principal(
    request: Request,
    token_service: TokenService = Depends(get_token_service),  # noqa: B008
) -> Principal:
    """解析并校验 access token。**不访问数据库**。"""
    credential = bearer_token(request.headers.get("authorization"))
    claims = token_service.verify(credential)
    return Principal(
        user_id=claims.user_id,
        tenant_id=claims.tenant_id,
        token_id=claims.token_id,
    )


async def _load_platform_role(*, database: Database, user_id: uuid.UUID) -> PlatformRole:
    """查用户的平台角色。

    用普通事务（不设租户上下文）：`users` 表不在 RLS_TABLES 里，
    因此不受租户策略限制，不需要管理连接。
    """
    async with database.transaction() as session:
        user = await UserRepository(session).get_active_by_id(user_id)
        if user is None:
            raise PermissionDeniedError("账号不可用")
        return PlatformRole(str(user.platform_role))


def require_platform_role(*allowed: PlatformRole) -> object:
    """生成校验平台角色的依赖。

    用法：

        @router.get(
            "/admin/dashboard",
            dependencies=[Depends(require_platform_role(PlatformRole.SUPER_ADMIN))],
        )

    返回 Callable 而非直接 Depends：调用方需要指定允许的角色集合。
    """

    async def _check(
        principal: Principal = Depends(get_current_principal),  # noqa: B008
        database: Database = Depends(get_database),  # noqa: B008
    ) -> None:
        role = await _load_platform_role(database=database, user_id=principal.user_id)
        if role not in allowed:
            raise PermissionDeniedError("需要平台管理权限")

    return _check
