"""认证服务：外部身份 → 本地用户映射，并签发本系统 token。

## 首登映射的并发安全（本步最需要小心的地方）

首次登录的流程是：

    1. 按 (issuer, subject) 查 identity_links
    2. 没查到 → 创建 user + identity_link
    3. 签发 token

并发首登时（用户连点两次、或前端重试），两个请求都会走到第 2 步。
如果实现是"先查再插"，会产生**两个本地用户**，后续登录随机命中其中一个
——表现为"用户的文档时有时无"。

正确做法：直接插入并依赖 `uq_identity_links_issuer_subject` 唯一约束。
冲突时捕获 `IntegrityError`，重新查询已有记录。

A3-1 的架构里已经有这个唯一索引，这里把它用起来。

## 租户选择

用户首次登录时可能还不属于任何租户。这里按顺序尝试：
1. IdP 声明的 `tenant_hint`（如 Keycloak 的 organization claim）；
2. 用户已有的第一个 active 成员关系；
3. 都没有 → 拒绝登录并提示"请联系管理员开通"。

**不自动创建租户**：自动建租户会让任何人登录即拥有一个租户，
积分/配额完全失控。开通租户必须是管理动作。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.identity.provider import ExternalIdentity
from rag_contracts.enums import MembershipStatus, UserStatus
from rag_core.errors import AuthenticationError, PermissionDeniedError
from rag_core.logging import get_logger
from rag_db.models.user import User
from rag_db.repositories import (
    MembershipRepository,
    TenantRepository,
    UserRepository,
)

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    """登录成功后的结果。"""

    user_id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    display_name: str
    created: bool
    """是否本次调用新建了用户（首登）。"""


class AuthService:
    """登录与用户映射。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._users = UserRepository(session)
        self._memberships = MembershipRepository(session)
        self._tenants = TenantRepository(session)

    async def resolve_login(self, identity: ExternalIdentity) -> AuthenticatedUser:
        """把外部身份解析成本地用户 + 当前租户。

        Raises:
            AuthenticationError: 用户被禁用
            PermissionDeniedError: 用户不属于任何可用租户
        """
        link = await self._users.find_by_identity(issuer=identity.issuer, subject=identity.subject)

        if link is not None:
            user = await self._users.get_by_id(link.user_id)
            if user is None:
                # 数据不一致：映射存在但用户被删（不该发生，FK 是 CASCADE）。
                # 不静默修复——那是掩盖数据损坏。让运维看到。
                log.error(
                    "identity_link_dangling",
                    user_id=str(link.user_id),
                    issuer=identity.issuer,
                )
                raise AuthenticationError("账号数据异常，请联系管理员")
            if user.status != UserStatus.ACTIVE.value:
                raise AuthenticationError("账号已被禁用")
            created = False
        else:
            user, created = await self._create_user_with_link(identity)

        tenant_id = await self._resolve_tenant(user.id, identity)
        return AuthenticatedUser(
            user_id=user.id,
            tenant_id=tenant_id,
            email=user.email,
            display_name=user.display_name,
            created=created,
        )

    async def _create_user_with_link(self, identity: ExternalIdentity) -> tuple[User, bool]:
        """创建用户与身份映射。

        **并发安全**：两个请求同时首登时，第二个会在
        `uq_identity_links_issuer_subject` 上撞唯一约束。
        捕获后重新查询，返回已存在的用户 —— 而不是报错或建出两个用户。

        注意这里**先建 user 再建 link**：如果 link 冲突说明别人已经建好了
        user+link，此时我们刚建的 user 会变成孤儿。所以要在同一个事务里，
        冲突后回滚整个事务，再重查。
        """
        email = identity.email or f"{identity.subject}@{identity.issuer}"
        display_name = identity.display_name or identity.subject

        try:
            # savepoint：冲突时只回滚这一段，外层事务还能继续用
            async with self._session.begin_nested():
                user = await self._users.create_user(
                    email=email,
                    display_name=display_name,
                    status=UserStatus.ACTIVE,
                )
                await self._users.create_link(
                    user_id=user.id,
                    issuer=identity.issuer,
                    subject=identity.subject,
                    email=identity.email,
                )
            return user, True
        except IntegrityError:
            # 并发首登：另一个请求已经建好了，重查即可
            link = await self._users.find_by_identity(
                issuer=identity.issuer, subject=identity.subject
            )
            if link is None:
                # 不是并发冲突，而是别的约束（如 email 长度）——如实抛出
                raise
            existing = await self._users.get_by_id(link.user_id)
            if existing is None:
                raise AuthenticationError("账号数据异常，请联系管理员") from None
            log.info(
                "concurrent_first_login_resolved",
                issuer=identity.issuer,
                user_id=str(existing.id),
            )
            return existing, False

    async def _resolve_tenant(self, user_id: uuid.UUID, identity: ExternalIdentity) -> uuid.UUID:
        """确定本次登录激活哪个租户。"""
        memberships = await self._memberships.list_for_user(user_id)
        active = [m for m in memberships if m.status == MembershipStatus.ACTIVE.value]

        if not active:
            raise PermissionDeniedError("尚未加入任何租户，请联系管理员开通")

        # IdP 给了租户提示时优先用它（多租户 SSO 场景）
        if identity.tenant_hint:
            for membership in active:
                tenant = await self._tenants.get_active_by_id(membership.tenant_id)
                if tenant is not None and tenant.slug == identity.tenant_hint:
                    return tenant.id
            # 提示的租户不在成员关系里：不静默回退，
            # 否则用户以为进了 A 租户实际进了 B
            raise PermissionDeniedError("您不属于指定的租户")

        # 默认取第一个成员关系对应的可用租户
        for membership in active:
            tenant = await self._tenants.get_active_by_id(membership.tenant_id)
            if tenant is not None:
                return tenant.id

        raise PermissionDeniedError("所属租户已停用，请联系管理员")
