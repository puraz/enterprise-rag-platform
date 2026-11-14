"""租户 / 用户 / 成员关系仓储。

仓储的职责边界：
- 只做数据访问，**不含业务规则**（"只有 owner 能删租户"这类判断在 service 层）；
- 接收 `AsyncSession`，不自己建连接、不开事务（由 service 决定事务边界）；
- 返回 ORM 对象或明确的标量，不返回裸 Row 让上层猜。

**所有查询都必须带 tenant_id**（涉及租户内资源的）。这条靠 RLS 兜底（A3-6），
但仓储层自己也要写对——RLS 是最后一道防线，不是唯一防线。
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums import MembershipStatus, TenantStatus, UserStatus
from rag_db.models.membership import TenantMembership
from rag_db.models.tenant import Tenant
from rag_db.models.user import IdentityLink, User


class TenantRepository:
    """租户数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_slug(self, slug: str) -> Tenant | None:
        """按 slug 查租户。

        用 limit(1) 而不是 scalar_one_or_none：有唯一索引保证最多一条，
        但显式 limit 让执行计划确定（某些版本会因统计信息选择不同计划）。
        """
        result = await self._session.execute(select(Tenant).where(Tenant.slug == slug).limit(1))
        return result.scalar_one_or_none()

    async def get_by_id(self, tenant_id: uuid.UUID) -> Tenant | None:
        result = await self._session.execute(select(Tenant).where(Tenant.id == tenant_id).limit(1))
        return result.scalar_one_or_none()

    async def get_active_by_id(self, tenant_id: uuid.UUID) -> Tenant | None:
        """只返回 active 状态的租户。

        鉴权路径必须用这个而不是 `get_by_id`：暂停/删除的租户不该能访问业务。
        把它做成独立方法，是为了让调用点一眼看出"这里做了状态检查"。
        """
        result = await self._session.execute(
            select(Tenant)
            .where(Tenant.id == tenant_id, Tenant.status == TenantStatus.ACTIVE)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def create(self, *, slug: str, name: str, description: str | None = None) -> Tenant:
        """创建租户。

        slug 冲突时让唯一索引抛 IntegrityError，**不在这里先查再插**：
        先查再插有竞态窗口（两个请求同时查到"不存在"然后都插入），
        只有唯一索引能真正防住。上层捕获 IntegrityError 转成业务错误。
        """
        tenant = Tenant(slug=slug, name=name, description=description)
        self._session.add(tenant)
        await self._session.flush()  # flush 才能拿到数据库生成的字段
        return tenant

    async def list_all(self, *, limit: int = 50, after_id: uuid.UUID | None = None) -> list[Tenant]:
        """管理端列出所有租户（keyset 分页）。

        不用 OFFSET：数据量大时 OFFSET 越翻越慢，且并发插入会导致
        重复/漏读。keyset 用"上一页最后一个 ID"定位。
        """
        stmt = select(Tenant).order_by(Tenant.id).limit(limit)
        if after_id is not None:
            stmt = stmt.where(Tenant.id > after_id)
        result = await self._session.execute(stmt)
        return list(result.scalars())


class UserRepository:
    """用户与身份映射数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        result = await self._session.execute(select(User).where(User.id == user_id).limit(1))
        return result.scalar_one_or_none()

    async def get_active_by_id(self, user_id: uuid.UUID) -> User | None:
        """只返回 active 用户。禁用用户的 token 不能继续用。"""
        result = await self._session.execute(
            select(User).where(User.id == user_id, User.status == UserStatus.ACTIVE).limit(1)
        )
        return result.scalar_one_or_none()

    async def find_by_identity(self, *, issuer: str, subject: str) -> IdentityLink | None:
        """按外部身份查映射。

        这是登录链路的第一跳：`(issuer, subject)` 命中就走登录，
        没命中就创建新用户。唯一索引保证并发首登不会建出两个用户。
        """
        result = await self._session.execute(
            select(IdentityLink)
            .where(IdentityLink.issuer == issuer, IdentityLink.subject == subject)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_links(self, user_id: uuid.UUID) -> list[IdentityLink]:
        """列出某用户绑定的所有外部身份（多 IdP 场景）。"""
        result = await self._session.execute(
            select(IdentityLink)
            .where(IdentityLink.user_id == user_id)
            .order_by(IdentityLink.issuer)
        )
        return list(result.scalars())

    async def create_user(
        self,
        *,
        email: str,
        display_name: str,
        status: UserStatus = UserStatus.ACTIVE,
    ) -> User:
        user = User(email=email, display_name=display_name, status=status)
        self._session.add(user)
        await self._session.flush()
        return user

    async def create_link(
        self,
        *,
        user_id: uuid.UUID,
        issuer: str,
        subject: str,
        email: str | None = None,
    ) -> IdentityLink:
        link = IdentityLink(user_id=user_id, issuer=issuer, subject=subject, email=email)
        self._session.add(link)
        await self._session.flush()
        return link


class MembershipRepository:
    """租户成员关系数据访问。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_active(
        self, *, tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> TenantMembership | None:
        """取生效中的成员关系。

        鉴权高频路径。复合索引 `(tenant_id, user_id)` 直接支撑这个查询。
        """
        result = await self._session.execute(
            select(TenantMembership)
            .where(
                TenantMembership.tenant_id == tenant_id,
                TenantMembership.user_id == user_id,
                TenantMembership.status == MembershipStatus.ACTIVE,
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_for_user(self, user_id: uuid.UUID) -> list[TenantMembership]:
        """用户所属的租户列表（登录后切换租户用）。"""
        result = await self._session.execute(
            select(TenantMembership)
            .where(
                TenantMembership.user_id == user_id,
                TenantMembership.status == MembershipStatus.ACTIVE,
            )
            .order_by(TenantMembership.created_at)
        )
        return list(result.scalars())

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        role: str,
    ) -> TenantMembership:
        membership = TenantMembership(tenant_id=tenant_id, user_id=user_id, role=role)
        self._session.add(membership)
        await self._session.flush()
        return membership
