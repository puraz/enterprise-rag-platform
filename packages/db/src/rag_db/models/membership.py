"""租户成员关系模型：谁在哪个租户、以什么角色。

这是"全局用户"与"多租户"之间的连接表，也是鉴权的核心查询点：
每次请求都要回答"这个用户在这个租户里是什么角色、还active吗"。

因为查询频率极高，索引设计比表结构本身更重要。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from rag_contracts.enums import MembershipStatus, TenantRole
from rag_db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class TenantMembership(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """用户在某租户内的成员关系与角色。

    `unique(tenant_id, user_id)` 保证一个人在一个租户里只有一条记录。
    没有这个约束时，"重复邀请"会产生两条记录，权限判断取哪条全看
    查询顺序——表现为"权限时有时无"，极难排查。
    """

    __tablename__ = "tenant_memberships"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        comment="租户 ID",
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="用户 ID",
    )
    role: Mapped[TenantRole] = mapped_column(
        String(32),
        nullable=False,
        default=TenantRole.MEMBER,
        server_default=TenantRole.MEMBER.value,
        comment="租户内角色：owner/admin/knowledge_admin/member/viewer",
    )
    status: Mapped[MembershipStatus] = mapped_column(
        String(32),
        nullable=False,
        default=MembershipStatus.ACTIVE,
        server_default=MembershipStatus.ACTIVE.value,
        comment="成员关系状态，与用户全局状态独立",
    )
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="加入时间（数据库侧生成，避免多进程时钟偏差）",
    )

    __table_args__ = (
        # 一人一租户一条记录
        UniqueConstraint("tenant_id", "user_id", name="uq_tenant_memberships_tenant_user"),
        # 鉴权主查询：WHERE tenant_id = ? AND user_id = ? AND status = 'active'
        # 复合索引第一列是 tenant_id，正好匹配 RLS 的强制过滤条件
        Index("ix_tenant_memberships_tenant_user", "tenant_id", "user_id"),
        # 用户视角的"我属于哪些租户"（登录后切换租户的列表）
        Index("ix_tenant_memberships_user", "user_id"),
        # 租户视角的成员列表（管理端），按角色过滤
        Index("ix_tenant_memberships_tenant_role", "tenant_id", "role"),
    )

    def __repr__(self) -> str:
        return f"<Membership t={self.tenant_id} u={self.user_id} {self.role}>"
