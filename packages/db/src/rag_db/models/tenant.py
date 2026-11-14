"""租户模型。

租户是整个系统的隔离边界：所有业务数据都带 `tenant_id`，
RLS 策略按它过滤（A3-6 建立）。
"""

from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from rag_contracts.enums import TenantStatus
from rag_db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Tenant(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """租户（企业客户）。

    没有 `TenantScopedMixin` —— 它自己就是租户，不隶属于任何租户。
    """

    __tablename__ = "tenants"

    slug: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        comment="URL 友好的租户标识，全局唯一（如 acme）",
    )
    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="租户显示名称",
    )
    status: Mapped[TenantStatus] = mapped_column(
        # 存字符串而非 PG 原生 enum：PG enum 加值要 ALTER TYPE，
        # 在滚动发布时新旧代码可能不认识新值；字符串 + 应用层枚举
        # 更容易做 expand-contract。
        String(32),
        nullable=False,
        default=TenantStatus.ACTIVE,
        server_default=TenantStatus.ACTIVE.value,
        index=True,
        comment="租户状态：active/suspended/deleted",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="租户描述",
    )

    def __repr__(self) -> str:
        return f"<Tenant {self.slug} ({self.status})>"
