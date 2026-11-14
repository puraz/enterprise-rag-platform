"""用户与身份映射模型。

关键设计：**用户是全局的，不属于任何租户**。
理由：
- `(issuer, subject)` 必须全局唯一 —— 这是"首登映射本地用户"的前提，
  放租户内会导致同一个人进两个租户就有两条用户记录，审计追不到一起；
- 一个人可以同时属于多个租户（顾问、外部审计员是真实场景）。

"某人属于哪个租户"由 `tenant_memberships` 表表达（见 membership.py）。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from rag_contracts.enums import PlatformRole, UserStatus
from rag_db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """全局用户。

    刻意**不带** tenant_id：用户是跨租户的实体。
    所有"这个用户在这个租户里能干什么"的问题都由 membership 回答。
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(320),  # RFC 5321 上限
        nullable=False,
        comment="邮箱（展示与通知用；不作为登录凭据）",
    )
    display_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="显示名",
    )
    status: Mapped[UserStatus] = mapped_column(
        String(32),
        nullable=False,
        default=UserStatus.ACTIVE,
        server_default=UserStatus.ACTIVE.value,
        index=True,
        comment="用户状态：active/disabled/pending",
    )
    platform_role: Mapped[PlatformRole] = mapped_column(
        String(32),
        nullable=False,
        default=PlatformRole.NONE,
        server_default=PlatformRole.NONE.value,
        comment="平台级角色，与租户内角色正交",
    )
    # 强制下线用：token 里带 iat，早于这个时间的一律失效
    tokens_valid_after: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="早于此时刻签发的 token 全部失效（用于强制下线）",
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后登录时间",
    )

    __table_args__ = (
        # 普通索引（非唯一）：
        # email 不能唯一 —— 不同租户可能有同名邮箱（外部顾问）。
        # 真正的登录身份唯一性在 identity_links 上保证。
        Index("ix_users_email", "email"),
    )

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.status})>"


class IdentityLink(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """外部身份 → 本地用户的映射。

    为什么独立成表而不是把 issuer/subject 挂在 users 上：
    "多 IdP 可配"意味着同一个人可能从 Keycloak 和企业微信两个入口进来，
    应该映射到**同一个**本地用户。独立表天然支持 1:N，
    而 `users.issuer + users.subject` 只能表达 1:1。

    唯一约束 `(issuer, subject)` 是整个认证体系的锚点：
    首次登录时用它查"这个外部身份是否已映射过"，没有就创建本地用户。
    靠唯一索引而不是"先查再插"——并发首登会同时插入，
    唯一索引是唯一能防重复的手段（A7 会写并发首登测试验证）。
    """

    __tablename__ = "identity_links"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="本地用户 ID",
    )
    issuer: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="IdP 的 issuer 标识（OIDC 的 iss 声明）",
    )
    subject: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="IdP 侧的用户唯一标识（OIDC 的 sub 声明）",
    )
    email: Mapped[str | None] = mapped_column(
        String(320),
        nullable=True,
        comment="IdP 返回的邮箱（可能与本地不同，仅作参考）",
    )
    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="最后一次从 IdP 同步属性的时间",
    )

    __table_args__ = (
        # 核心唯一约束：一个外部身份只能映射到一个本地用户
        UniqueConstraint("issuer", "subject", name="uq_identity_links_issuer_subject"),
        # 同一用户在同一 issuer 下不应有多条链接（防止重复绑定）
        UniqueConstraint("user_id", "issuer", name="uq_identity_links_user_issuer"),
    )

    def __repr__(self) -> str:
        return f"<IdentityLink {self.issuer}:{self.subject}>"
