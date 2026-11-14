"""租户、用户、成员关系、平台角色枚举。

对应模型：`rag_db.models.tenant` / `user` / `membership`。
一个领域文件对应一组聚合，新增相关枚举直接加在这里。
"""

from __future__ import annotations

from enum import StrEnum


class TenantStatus(StrEnum):
    """租户状态。"""

    ACTIVE = "active"
    """正常可用。"""

    SUSPENDED = "suspended"
    """暂停：可登录但所有业务接口拒绝（欠费/违规）。"""

    DELETED = "deleted"
    """已删除：逻辑删除，数据保留供审计，不可再访问。"""


class UserStatus(StrEnum):
    """用户全局状态。

    注意与 `MembershipStatus` 的区别：这个字段管"这个人还能不能用"，
    `MembershipStatus` 管"这个人在某个租户里是否生效"。
    """

    ACTIVE = "active"

    DISABLED = "disabled"
    """被管理员禁用：已签发的 token 也要能立刻失效（A7 的强制下线）。"""

    PENDING = "pending"
    """已创建但从未登录过（例如邀请后未激活）。"""


class TenantRole(StrEnum):
    """租户内角色，权限从高到低。

    **这是租户内角色，不是平台角色**——平台侧看 `PlatformRole`，两者正交。
    """

    OWNER = "owner"
    """租户所有者：可删除租户、转让所有权。每个租户至少一个。"""

    ADMIN = "admin"
    """租户管理员：管成员、管知识库、管配置。"""

    KNOWLEDGE_ADMIN = "knowledge_admin"
    """知识管理员：只管知识库内容（上传/删除/重建索引），不管成员。"""

    MEMBER = "member"
    """普通成员：可问可查，可上传到自己有权限的空间。"""

    VIEWER = "viewer"
    """只读：只能问和查，不能上传不能改。"""


class PlatformRole(StrEnum):
    """平台级角色（管理端）。

    与 `TenantRole` 正交：平台超管未必是任何租户的成员。
    两者分开是因为「跨租户治理」和「租户内协作」是两种权力来源，
    混成一个字段会导致越权判断写成一团 if-else。
    """

    NONE = "none"
    """普通用户，无平台权限。"""

    SUPER_ADMIN = "super_admin"
    """平台超管：全部管理能力，含高危操作（重建索引、改权限）。"""

    PLATFORM_OPERATOR = "platform_operator"
    """平台运营：看板、供应商配置、成本，不能改权限。"""

    AUDITOR = "auditor"
    """安全审计员：只能读审计日志与安全事件，不能改任何东西。"""


class MembershipStatus(StrEnum):
    """成员关系状态。

    与 `UserStatus` 分开：用户可以是 active，但在这个租户里被移除。
    把两者合成一个字段会导致「禁用某人」和「从租户移除某人」无法区分。
    """

    ACTIVE = "active"

    SUSPENDED = "suspended"
    """在该租户内暂停：保留成员关系但不授予权限（如临时停职）。"""

    REMOVED = "removed"
    """已移出：保留记录供审计，权限立即失效。"""
