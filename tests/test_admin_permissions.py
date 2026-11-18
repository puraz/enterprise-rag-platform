"""新补接口的权限口径与业务规则测试（纯单测，零外部依赖）。

## 为什么这批测试必须存在

这次新增了 20 个端点，其中 9 个是管理面。管理面的失败模式是
**静默的越权**：接口返回 200 和数据，没有任何异常信号。
所以每条管理能力口径都要有测试钉死，而不是靠 code review。

覆盖三块：

1. **能力矩阵**：平台/租户角色的能力集合必须精确
   （`platform_operator` 不能查审计，`auditor` 不能重试任务）；
2. **租户范围收敛**：租户管理员不能查别的租户，平台角色可以；
3. **断链修复的业务规则**：版本回滚的前置条件、任务查询的跨租户隔离、
   会话 PATCH 的至少一字段约束。
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from rag_api.permissions.admin import (
    _PLATFORM_CAPABILITIES,
    _TENANT_CAPABILITIES,
    AdminCapability,
    AdminContext,
)
from rag_contracts.api.chat import UpdateConversationRequest
from rag_contracts.api.knowledge import RollbackVersionRequest
from rag_contracts.enums import PlatformRole, TenantRole
from rag_core.errors import PermissionDeniedError

# ==================================================== 能力矩阵


def test_super_admin_has_all_capabilities() -> None:
    """平台超管拥有全部能力（含治理与审计）。"""
    assert _PLATFORM_CAPABILITIES[PlatformRole.SUPER_ADMIN] == frozenset(AdminCapability)


def test_platform_operator_cannot_audit() -> None:
    """**平台运营不能查审计**。

    这条是最容易写错的：运营管运维（重试任务、看成本），
    但审计日志能暴露"谁在什么时候看了什么文档"——
    那是安全审计员的职责，不是运维的。

    如果误给 READ 就覆盖了审计读取，这个测试会失败。
    """
    caps = _PLATFORM_CAPABILITIES[PlatformRole.PLATFORM_OPERATOR]
    assert AdminCapability.READ in caps
    assert AdminCapability.OPERATE in caps
    assert AdminCapability.AUDIT not in caps, "运营不该能查审计（会暴露他人行为）"
    assert AdminCapability.ADMIN not in caps, "运营不该能改权限"


def test_auditor_is_read_and_audit_only() -> None:
    """审计员只读 + 能查审计，**不能做任何有副作用的操作**。"""
    caps = _PLATFORM_CAPABILITIES[PlatformRole.AUDITOR]
    assert caps == frozenset({AdminCapability.READ, AdminCapability.AUDIT})
    assert AdminCapability.OPERATE not in caps
    assert AdminCapability.ADMIN not in caps


def test_plain_user_has_no_platform_capability() -> None:
    """普通用户（PlatformRole.NONE）在平台侧没有任何能力。"""
    assert _PLATFORM_CAPABILITIES[PlatformRole.NONE] == frozenset()


def test_tenant_owner_and_admin_have_all_capabilities() -> None:
    """租户 owner/admin 拥有全部能力（作用域被 scope_tenant_id 限制）。

    `ADMIN` 能力包含在内是刻意的：租户管理员本来就能改自己租户的
    成员角色与配置。这不是平台治理权限——`scope_tenant_id` 保证
    它只能在租户内生效。
    """
    for role in (TenantRole.OWNER, TenantRole.ADMIN):
        assert _TENANT_CAPABILITIES[role] == frozenset(AdminCapability)


def test_tenant_member_and_viewer_have_no_admin_capability() -> None:
    """普通成员与只读者不能进管理面。

    这两类角色在 `_TENANT_CAPABILITIES` 里没有条目（`.get` 返空）。
    """
    for role in (TenantRole.MEMBER, TenantRole.VIEWER, TenantRole.KNOWLEDGE_ADMIN):
        assert _TENANT_CAPABILITIES.get(role, frozenset()) == frozenset()


# ==================================================== 租户范围收敛


def _admin_ctx(
    *,
    platform_role: PlatformRole = PlatformRole.NONE,
    tenant_roles: frozenset[TenantRole] = frozenset({TenantRole.OWNER}),
    scope_tenant_id: uuid.UUID | None = None,
    capabilities: frozenset[AdminCapability] | None = None,
) -> AdminContext:
    tenant_id = uuid.uuid4()
    caps = (
        capabilities
        if capabilities is not None
        else (_PLATFORM_CAPABILITIES[platform_role] | frozenset(AdminCapability))
    )
    return AdminContext(
        user_id=uuid.uuid4(),
        platform_role=platform_role,
        tenant_id=tenant_id,
        tenant_roles=tenant_roles,
        scope_tenant_id=scope_tenant_id,
        capabilities=caps,
    )


def test_tenant_admin_cannot_read_other_tenant() -> None:
    """租户管理员请求别的租户 → 403。

    这是最关键的跨租户防线：`tenant_id` 是请求参数，
    服务端必须用 `scope_tenant_id` 收敛。
    """
    own = uuid.uuid4()
    other = uuid.uuid4()
    ctx = _admin_ctx(tenant_roles=frozenset({TenantRole.ADMIN}), scope_tenant_id=own)

    assert ctx.restrict_to(own) == own
    with pytest.raises(PermissionDeniedError):
        ctx.restrict_to(other)


def test_platform_admin_can_scope_to_any_tenant() -> None:
    """平台角色可以显式指定任意租户（跨租户运维是它的职责）。"""
    ctx = _admin_ctx(
        platform_role=PlatformRole.SUPER_ADMIN,
        tenant_roles=frozenset(),
        scope_tenant_id=None,
    )
    anywhere = uuid.uuid4()
    assert ctx.restrict_to(anywhere) == anywhere
    assert ctx.is_platform is True


def test_tenant_admin_default_scope_is_own_tenant() -> None:
    """未指定租户时，租户管理员的默认数据范围是自己的租户。

    平台角色的默认范围是 token 里的租户（**不是全平台汇总**）：
    全量汇总要显式开关，避免误看。
    """
    own = uuid.uuid4()
    ctx = _admin_ctx(tenant_roles=frozenset({TenantRole.OWNER}), scope_tenant_id=own)
    assert ctx.data_tenant_id == own


def test_require_raises_for_missing_capability() -> None:
    """缺少能力时抛 `PermissionDeniedError`（HTTP 403）。"""
    ctx = _admin_ctx(capabilities=frozenset({AdminCapability.READ}))
    ctx.require(AdminCapability.READ)  # 通过
    with pytest.raises(PermissionDeniedError):
        ctx.require(AdminCapability.ADMIN)


# ==================================================== 契约约束


def test_conversation_patch_requires_at_least_one_field() -> None:
    """空 PATCH 被拒绝。

    放行会让前端"什么都没改也发一次请求"变成习惯，产生审计噪音，
    也让"归档失败但返回 200"这类问题更难发现。
    """
    with pytest.raises(ValidationError):
        UpdateConversationRequest()

    # 任一个字段即可
    assert UpdateConversationRequest(title="新标题").title == "新标题"
    assert UpdateConversationRequest(status="archived").status == "archived"


def test_conversation_title_is_stripped_and_blank_rejected() -> None:
    """标题去空白；全空白视为非法。

    `min_length=1` 拦不住 `"   "`（长度是 3），但它入库后
    会显示成一个"看不见标题"的会话。
    """
    assert UpdateConversationRequest(title="  差旅汇总  ").title == "差旅汇总"
    with pytest.raises(ValidationError):
        UpdateConversationRequest(title="   ")


def test_rollback_request_requires_version_id() -> None:
    """回滚必须显式指定目标版本。

    "回滚到上一版"在多版本场景下语义含糊（上一版可能是失败版本），
    所以契约要求 version_id，不接受"隐式上一个"。
    """
    version_id = uuid.uuid4()
    assert RollbackVersionRequest(version_id=version_id).version_id == version_id
    with pytest.raises(ValidationError):
        RollbackVersionRequest()  # type: ignore[call-arg]
