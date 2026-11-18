"""权限层：RBAC 角色校验、文档级 ACL、密级过滤。

## 四层隔离（`docs/PROMPT.md` 要求）

| 层 | 模块 | 实现 |
|---|---|---|
| 租户 | RLS（A3-6）+ `tenant_id` 参数 | 数据库层兜底 |
| 空间/部门 | `acl.knowledge_base_visibility_filter` | SQL 条件 |
| 密级 | `acl.confidentiality_filter_for_chunks` | SQL 条件 |
| 字段级脱敏 | 序列化层 | 响应模型控制 |

## 使用约定

- **列表/检索**：把 `document_access_filter(ctx)` 加进查询的 `WHERE`
- **详情**：`assert_document_readable(...)`（与列表同一套规则）
- **写入**：`ctx.require_kb_write()` / `require_member_manage()`
- **审计**：`AuditService(ctx).record(...)` 显式调用
"""

from rag_api.permissions.acl import (
    LEVEL_RANK,
    ROLE_MAX_LEVEL,
    PermissionContext,
    assert_document_readable,
    assert_same_tenant,
    confidentiality_filter_for_chunks,
    document_access_filter,
    knowledge_base_visibility_filter,
    verify_citations_visible,
)
from rag_api.permissions.admin import (
    AdminCapability,
    AdminContext,
    get_admin_context,
    get_admin_db,
    require_admin_capability,
    to_permission_context,
)
from rag_api.permissions.deps import (
    ensure_own_resource,
    get_permission_context,
    require_kb_write,
    require_member_manage,
    require_tenant_role,
)

__all__ = [
    "LEVEL_RANK",
    "ROLE_MAX_LEVEL",
    "AdminCapability",
    "AdminContext",
    "PermissionContext",
    "assert_document_readable",
    "assert_same_tenant",
    "confidentiality_filter_for_chunks",
    "document_access_filter",
    "ensure_own_resource",
    "get_admin_context",
    "get_admin_db",
    "get_permission_context",
    "knowledge_base_visibility_filter",
    "require_admin_capability",
    "require_kb_write",
    "require_member_manage",
    "require_tenant_role",
    "to_permission_context",
    "verify_citations_visible",
]
