"""权限上下文与 ACL 过滤条件构建。

## 为什么 ACL 过滤必须在 SQL 里

`docs/PROMPT.md` 要求"过滤条件服务端强制注入 + PostgreSQL RLS 兜底"。

两层分工：

| 层 | 能力 | 局限 |
|---|---|---|
| RLS | 租户级隔离（`tenant_id = current_setting(...)`） |
  **无法表达"这个用户能不能看这个文档"**：RLS 里拿不到用户角色/部门/密级 |
| **ACL（本模块）** | 文档级、密级、可见范围 | 必须是 SQL 条件，因为它要参与 JOIN 与排序 |

所以 RLS 防"跨租户"，ACL 防"同租户内越权"。缺任何一层都有洞：

- 只有 RLS：同租户的实习生能看到所有文档；
- 只有 ACL：一旦某个 SQL 漏写条件就跨租户泄漏。

## 为什么不能"查出来再过滤"

常见错误：

```python
docs = await repo.list_all(tenant_id)          # 查全部
visible = [d for d in docs if can_read(d)]     # Python 里过滤
```

三个问题：
1. **分页错乱**：limit 20 查出 20 条，过滤后剩 3 条，前端以为只有 3 条；
2. **信息泄漏**：越权文档已经进了应用内存，任何一处日志/异常都可能带出标题；
3. **性能**：查 1000 条只返回 3 条，浪费 99.7% 的 I/O。

所以过滤条件必须进 `WHERE`，在数据库里做。

## 四层隔离

`docs/PROMPT.md` 要求"四层隔离：租户→空间/部门→密级→字段级脱敏"：

| 层 | 实现位置 |
|---|---|
| 租户 | RLS + 每处 SQL 的 `tenant_id` |
| 空间/部门 | `access_filter()` → `visibility` + 成员关系 |
| 密级 | `access_filter()` → `confidentiality <= user_level` |
| 字段级脱敏 | 序列化层（`redact_fields()`） |

本模块负责前两层与密级；字段级在 contracts 的响应模型里做。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import ColumnElement, or_

from rag_contracts.enums import ConfidentialityLevel, KnowledgeBaseVisibility, TenantRole
from rag_core.errors import PermissionDeniedError, TenantMismatchError

# 密级的大小顺序（用于 `<=` 比较）。
# 用显式映射而不是枚举定义顺序：枚举成员的书写顺序不该被当作业务语义，
# 那样加一个中间级别就得小心插入位置。
LEVEL_RANK: dict[ConfidentialityLevel, int] = {
    ConfidentialityLevel.PUBLIC: 0,
    ConfidentialityLevel.INTERNAL: 1,
    ConfidentialityLevel.CONFIDENTIAL: 2,
    ConfidentialityLevel.SECRET: 3,
}

# 角色 → 可访问的最高密级
ROLE_MAX_LEVEL: dict[TenantRole, ConfidentialityLevel] = {
    TenantRole.OWNER: ConfidentialityLevel.SECRET,
    TenantRole.ADMIN: ConfidentialityLevel.SECRET,
    TenantRole.KNOWLEDGE_ADMIN: ConfidentialityLevel.CONFIDENTIAL,
    TenantRole.MEMBER: ConfidentialityLevel.INTERNAL,
    # 只读者能看到内部级别：公开与内部文档都该能读，
    # 否则"只读"变成"什么都看不到"
    TenantRole.VIEWER: ConfidentialityLevel.INTERNAL,
}

# 能管理知识库内容的角色
_KB_WRITE_ROLES = frozenset({TenantRole.OWNER, TenantRole.ADMIN, TenantRole.KNOWLEDGE_ADMIN})
# 能管理租户成员的角色
_MEMBER_MANAGE_ROLES = frozenset({TenantRole.OWNER, TenantRole.ADMIN})


@dataclass(frozen=True, slots=True)
class PermissionContext:
    """一次请求的权限上下文。

    **只从 token 与数据库构造**，绝不接受客户端传入的角色/密级。

    字段：
    - `tenant_id` / `user_id`：来自 token；
    - `roles`：来自 `tenant_memberships`（数据库）；
    - `department_ids`：用户所属部门（A7 的 `visibility=department` 用）。
    """

    user_id: uuid.UUID
    tenant_id: uuid.UUID
    roles: frozenset[TenantRole] = field(default_factory=frozenset)
    department_ids: frozenset[uuid.UUID] = field(default_factory=frozenset)

    @property
    def max_confidentiality(self) -> ConfidentialityLevel:
        """可访问的最高密级。

        取用户所有角色里**最低**的那个上限：
        一个人同时是 owner（SECRET）和 viewer（INTERNAL）时，
        按 viewer 算——降级授权不该因为"还挂着高权限角色"而被绕过。

        没有任何角色时返回 PUBLIC（只能看公开内容）。
        """
        if not self.roles:
            return ConfidentialityLevel.PUBLIC
        highest_rank = min(LEVEL_RANK[ROLE_MAX_LEVEL[role]] for role in self.roles)
        for level, rank in LEVEL_RANK.items():
            if rank == highest_rank:
                return level
        return ConfidentialityLevel.PUBLIC  # pragma: no cover

    def to_sql_rank(self) -> int:
        """当前用户的最大密级对应的数值（供 SQL 比较）。"""
        return LEVEL_RANK[self.max_confidentiality]

    def has_any_role(self, *roles: TenantRole) -> bool:
        return bool(self.roles & set(roles))

    def require_role(self, *roles: TenantRole, action: str = "此操作") -> None:
        """要求具备其中之一角色。"""
        if not self.has_any_role(*roles):
            raise PermissionDeniedError(f"需要 {self._role_names(roles)} 权限才能{action}")

    def require_kb_write(self) -> None:
        """要求能写知识库内容。"""
        self.require_role(*_KB_WRITE_ROLES, action="上传或修改文档")

    def require_member_manage(self) -> None:
        """要求能管理租户成员。"""
        self.require_role(*_MEMBER_MANAGE_ROLES, action="管理租户成员")

    @staticmethod
    def _role_names(roles: tuple[TenantRole, ...]) -> str:
        return " / ".join(role.value for role in roles)


def assert_same_tenant(*, token_tenant_id: uuid.UUID, resource_tenant_id: uuid.UUID) -> None:
    """校验资源属于 token 里的租户。

    **这是最后一道业务层检查**（RLS 是第一道）：即使仓储漏写了
    `WHERE tenant_id`，这里也能拦住跨租户访问。

    severity=CRITICAL：正常流程不会出现，出现即越权尝试。
    """
    if token_tenant_id != resource_tenant_id:
        raise TenantMismatchError()


def knowledge_base_visibility_filter(
    ctx: PermissionContext,
) -> ColumnElement[bool]:
    """构造知识库的可见性过滤条件。

    三类可见性：
    - `tenant`：租户内所有成员可见；
    - `private`：仅创建者可见；
    - `department`：用户在指定部门内可见。

    **`department` 的实现在 A7 只是骨架**：`department_ids` 来自
    `PermissionContext`，而部门表的建模推迟（当前没有 departments 表）。
    这里先按"用户无部门则不匹配"处理，行为安全（看不到而非看太多）。
    """
    from rag_db.models.knowledge import KnowledgeBase

    conditions: list[ColumnElement[bool]] = [
        # 租户内公开：所有成员可见
        KnowledgeBase.visibility == KnowledgeBaseVisibility.TENANT.value,
        # 私有：只有创建者
        (
            (KnowledgeBase.visibility == KnowledgeBaseVisibility.PRIVATE.value)
            & (KnowledgeBase.created_by == ctx.user_id)
        ),
    ]

    # 部门可见：有部门归属且交集非空。
    # 没有部门时该分支不匹配（安全默认）。
    if ctx.department_ids:
        conditions.append(
            (KnowledgeBase.visibility == KnowledgeBaseVisibility.DEPARTMENT.value)
            # 部门的关联表尚未建模；用 created_by 兜底会让权限过宽，
            # 所以这里显式不匹配，等 A7 后续补 departments 表。
            & (KnowledgeBase.id.is_(None))
        )

    return or_(*conditions)


def document_access_filter(ctx: PermissionContext) -> ColumnElement[bool]:
    """构造文档的可读过滤条件。

    文档可读需要**同时**满足：
    1. 所属知识库对该用户可见（`knowledge_base_visibility_filter`）；
    2. 文档密级 ≤ 用户的密级上限。

    **必须用 AND 连接**：用 OR 会让"库可见但密级不够"的文档泄漏。
    """
    from rag_db.models.document import Document

    confidentiality_rank = _confidentiality_sql_rank(Document.confidentiality)

    return (
        Document.deleted_at.is_(None)
        & (confidentiality_rank <= ctx.to_sql_rank())
        # 库可见性用子查询表达（避免依赖 JOIN 别名）
        & Document.knowledge_base_id.in_(_visible_kb_subquery(ctx))
    )


def _visible_kb_subquery(ctx: PermissionContext) -> Any:
    """可见知识库 ID 的子查询。"""
    from sqlalchemy import select

    from rag_db.models.knowledge import KnowledgeBase

    return select(KnowledgeBase.id).where(
        KnowledgeBase.tenant_id == ctx.tenant_id,
        knowledge_base_visibility_filter(ctx),
    )


def _confidentiality_sql_rank(column: object) -> ColumnElement[int]:
    """把密级列转成可比较的数值。

    用 SQL `CASE WHEN` 而不是依赖枚举的字母序：
    `confidential` < `internal` < `public` < `secret` 是按字母排的，
    与"敏感程度"完全无关。按字母比较会让 `secret` 被当成最低级。

    **这是最容易写错的地方**：如果直接 `Document.confidentiality <= 'internal'`
    字符串比较，结果看起来"能跑"，但权限完全错了且不报错。
    """
    from sqlalchemy import case

    from rag_db.models.document import Document

    return case(
        {
            ConfidentialityLevel.PUBLIC.value: LEVEL_RANK[ConfidentialityLevel.PUBLIC],
            ConfidentialityLevel.INTERNAL.value: LEVEL_RANK[ConfidentialityLevel.INTERNAL],
            ConfidentialityLevel.CONFIDENTIAL.value: LEVEL_RANK[ConfidentialityLevel.CONFIDENTIAL],
            ConfidentialityLevel.SECRET.value: LEVEL_RANK[ConfidentialityLevel.SECRET],
        },
        value=Document.confidentiality,
        # 未知值按最高密级处理（安全默认：不认识的密级当作最敏感）
        else_=LEVEL_RANK[ConfidentialityLevel.SECRET],
    )


def assert_document_readable(
    *, document_confidentiality: str, kb_visible: bool, ctx: PermissionContext
) -> None:
    """单条文档的可读性检查（用于详情接口）。

    列表用 SQL 过滤，详情用这个方法——两者必须用**同一套规则**，
    否则会出现"列表里看不到但直接访问 URL 能看到"的越权。

    Raises:
        NotFoundError: 不可读时返回 404 而非 403（不泄漏存在性）。
    """
    from rag_core.errors import NotFoundError

    try:
        level = ConfidentialityLevel(document_confidentiality)
    except ValueError:
        # 未知密级：当作最高密级（安全默认）
        level = ConfidentialityLevel.SECRET

    if not kb_visible or LEVEL_RANK[level] > ctx.to_sql_rank():
        raise NotFoundError("文档不存在")


def confidentiality_filter_for_chunks(ctx: PermissionContext) -> ColumnElement[bool]:
    """chunk 检索的密级与可见性过滤条件。

    **检索是最危险的路径**：它返回的是文档内容本身（不只是元数据）。
    如果检索不做 ACL 过滤，用户可以通过提问把无权文档的内容"问出来"——
    这比直接返回文档更隐蔽，因为泄漏发生在答案里。

    过滤条件（三个 AND）：
    1. 文档未删除；
    2. 文档密级 ≤ 用户密级上限；
    3. 文档所属知识库对该用户可见。

    用 `Chunk.document_id.in_(子查询)` 而不是 JOIN：
    JOIN 会改变向量检索的查询计划——pgvector 的 HNSW 索引在
    "先扫描索引再过滤"时表现最好，JOIN 可能让规划器改选顺序扫描。
    """
    from sqlalchemy import select

    from rag_db.models.chunk import Chunk
    from rag_db.models.document import Document

    confidentiality_rank = _confidentiality_sql_rank(Document.confidentiality)

    visible_documents = select(Document.id).where(
        Document.tenant_id == ctx.tenant_id,
        Document.deleted_at.is_(None),
        confidentiality_rank <= ctx.to_sql_rank(),
        Document.knowledge_base_id.in_(_visible_kb_subquery(ctx)),
    )
    return Chunk.document_id.in_(visible_documents)


def verify_citations_visible(
    *, cited_document_ids: list[uuid.UUID], visible_document_ids: set[uuid.UUID]
) -> list[uuid.UUID]:
    """生成后二次校验：引用的文档必须都在可见集内。

    返回**越界的文档 ID**（空列表表示全部合法）。

    为什么需要这一步（SQL 已经过滤了）：
    - 模型可能"记错"了来源（把 A 文档的内容归到 B 文档）；
    - 上下文压缩/父块回填可能引入未过滤的内容；
    - 多查询融合时某个子查询的过滤条件可能写漏。

    任何一条都会导致"引用了用户无权看的文档"，这是最严重的泄漏形态
    ——比直接返回文档更隐蔽，因为它嵌在答案里。
    """
    return [doc_id for doc_id in cited_document_ids if doc_id not in visible_document_ids]
