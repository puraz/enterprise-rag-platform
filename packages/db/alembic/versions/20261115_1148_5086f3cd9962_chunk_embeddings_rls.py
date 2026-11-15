"""chunk_embeddings_rls

Revision ID: 5086f3cd9962
Revises: f04725f245ac
Create Date: 2026-11-15 11:48:00.000000+00:00

## 这次变更做什么

给 A10 新建的 `chunk_embeddings` 旁表补上 RLS 策略与显式授权。

## 为什么这是**安全遗漏**，不只是配置问题

A10 建表时只加了模型定义与索引，忘了两件事：

1. **没有 RLS 策略**：`chunk_embeddings` 带 `tenant_id`，但策略清单
   `RLS_TABLES` 里没有它。后果是应用角色 `rag_app` 能读到**所有租户**的向量。
2. **没有 `FORCE ROW LEVEL SECURITY`**：表所有者仍可绕过。

第 1 点最严重：向量是文档正文的语义指纹。不隔离意味着一个租户
可以通过向量相似度**探测到别的租户有什么内容**——即使看不到原文，
相似度分数本身就泄漏了信息。

## 为什么之前没暴露

`scripts/drill_index_rebuild.py` 与 A10 的测试都用**迁移角色**
（`puraz`，superuser）连库，superuser 无条件绕过 RLS。
只有评估脚本走应用角色时才报"权限不够"——**这正是 RLS 相关 bug
的典型特征：用高权限角色测试永远发现不了**。

修复后 `RLS_TABLES` 已包含这张表，`scripts/verify_rls.py` 会覆盖它。

## 授权策略

- `rag_app`：`SELECT/INSERT/UPDATE/DELETE`（应用正常读写向量）；
- `rag_admin`：同上 + `BYPASSRLS`（管理员跨租户重建索引时必须，
  见 `IndexRebuilder` 的说明）。

**不给 `rag_app` 建表/删表权限**：DDL 只走迁移角色，运行时角色
不该能改表结构。
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

from rag_db.rls import (
    RLS_TABLES,
    create_bypass_statements,
    create_tenant_isolation_policy_statements,
    enable_rls_statements,
)

# 迁移标识。不要手改，Alembic 用它们串成链。
revision: str = '5086f3cd9962'
down_revision: str | None = 'f04725f245ac'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "chunk_embeddings"


def upgrade() -> None:
    # 双重保险：确认这张表确实在 RLS 清单里（漏了就是安全洞）
    if TABLE not in RLS_TABLES:
        raise RuntimeError(
            f"{TABLE} 不在 RLS_TABLES 中。补 RLS 前必须先把表加入清单，"
            "否则这次迁移只加了策略却没人维护它的一致性。"
        )

    # 1. 显式授权（应用角色 + 管理员角色）。GRANT 本身幂等，重复执行无害。
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {TABLE} TO rag_app")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {TABLE} TO rag_admin")

    # 2. 启用 RLS（含 FORCE，让表所有者也不能绕过）
    for statement in enable_rls_statements(TABLE):
        op.execute(statement)

    # 3. 租户隔离策略（USING + WITH CHECK，读写都校验）
    for statement in create_tenant_isolation_policy_statements(TABLE):
        op.execute(statement)

    # 4. 管理员绕过策略
    for statement in create_bypass_statements(TABLE):
        op.execute(statement)


def downgrade() -> None:
    # 只撤销策略与 RLS 开关，**不撤销 GRANT**：
    # 撤销授权可能让仍在用这张表的代码立刻报错，
    # 而"多一份授权"在降级场景下的风险远小于"误伤线上读写"。
    op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {TABLE}")
    op.execute(f"ALTER TABLE {TABLE} NO FORCE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {TABLE} DISABLE ROW LEVEL SECURITY")
