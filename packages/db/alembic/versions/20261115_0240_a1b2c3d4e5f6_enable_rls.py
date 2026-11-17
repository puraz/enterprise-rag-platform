"""启用 RLS 与租户隔离策略。

Revision ID: a1b2c3d4e5f6
Revises: 9332991f6731
Create Date: 2026-11-15

## 这一步做了什么

1. 给所有租户内表 `ENABLE` + `FORCE` ROW LEVEL SECURITY；
2. 为每张表创建 `tenant_isolation` 策略，条件为
   `tenant_id = current_setting('app.tenant_id', true)::uuid`；
3. 给 `rag_admin` 角色（若存在）创建 bypass 策略，用于合法的跨租户路径。

## 为什么必须用非超级用户连库

**superuser 无条件绕过所有 RLS 策略，且 `FORCE` 也无效。**
当前开发环境的应用连接串用的是 `puraz`（超级用户），
所以启用 RLS 后如果继续用 puraz 连库，策略**完全不起作用且无任何提示**。

应用必须改用 `rag_app`（NOSUPERUSER NOBYPASSRLS）。迁移本身用 puraz 执行
（需要 DDL 权限），这是正确的分工：**迁移高权限、运行时低权限**。

## 回滚

downgrade 只禁用 RLS，不删策略（策略可复用，重新 enable 即可生效）。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from rag_db.rls import (
    AUDIT_TABLE,
    RLS_TABLES,
    audit_policy_statements,
    create_tenant_isolation_policy_statements,
    enable_rls_statements,
)

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "9332991f6731"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_exists(table: str) -> bool:
    """该表在当前迁移时点是否已存在。

    **必须判断，不能直接遍历 `RLS_TABLES`**：那个元组是"当前所有租户表"，
    包含了后续迁移才创建的表（chunk_embeddings / conversations / messages）。
    在全新数据库上按修订顺序重放时，直接 ALTER 会报 relation does not exist，
    而这个错误只在"从零建库"时出现（增量开发的库早就存在这些表了）。

    后建表由各自的迁移负责 ENABLE RLS（如 `chunk_embeddings_rls`、
    `conversations_messages`），这里跳过不会有安全缺口。
    """
    exists = op.get_bind().execute(
        sa.text("SELECT to_regclass(:table_name)"), {"table_name": table}
    ).scalar()
    return exists is not None


def upgrade() -> None:
    for table in RLS_TABLES:
        if not _table_exists(table):
            continue
        for statement in enable_rls_statements(table):
            op.execute(statement)
        for statement in create_tenant_isolation_policy_statements(table):
            op.execute(statement)

    # 审计日志单独处理（tenant_id 可为 NULL）
    if _table_exists(AUDIT_TABLE):
        for statement in enable_rls_statements(AUDIT_TABLE):
            op.execute(statement)
        for statement in audit_policy_statements():
            op.execute(statement)


def downgrade() -> None:
    for table in (*RLS_TABLES, AUDIT_TABLE):
        if not _table_exists(table):
            continue
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
