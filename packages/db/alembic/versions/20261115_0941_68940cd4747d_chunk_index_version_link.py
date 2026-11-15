"""chunk_index_version_link

Revision ID: 68940cd4747d
Revises: a1b2c3d4e5f6
Create Date: 2026-11-15 09:41:09.858878+00:00

## 这次变更做什么

给 `chunks` 加 `index_version_id`（可空 FK），让同一份内容能在库里
同时存在**两套向量**——这是 A10"零停机重建索引"的前提。

没有这一列时的困境：
- 向量只有 `chunks.embedding` 一列；
- 重建（换 embedding 模型 / 调 HNSW 参数）必须覆盖旧向量；
- 回填期间新旧向量混杂，查询结果不可解释，或者必须停服。

加列之后：
    新 chunk 双写两个版本 → 后台回填旧 chunk → UPDATE 切 ACTIVE → 回收旧版本

## 为什么这个迁移是安全的（expand-contract 的 expand 阶段）

1. **新列可空**：旧代码插入不带这一列的 chunk 不会失败。
2. **无回填需求**：只有 `index_version_id IS NULL` 的语义是"未向量化"，
   而库里现存的 chunk 本来就没有向量（A9 只做切分）。滚动发布期间
   新旧代码都能跑。
3. **只加不删**：没有 ALTER COLUMN TYPE、没有 DROP，不重写表。
4. **ADD COLUMN 带 NULL 默认值**：PG 11+ 只改 catalog，不重写表，
   对热表几乎无锁。

## 锁与耗时（真实数据量下要评估）

- `ALTER TABLE ... ADD COLUMN`（可空、无默认）：ACCESS EXCLUSIVE 锁但瞬间完成。
- `CREATE INDEX`（非 CONCURRENTLY）：会阻塞写入。
  **本迁移用的就是非 CONCURRENTLY**，因为：
  - 迁移在同一事务里（Alembic 默认），CONCURRENTLY 不能在事务内执行；
  - 部分索引 `WHERE index_version_id IS NOT NULL` 在存量数据上匹配 0 行，
    构建耗时接近 0。

**如果生产环境 `chunks` 已有大量非空 `index_version_id` 数据**，
这里必须改成 `CREATE INDEX CONCURRENTLY` 并把迁移拆成独立的
非事务迁移（`op.execute("COMMIT")` 或 `transaction_per_migration=False`）。
本步骤库里 chunk 数为 0，故未做此改造——但注释留在这里，
避免后人照着这个迁移在有数据的库上直接跑。
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# 迁移标识。不要手改，Alembic 用它们串成链。
revision: str = '68940cd4747d'
down_revision: str | None = 'a1b2c3d4e5f6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 可空列：旧代码插入不受影响（expand 阶段的核心）
    op.add_column(
        'chunks',
        sa.Column(
            'index_version_id',
            sa.UUID(),
            nullable=True,
            comment='索引版本归属（NULL = 尚未向量化/未纳入索引）',
        ),
    )
    # 部分索引：未向量化的 chunk 不属于任何版本，不占索引空间
    op.create_index(
        'ix_chunks_index_version',
        'chunks',
        ['index_version_id'],
        unique=False,
        postgresql_where=sa.text('index_version_id IS NOT NULL'),
    )
    op.create_foreign_key(
        op.f('fk_chunks_index_version_id_index_versions'),
        'chunks',
        'index_versions',
        ['index_version_id'],
        ['id'],
        ondelete='SET NULL',
    )


def downgrade() -> None:
    # 先删约束与索引，再删列（顺序不能反：
    # 索引依赖列，直接删列会让 PG 连带删除索引但约束会报错）
    op.drop_constraint(
        op.f('fk_chunks_index_version_id_index_versions'), 'chunks', type_='foreignkey'
    )
    op.drop_index(
        'ix_chunks_index_version',
        table_name='chunks',
        postgresql_where=sa.text('index_version_id IS NOT NULL'),
    )
    op.drop_column('chunks', 'index_version_id')
