"""chunk_embeddings_side_table

Revision ID: acc6a980ad64
Revises: 68940cd4747d
Create Date: 2026-11-15 10:03:31.615167+00:00

## 这次变更做什么

建 `chunk_embeddings` 旁表，让**一个 chunk 能同时拥有多个版本的向量**。

## 为什么放弃 chunks.index_version_id

上一个迁移（68940cd4747d）加了 `chunks.index_version_id`，方案是
"回填时就地覆盖 chunks.embedding 并改 index_version_id"。
`scripts/drill_index_rebuild.py` 演练暴露了它的本质缺陷：

    回填时 chunk 的 index_version_id 从 V1 改成 V2
      → 该 chunk 立刻从 V1 的查询结果里消失
      → 回填完成时 V1 一个 chunk 都没有
      → 既有停机窗口，又无法回滚（旧向量被覆盖）

根因：**一行只能存一个向量**，"两份向量并存"物理上做不到。
旁表把这个限制解除了。所以本迁移**删掉** `chunks.index_version_id`。

## expand-contract 的处理

- **删除 `chunks.index_version_id`**：它是上一迁移刚加的、**从未被
  生产代码写过**（代码还在同时改，没有部署窗口）。所以不算破坏兼容性。
  如果它已在生产写过数据，必须先双写再删——本步骤不存在这个情况。
- **`chunks.embedding` 保留不删**：它是 A3 就有的列，A9 之前的数据在它上面。
  本迁移把它回填到旁表（归入默认版本），但**不删列**——
  删列留到下一个版本，确认旁表稳定后再做。
  这是标准 expand-contract：先加、再回填、后删。

## 回填逻辑

1. 为每个有 chunk 的知识库找/建一个默认 IndexVersion，标 active；
2. 把 `chunks.embedding IS NOT NULL` 的行复制进 chunk_embeddings；
3. `chunks.embedding_model` 一起复制（可能是 NULL，用 'unknown' 兜底）。

**`embedding_model` 不能为 NULL**：旁表该列是 NOT NULL，
且"这个向量是谁算的"是重建时判断"哪些 chunk 需要重算"的依据。

## 锁与耗时

- CREATE TABLE + 索引：新表，无并发写入，快。
- 回填：`INSERT ... SELECT`，行数 = 有向量的 chunk 数。
  本步骤库里 chunk 数为 0，毫秒级。
  **生产上若有百万级 chunk**，应拆成分批回填（脚本 + 后台任务），
  不要在大迁移里一次插完——大事务会长时间持锁且膨胀 WAL。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# 迁移标识。不要手改，Alembic 用它们串成链。
revision: str = 'acc6a980ad64'
down_revision: str | None = '68940cd4747d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# 回填时给"没有记录模型名"的旧数据用的占位符。
_LEGACY_MODEL = "legacy-unknown"


def upgrade() -> None:
    op.create_table(
        'chunk_embeddings',
        sa.Column('chunk_id', sa.UUID(), nullable=False, comment='所属 chunk。CASCADE：删 chunk 时向量一起删（它们没有独立意义）'),
        sa.Column('index_version_id', sa.UUID(), nullable=False, comment='所属索引版本。CASCADE：回收版本时清掉它的全部向量'),
        sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1024), nullable=False, comment='向量（维度 1024）'),
        sa.Column('embedding_model', sa.String(length=128), nullable=False, comment='生成此向量的模型（换模型必然换版本，这里是双保险）'),
        sa.Column('token_count', sa.Integer(), nullable=True, comment='该 chunk 的 token 数（成本估算；从 chunk 冗余过来便于按版本聚合）'),
        sa.Column('id', sa.UUID(), nullable=False, comment='主键'),
        sa.Column('tenant_id', sa.UUID(), nullable=False, comment='租户 ID（服务端从 token 注入，不取请求参数）'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='创建时间（数据库侧生成，UTC）'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='最后更新时间（数据库侧生成，UTC）'),
        sa.CheckConstraint('token_count IS NULL OR token_count >= 0', name=op.f('ck_chunk_embeddings_token_non_negative')),
        sa.ForeignKeyConstraint(['chunk_id'], ['chunks.id'], name=op.f('fk_chunk_embeddings_chunk_id_chunks'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['index_version_id'], ['index_versions.id'], name=op.f('fk_chunk_embeddings_index_version_id_index_versions'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], name=op.f('fk_chunk_embeddings_tenant_id_tenants'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_chunk_embeddings')),
        sa.UniqueConstraint('chunk_id', 'index_version_id', name='uq_chunk_embeddings_chunk_version'),
    )
    op.create_index(
        'ix_chunk_embeddings_hnsw',
        'chunk_embeddings',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64},
        postgresql_ops={'embedding': 'vector_cosine_ops'},
        postgresql_where=sa.text('embedding IS NOT NULL'),
    )
    op.create_index(
        op.f('ix_chunk_embeddings_tenant_id'),
        'chunk_embeddings',
        ['tenant_id'],
        unique=False,
    )
    op.create_index(
        'ix_chunk_embeddings_version',
        'chunk_embeddings',
        ['index_version_id', 'chunk_id'],
        unique=False,
    )

    # ---- 回填：把 chunks.embedding 复制到旁表 ----
    #
    # 1) 为每个有向量的知识库建默认版本（若还没有 active 的）
    op.get_bind().execute(
        sa.text(
            """
            INSERT INTO index_versions (
                id, tenant_id, knowledge_base_id, version_tag, status,
                embedding_model, embedding_dim, index_type, distance_metric,
                index_params, chunk_count, created_at, updated_at
            )
            SELECT
                gen_random_uuid(),
                c.tenant_id,
                c.knowledge_base_id,
                'legacy-backfill',
                'active',
                COALESCE(MAX(c.embedding_model), :legacy_model),
                1024,
                'hnsw',
                'cosine',
                '{"m": 16, "ef_construction": 64}'::jsonb,
                0,
                now(),
                now()
            FROM chunks c
            WHERE c.embedding IS NOT NULL
            GROUP BY c.tenant_id, c.knowledge_base_id
            -- 该知识库还没有任何版本时才建，避免与已存在的 active 版本冲突
            HAVING NOT EXISTS (
                SELECT 1 FROM index_versions v
                WHERE v.knowledge_base_id = c.knowledge_base_id
            )
            """
        ),
        {"legacy_model": _LEGACY_MODEL},
    )

    # 2) 复制向量到旁表，归入该知识库的 active 版本
    op.get_bind().execute(
        sa.text(
            """
            INSERT INTO chunk_embeddings (
                id, tenant_id, chunk_id, index_version_id, embedding,
                embedding_model, token_count, created_at, updated_at
            )
            SELECT
                gen_random_uuid(),
                c.tenant_id,
                c.id,
                v.id,
                c.embedding,
                COALESCE(c.embedding_model, :legacy_model),
                c.token_count,
                now(),
                now()
            FROM chunks c
            JOIN index_versions v
              ON v.knowledge_base_id = c.knowledge_base_id
             AND v.status = 'active'
            WHERE c.embedding IS NOT NULL
            """
        ),
        {"legacy_model": _LEGACY_MODEL},
    )

    # 3) 修正版本计数（回填后的真实条数）
    op.execute(
        """
        UPDATE index_versions v
        SET chunk_count = sub.cnt
        FROM (
            SELECT index_version_id, count(*) AS cnt
            FROM chunk_embeddings
            GROUP BY index_version_id
        ) sub
        WHERE v.id = sub.index_version_id
        """
    )

    # ---- 删除上一迁移引入的 chunks.index_version_id ----
    #
    # 它从未被生产代码写入（上一个迁移与本次在同一开发周期内），
    # 所以删除不破坏滚动发布兼容性。
    op.drop_index(
        op.f('ix_chunks_index_version'),
        table_name='chunks',
        postgresql_where=sa.text('index_version_id IS NOT NULL'),
    )
    op.drop_constraint(
        op.f('fk_chunks_index_version_id_index_versions'), 'chunks', type_='foreignkey'
    )
    op.drop_column('chunks', 'index_version_id')

    # 注意：**不删 chunks.embedding**。
    # 它是 A3 的列，删列要留到下一个版本确认旁表稳定之后（expand-contract）。
    # 保留期间写入以旁表为准，chunks.embedding 作为历史快照不再更新。


def downgrade() -> None:
    # 恢复 chunks.index_version_id（让它与旁表语义一致：从旁表推导）
    op.add_column(
        'chunks',
        sa.Column(
            'index_version_id',
            sa.UUID(),
            autoincrement=False,
            nullable=True,
            comment='索引版本归属（NULL = 尚未向量化/未纳入索引）',
        ),
    )
    op.create_foreign_key(
        op.f('fk_chunks_index_version_id_index_versions'),
        'chunks',
        'index_versions',
        ['index_version_id'],
        ['id'],
        ondelete='SET NULL',
    )
    op.create_index(
        op.f('ix_chunks_index_version'),
        'chunks',
        ['index_version_id'],
        unique=False,
        postgresql_where=sa.text('index_version_id IS NOT NULL'),
    )
    # 把旁表里的向量写回 chunks.embedding（取任意一个版本，优先 active）
    op.execute(
        """
        UPDATE chunks c
        SET embedding = ce.embedding,
            embedding_model = ce.embedding_model,
            index_version_id = ce.index_version_id
        FROM (
            SELECT DISTINCT ON (chunk_id) chunk_id, embedding, embedding_model, index_version_id
            FROM chunk_embeddings
            ORDER BY chunk_id, created_at
        ) ce
        WHERE c.id = ce.chunk_id
        """
    )
    op.drop_index('ix_chunk_embeddings_version', table_name='chunk_embeddings')
    op.drop_index(op.f('ix_chunk_embeddings_tenant_id'), table_name='chunk_embeddings')
    op.drop_index(
        'ix_chunk_embeddings_hnsw',
        table_name='chunk_embeddings',
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64},
        postgresql_ops={'embedding': 'vector_cosine_ops'},
        postgresql_where=sa.text('embedding IS NOT NULL'),
    )
    op.drop_table('chunk_embeddings')


_ = uuid  # 保留导入：迁移里用了 gen_random_uuid()，本地无需 Python 侧 uuid
