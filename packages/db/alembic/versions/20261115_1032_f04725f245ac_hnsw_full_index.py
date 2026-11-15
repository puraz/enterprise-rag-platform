"""hnsw_full_index

Revision ID: f04725f245ac
Revises: acc6a980ad64
Create Date: 2026-11-15 10:32:00.000000+00:00

## 这次变更做什么

把 `chunk_embeddings` 上的 HNSW 索引从**部分索引**改成**完整索引**。

## 为什么要改（A10 实测结论）

原定义：

    CREATE INDEX ix_chunk_embeddings_hnsw ON chunk_embeddings
      USING hnsw (embedding vector_cosine_ops)
      WITH (m=16, ef_construction=64)
      WHERE embedding IS NOT NULL;     -- ← 问题在这一行

`embedding` 列本身是 `NOT NULL`，所以这个 WHERE 恒真，看起来无害。
但实测（pgvector 0.8.7，20000 行 × 1024 维）证明它让索引**完全无法使用**：

| 索引形式 | 查询 | 计划 | 耗时 |
|---|---|---|---|
| 完整 HNSW | `ORDER BY embedding <=> ?` | Index Scan | **4.2ms** |
| 部分 HNSW | 同上 | Seq Scan + Sort | 131ms |
| 部分 HNSW | 加 `WHERE embedding IS NOT NULL` | Seq Scan + Sort | 131ms |
| 部分 HNSW | `enable_seqscan=off` 强制 | 仍然 Seq Scan | 131ms |

即：**"部分索引"这个形式本身让 HNSW 无法服务于 ORDER BY 向量排序**，
加上对应谓词也无效，强制关闭顺序扫描也无效。

代价是 30 倍延迟，且**不报任何错误**——只有 EXPLAIN 能发现。
这是"索引建了但完全没生效"的典型，也是 A10 最值得记录的坑。

## 锁与耗时

`DROP INDEX` + `CREATE INDEX` 会锁表（ACCESS EXCLUSIVE / SHARE）。
生产库有数据时必须改成：

    CREATE INDEX CONCURRENTLY ix_chunk_embeddings_hnsw_new ...
    DROP INDEX CONCURRENTLY ix_chunk_embeddings_hnsw;
    ALTER INDEX ... RENAME TO ix_chunk_embeddings_hnsw;

且不能在事务里跑（CONCURRENTLY 限制）。本步骤库里 chunk_embeddings 为空，
直接重建。**迁移文件里保留这个说明**，避免后人照着在有数据的库上跑。
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# 迁移标识。不要手改，Alembic 用它们串成链。
revision: str = 'f04725f245ac'
down_revision: str | None = 'acc6a980ad64'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 部分索引 → 完整索引。
    #
    # 必须先 DROP 再 CREATE：PostgreSQL 不支持"去掉索引谓词"的 ALTER，
    # ALTER INDEX 只能改存储参数与表空间。
    op.drop_index('ix_chunk_embeddings_hnsw', table_name='chunk_embeddings')
    op.create_index(
        'ix_chunk_embeddings_hnsw',
        'chunk_embeddings',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64},
        postgresql_ops={'embedding': 'vector_cosine_ops'},
        # **没有 postgresql_where**：完整索引才能被 planner 用于向量排序
    )


def downgrade() -> None:
    op.drop_index('ix_chunk_embeddings_hnsw', table_name='chunk_embeddings')
    op.create_index(
        'ix_chunk_embeddings_hnsw',
        'chunk_embeddings',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64},
        postgresql_ops={'embedding': 'vector_cosine_ops'},
        # 回退到 A10 修复前的状态。**注意**：部分索引会让 HNSW 不被使用
        # （见 upgrade 的说明），所以 downgrade 只用于验证迁移可逆，
        # 不应作为生产回滚目标——正确的回滚是"用旧代码 + 新索引"。
        postgresql_where=sa.text('embedding IS NOT NULL'),
    )
