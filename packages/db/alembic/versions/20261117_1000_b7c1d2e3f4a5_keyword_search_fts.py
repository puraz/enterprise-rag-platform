"""keyword_search_fts

Revision ID: b7c1d2e3f4a5
Revises: 6dcb1c8888b1
Create Date: 2026-11-17 10:00:00.000000+00:00

## 这次变更做什么

关键词检索从 `to_tsvector('simple', content)`（不认中文）改为
**jieba 预分词 + 内置 FTS**：

1. 新增 `chunks.search_text`：worker 入库时写入 jieba 切好的空格分隔词项；
2. 新增生成列 `chunks.search_vector`
   = `to_tsvector('simple', coalesce(nullif(search_text, ''), content))`；
3. 在 `search_vector` 上建 GIN 索引。

## 为什么不是 db 扩展

`zhparser` / `pg_bigm` 在当前 PG 18 上不可用（需编译扩展）；
ParadeDB `pg_search` 需要 Rust 扩展。预分词 + `simple` 分词器
（它按空格切词）在零扩展的前提下得到真正的"词项索引 + ts_rank"，
比 `ILIKE '%词%'` 的纯包含匹配准确得多（见 ADR 0009）。

## 为什么空 search_text 回退到 content

- 直接 SQL 插入的测试/历史数据没有 search_text，回退后英文关键词仍可用；
- 中文历史数据必须**跑回填脚本**（`scripts/backfill_search_text.py`），
  否则整句是一个 token，查询切出的词匹配不到。

## 锁与耗时

`ADD COLUMN ... GENERATED ... STORED` 会重写整表（PG 的生成列必须回填）。
当前库 chunk 量很小，直接执行。**生产大表应走 expand-contract**：

1. 先加可空的 `search_text`（只改 catalog）；
2. 后台回填 search_text；
3. 低峰期再加生成列与 GIN 索引（`CREATE INDEX CONCURRENTLY` 需迁移外执行）。

迁移文件里保留这个说明，避免后人照着在大表上直接跑。
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = 'b7c1d2e3f4a5'
down_revision: str | None = '6dcb1c8888b1'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'chunks',
        sa.Column(
            'search_text',
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
            comment=(
                'jieba 预分词后的空格分隔词项（关键词检索用）。'
                '必须与查询侧用同一分词器，否则中文 FTS 静默不命中'
            ),
        ),
    )
    op.add_column(
        'chunks',
        sa.Column(
            'search_vector',
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple', coalesce(nullif(search_text, ''), content))",
                persisted=True,
            ),
            nullable=True,
            comment='search_text 的 tsvector（生成列，GIN 索引；空 search_text 回退到 content）',
        ),
    )
    op.create_index(
        'ix_chunks_search_vector',
        'chunks',
        ['search_vector'],
        unique=False,
        postgresql_using='gin',
    )


def downgrade() -> None:
    op.drop_index('ix_chunks_search_vector', table_name='chunks')
    op.drop_column('chunks', 'search_vector')
    op.drop_column('chunks', 'search_text')
