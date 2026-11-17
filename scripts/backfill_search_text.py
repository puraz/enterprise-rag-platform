"""回填 `chunks.search_text`（关键词检索的预分词列）。

## 什么时候需要跑

新增 `search_text` 后，**历史 chunk 是空串**：FTS 表达式会回退到
`content`，英文关键词仍可检索，但中文整句是一个 token，查询切出的词
匹配不到——表现为"混合检索好像没生效"。

本脚本用与线上完全相同的分词器（`tokenize_for_index`）回填：

    uv run python scripts/backfill_search_text.py
    uv run python scripts/backfill_search_text.py --batch-size 1000
    uv run python scripts/backfill_search_text.py --dry-run

## 为什么用 admin（BYPASSRLS）会话

回填要跨租户扫全表。用普通会话（未设 `app.tenant_id`）会被 RLS
过滤成空——**不报错，只是永远回填 0 条**。这是本项目最典型的
"RLS 静默失败"形态之一。

## 为什么分批 + 事务外分词

- 一次 `SELECT` 全表会吃满内存；
- jieba 分词是 CPU 密集，放在事务里会长时间占住连接、阻塞 vacuum。
  所以每批：读一批 → 关事务 → 线程池分词 → 开事务写回。

`search_vector` 是生成列，更新 `search_text` 后由数据库自动重算，
不需要应用层再写 tsvector。
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from sqlalchemy import text

from rag_adapters.retrieval.keyword import tokenize_for_index
from rag_core.db import Database
from rag_core.settings import get_settings


async def _fetch_batch(db: Database, *, batch_size: int) -> list[tuple[uuid.UUID, str]]:
    async with db.transaction_bypass_rls() as session:
        result = await session.execute(
            text(
                "SELECT id, content FROM chunks "
                "WHERE search_text = '' AND content <> '' "
                "ORDER BY id LIMIT :n"
            ),
            {"n": batch_size},
        )
        return [(row[0], str(row[1])) for row in result]


async def _store_batch(db: Database, updates: list[tuple[uuid.UUID, str]]) -> None:
    async with db.transaction_bypass_rls() as session:
        await session.execute(
            text("UPDATE chunks SET search_text = :s WHERE id = :i"),
            [{"s": search_text, "i": chunk_id} for chunk_id, search_text in updates],
        )


async def backfill(*, batch_size: int, dry_run: bool) -> int:
    settings = get_settings()
    db = Database(settings)
    total = 0
    try:
        while True:
            batch = await _fetch_batch(db, batch_size=batch_size)
            if not batch:
                break

            # 分词放到线程池：脚本也要遵守"不阻塞事件循环"的约定，
            # 且未来若把它改成 API 管理端点，不需要再改这一层。
            texts = await asyncio.to_thread(
                lambda items=batch: [tokenize_for_index(content) for _id, content in items]
            )
            updates = [
                (chunk_id, search_text)
                for (chunk_id, _content), search_text in zip(batch, texts, strict=True)
            ]

            if not dry_run:
                await _store_batch(db, updates)

            total += len(updates)
            print(f"  {'[dry-run] ' if dry_run else ''}已处理 {total} 条")
    finally:
        await db.close()
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="回填 chunks.search_text")
    parser.add_argument("--batch-size", type=int, default=500, help="每批条数")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写库")
    args = parser.parse_args()

    total = asyncio.run(backfill(batch_size=args.batch_size, dry_run=args.dry_run))
    print(f"完成：{total} 条{'（dry-run，未写入）' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
