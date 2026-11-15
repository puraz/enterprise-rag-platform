"""验证 HNSW 索引在真实数据量下被查询计划使用。

用法：uv run python scripts/bench_hnsw.py [行数]
"""

from __future__ import annotations

import asyncio
import math
import sys
import uuid

from sqlalchemy import text

from rag_core.db import Database
from rag_core.settings import get_settings

INSERT_CHUNK = text(
    """
    INSERT INTO chunks (
        id, tenant_id, knowledge_base_id, document_id, document_version_id,
        chunk_index, chunk_type, content, content_hash, section_path,
        chunk_level, is_current, chunk_metadata, embedding, created_at, updated_at
    ) VALUES (
        :id, :tenant_id, :kb, :doc, :ver,
        :idx, 'text', :content, :hash, '[]'::jsonb,
        0, true, '{}'::jsonb, :embedding, now(), now()
    )
    """
)


def make_vector(seed: int, dim: int = 1024) -> list[float]:
    """确定性伪随机单位向量。"""
    values: list[float] = []
    state = seed * 2654435761 % (2**32)
    for _ in range(dim):
        state = (state * 1103515245 + 12345) % (2**31)
        values.append((state / (2**31)) - 0.5)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


async def seed(db: Database, rows: int) -> tuple[str, uuid.UUID]:
    tenant_id = uuid.uuid4()
    slug_prefix = f"th-{tenant_id.hex[:8]}"
    async with db.transaction() as s:
        await s.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:id, :slug, 'bench', 'active', now(), now())"
            ),
            {"id": tenant_id, "slug": slug_prefix},
        )
        user_id = uuid.uuid4()
        await s.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "tokens_valid_after, created_at, updated_at) "
                "VALUES (:id, :e, 'u', 'active', 'none', now(), now(), now())"
            ),
            {"id": user_id, "e": f"u-{tenant_id.hex[:8]}@bench.test"},
        )
        kb_id = uuid.uuid4()
        await s.execute(
            text(
                "INSERT INTO knowledge_bases (id, tenant_id, name, status, visibility, "
                "confidentiality, created_by, default_chunking_strategy, "
                "default_chunk_size, default_chunk_overlap, document_count, "
                "created_at, updated_at) VALUES ("
                ":id, :t, :n, 'active', 'tenant', 'internal', :u, 'recursive', "
                "800, 100, 0, now(), now())"
            ),
            {"id": kb_id, "t": tenant_id, "n": f"kb-{kb_id.hex[:8]}", "u": user_id},
        )
        blob_id = uuid.uuid4()
        await s.execute(
            text(
                "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, storage_key, "
                "storage_backend, content_type, status, ref_count, created_at, updated_at) "
                "VALUES (:id, :t, :h, 1, :k, 'local', 'text/plain', 'complete', 1, now(), now())"
            ),
            {
                "id": blob_id,
                "t": tenant_id,
                "h": uuid.uuid4().hex + uuid.uuid4().hex,
                "k": f"bench-{blob_id.hex[:8]}",
            },
        )
        doc_id = uuid.uuid4()
        await s.execute(
            text(
                "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, "
                "original_filename, format, source_type, confidentiality, status, "
                "current_version_no, latest_version_no, retry_count, chunk_count, "
                "size_bytes, created_by, version, created_at, updated_at) VALUES ("
                ":id, :t, :k, 'bench-doc', 'x.pdf', 'pdf', 'upload', 'internal', "
                "'ready', 1, 1, 0, 0, 1, :u, 1, now(), now())"
            ),
            {"id": doc_id, "t": tenant_id, "k": kb_id, "u": user_id},
        )
        version_id = uuid.uuid4()
        await s.execute(
            text(
                "INSERT INTO document_versions (id, tenant_id, document_id, version_no, blob_id, "
                "status, chunking_strategy, chunking_config, chunk_count, parse_metadata, "
                "created_by, is_current, created_at, updated_at) "
                "VALUES (:id, :t, :d, 1, :b, 'ready', 'recursive', '{}'::jsonb, 0, '{}'::jsonb, "
                ":u, true, now(), now())"
            ),
            {"id": version_id, "t": tenant_id, "d": doc_id, "b": blob_id, "u": user_id},
        )
        await s.execute(
            text("UPDATE documents SET current_version_id = :v WHERE id = :d"),
            {"v": version_id, "d": doc_id},
        )

        payload = []
        for i in range(rows):
            payload.append(
                {
                    "id": uuid.uuid4(),
                    "tenant_id": tenant_id,
                    "kb": kb_id,
                    "doc": doc_id,
                    "ver": version_id,
                    "idx": i,
                    "content": f"benchmark content number {i}",
                    "hash": uuid.uuid4().hex + uuid.uuid4().hex,
                    "embedding": str(make_vector(i + 1)),
                }
            )
        await s.execute(INSERT_CHUNK, payload)

    return slug_prefix, tenant_id


async def cleanup(db: Database, tenant_slug_prefix: str) -> None:
    async with db.transaction() as s:
        await s.execute(
            text("DELETE FROM tenants WHERE slug = :slug"), {"slug": tenant_slug_prefix}
        )
        await s.execute(
            text("DELETE FROM users WHERE email LIKE :pattern"),
            {"pattern": f"u-{tenant_slug_prefix[3:]}%@bench.test"},
        )


async def explain(db: Database, query: str, *, force_index: bool) -> list[str]:
    """执行 EXPLAIN，返回精简后的计划行。

    EXPLAIN 会把整个查询向量的 1024 个浮点数都打出来，输出几百行。
    这里只保留有意义的结构行（节点类型、索引名、耗时）。
    """
    sql = text(
        "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) "
        "SELECT id FROM chunks "
        "WHERE embedding IS NOT NULL AND is_current = true "
        "ORDER BY embedding <=> :q LIMIT 5"
    )
    keep_patterns = (
        "Scan",
        "Limit",
        "Sort",
        "Index Searches",
        "Execution Time",
        "Planning Time",
        "Buffers: shared",
    )
    if force_index:
        async with db.transaction() as s:
            await s.execute(text("SET LOCAL enable_seqscan = off"))
            result = await s.execute(sql, {"q": query})
            raw = [str(row[0]) for row in result]
    else:
        async with db.session() as s:
            result = await s.execute(sql, {"q": query})
            raw = [str(row[0]) for row in result]

    lines: list[str] = []
    for line in raw:
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过纯向量字面量行（超长，没有诊断价值）
        if "Order By: (embedding" in stripped:
            lines.append("Order By: (embedding <=> :query)")
            continue
        if any(pattern in stripped for pattern in keep_patterns):
            # 去掉代价估算部分，只留实际耗时
            if "(actual time=" in stripped:
                head, _, tail = stripped.partition("(actual time=")
                actual = tail.split(" rows=")[0]
                lines.append(f"{head.strip()} actual={actual}ms")
            else:
                lines.append(stripped)
    return lines


async def main() -> None:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    db = Database(get_settings())
    try:
        slug, _tenant_id = await seed(db, rows)
        async with db.transaction() as s:
            await s.execute(text("ANALYZE chunks"))
        print(f"已插入 {rows} 个 chunk 并 ANALYZE\n")

        query = str(make_vector(1))

        print("=== 默认设置 ===")
        for line in await explain(db, query, force_index=False):
            print(" ", line)

        print("\n=== 强制关顺序扫描（证明索引可用）===")
        for line in await explain(db, query, force_index=True):
            print(" ", line)

        await cleanup(db, slug)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
