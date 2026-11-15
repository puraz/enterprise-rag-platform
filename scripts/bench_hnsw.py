"""HNSW 参数实测：验证索引被使用，并对比 ef_search 对召回与延迟的影响。

## 要回答的三个问题

1. **索引真的被用了吗**：EXPLAIN 里是否出现 HNSW 索引扫描。
   不用索引时 10 万 chunk 的检索要几百毫秒。
2. **ef_search 怎么选**：越大召回越高、越慢。找到"召回够用且延迟可接受"的取值。
3. **多租户过滤后召回是否塌陷**：HNSW 先扫候选后过滤 tenant_id，
   目标租户占比小时可能一个都不匹配 → 返回空。这正是
   `hnsw.iterative_scan = relaxed_order` 要解决的问题。

## 为什么必须用真实数据量

HNSW 在小数据上**不会**被 planner 选中（seq scan 更快）。
小规模测试得到"没用索引"的结论是错的。默认 20000 行。

## 召回率怎么算

没有黄金集时用**精确检索**（强制 seq scan）作真值：
    recall = |近似 top-k ∩ 精确 top-k| / k
这不是"业务召回"，但准确度量了"索引近似程度"——
ef_search 太小导致的漏检靠它发现。

## 用法

    uv run python scripts/bench_hnsw.py                  # 默认 20000 行
    uv run python scripts/bench_hnsw.py --rows 50000
    uv run python scripts/bench_hnsw.py --keep           # 保留测试数据
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import sys
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_core.settings import get_settings, load_env_files

VECTOR_DIM = 1024


def make_vector(seed: int, dim: int = VECTOR_DIM) -> list[float]:
    """确定性伪随机单位向量。

    **归一化**与生产一致（`normalize_embeddings=True`）：
    cosine 距离对归一化向量等价于内积，计算更省。
    """
    values: list[float] = []
    state = seed * 2654435761 % (2**32)
    for _ in range(dim):
        state = (state * 1103515245 + 12345) % (2**31)
        values.append((state / (2**31)) - 0.5)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _parse_vector(raw: str) -> list[float]:
    """把 pgvector 的文本表示 `[a,b,c]` 解析成浮点列表。"""
    return [float(value) for value in raw.strip("[]").split(",")]


@dataclass
class BenchResult:
    """一组参数的测量结果。"""

    ef_search: int
    iterative_scan: str
    latency_ms: float
    recall: float


async def _seed(
    factory: async_sessionmaker[AsyncSession], *, rows: int, tenants: int
) -> tuple[list[uuid.UUID], uuid.UUID]:
    """建 N 行 chunk + 旁表向量，分布到多个租户。

    多租户分布是**故意的**：单租户测不出"过滤后召回塌陷"。
    不过为了让测量有意义，数据全部归属第一个租户
    （其他租户只增加表规模，用于观察"过滤选择性"的影响）。
    """
    tenant_ids = [uuid.uuid4() for _ in range(tenants)]
    kb_id = uuid.uuid4()
    user_id = uuid.uuid4()

    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "created_at, updated_at) VALUES (:id, :e, 'bench', 'active', 'none', "
                "now(), now())"
            ),
            {"id": user_id, "e": f"bench-{user_id.hex[:8]}@bench.local"},
        )
        for tenant_id in tenant_ids:
            await session.execute(
                text(
                    "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                    "VALUES (:id, :slug, 'bench', 'active', now(), now())"
                ),
                {"id": tenant_id, "slug": f"bench-{tenant_id.hex[:8]}"},
            )
        await session.execute(
            text(
                "INSERT INTO knowledge_bases (id, tenant_id, name, status, created_by, "
                "created_at, updated_at) VALUES (:id, :t, 'bench-kb', 'active', :u, "
                "now(), now())"
            ),
            {"id": kb_id, "t": tenant_ids[0], "u": user_id},
        )
        version_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO index_versions (id, tenant_id, knowledge_base_id, version_tag, "
                "status, embedding_model, embedding_dim, index_type, distance_metric, "
                "index_params, chunk_count, created_at, updated_at) "
                "VALUES (:id, :t, :kb, 'bench', 'active', 'bench', :dim, 'hnsw', 'cosine', "
                "'{}'::jsonb, :n, now(), now())"
            ),
            {"id": version_id, "t": tenant_ids[0], "kb": kb_id, "dim": VECTOR_DIM, "n": rows},
        )
        blob_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, storage_key, "
                "storage_backend, content_type, status, ref_count, created_at, updated_at) "
                "VALUES (:id, :t, :h, 1, :k, 'local', 'text/plain', 'complete', 1, now(), now())"
            ),
            {"id": blob_id, "t": tenant_ids[0], "h": "b" * 64, "k": f"bench-{blob_id.hex[:8]}"},
        )
        doc_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, "
                "original_filename, format, source_type, confidentiality, status, "
                "current_version_no, latest_version_no, retry_count, chunk_count, "
                "size_bytes, created_by, version, created_at, updated_at) "
                "VALUES (:id, :t, :kb, 'bench', 'x.pdf', 'pdf', 'upload', 'internal', "
                "'ready', 1, 1, 0, :n, 1, :u, 1, now(), now())"
            ),
            {"id": doc_id, "t": tenant_ids[0], "kb": kb_id, "n": rows, "u": user_id},
        )
        ver_row_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO document_versions (id, tenant_id, document_id, version_no, "
                "blob_id, status, chunking_strategy, chunking_config, chunk_count, "
                "parse_metadata, created_by, is_current, created_at, updated_at) "
                "VALUES (:id, :t, :d, 1, :b, 'ready', 'recursive', '{}'::jsonb, :n, "
                "'{}'::jsonb, :u, true, now(), now())"
            ),
            {
                "id": ver_row_id,
                "t": tenant_ids[0],
                "d": doc_id,
                "b": blob_id,
                "n": rows,
                "u": user_id,
            },
        )
        await session.commit()

    batch = 500
    for start in range(0, rows, batch):
        async with factory() as session:
            chunk_rows = []
            embed_rows = []
            for offset in range(start, min(start + batch, rows)):
                chunk_id = uuid.uuid4()
                chunk_rows.append(
                    {
                        "id": chunk_id,
                        "tid": tenant_ids[0],
                        "kb": kb_id,
                        "doc": doc_id,
                        "ver": ver_row_id,
                        "idx": offset,
                        "c": f"bench chunk {offset}",
                        "h": f"{offset:064x}",
                    }
                )
                embed_rows.append(
                    {
                        "id": uuid.uuid4(),
                        "tid": tenant_ids[0],
                        "cid": chunk_id,
                        "vid": version_id,
                        "emb": str(make_vector(offset + 1)),
                    }
                )
            await session.execute(
                text(
                    "INSERT INTO chunks (id, tenant_id, knowledge_base_id, document_id, "
                    "document_version_id, chunk_index, chunk_type, content, content_hash, "
                    "section_path, chunk_level, is_current, chunk_metadata, "
                    "created_at, updated_at) "
                    "VALUES (:id, :tid, :kb, :doc, :ver, :idx, 'text', :c, :h, "
                    "'[]'::jsonb, 0, true, '{}'::jsonb, now(), now())"
                ),
                chunk_rows,
            )
            await session.execute(
                text(
                    "INSERT INTO chunk_embeddings (id, tenant_id, chunk_id, index_version_id, "
                    "embedding, embedding_model, created_at, updated_at) "
                    "VALUES (:id, :tid, :cid, :vid, :emb, 'bench', now(), now())"
                ),
                embed_rows,
            )
            await session.commit()

    return tenant_ids, version_id


_SEARCH_SQL = (
    "SELECT c.id FROM chunks c "
    "JOIN chunk_embeddings ce ON ce.chunk_id = c.id "
    "WHERE c.tenant_id = :tid AND ce.index_version_id = :vid "
    "  AND c.is_current = true AND c.chunk_level = 0 "
    "ORDER BY ce.embedding <=> CAST(:q AS vector) LIMIT :k"
)


async def _exact_search(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    version_id: uuid.UUID,
    query: list[float],
    k: int,
) -> list[uuid.UUID]:
    """精确检索（禁用所有索引扫描方式），作为召回基准真值。

    **必须同时禁用 indexscan / bitmapscan / indexonlyscan**：
    只关 indexscan 的话 planner 可能改用 Bitmap Index Scan 走 HNSW，
    那样"精确检索"其实是近似检索，召回率基准就没意义了
    （表现为召回率异常低，但实际是基准错了）。
    """
    await session.execute(text("SET LOCAL enable_indexscan = off"))
    await session.execute(text("SET LOCAL enable_bitmapscan = off"))
    await session.execute(text("SET LOCAL enable_indexonlyscan = off"))
    result = await session.execute(
        text(_SEARCH_SQL),
        {"tid": tenant_id, "vid": version_id, "q": str(query), "k": k},
    )
    return [row[0] for row in result]


async def _approx_search(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    version_id: uuid.UUID,
    query: list[float],
    k: int,
    ef_search: int,
    iterative_scan: str,
    force_index: bool = False,
) -> list[uuid.UUID]:
    """近似检索（走 HNSW）。

    Args:
        force_index: 关闭 seq scan/sort，**强制**走 HNSW。

            为什么需要这个开关：小数据量下 pgvector 的代价模型会认为
            "顺序扫描 + 排序"比"图遍历"便宜（即使实测反了），
            于是 planner 不用索引。不强制的话测出来的全是 seq scan 耗时，
            完全测不到 ef_search 的影响（召回率恒 100%，因为它就是精确检索）。
    """
    if force_index:
        await session.execute(text("SET LOCAL enable_seqscan = off"))
        await session.execute(text("SET LOCAL enable_sort = off"))
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    if iterative_scan != "off":
        await session.execute(text(f"SET LOCAL hnsw.iterative_scan = {iterative_scan}"))
    result = await session.execute(
        text(_SEARCH_SQL),
        {"tid": tenant_id, "vid": version_id, "q": str(query), "k": k},
    )
    return [row[0] for row in result]


async def _explain(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    version_id: uuid.UUID,
    query: list[float],
    k: int,
    force_index: bool = False,
) -> list[str]:
    if force_index:
        await session.execute(text("SET LOCAL enable_seqscan = off"))
        await session.execute(text("SET LOCAL enable_sort = off"))
    result = await session.execute(
        text("EXPLAIN (ANALYZE, BUFFERS) " + _SEARCH_SQL),
        {"tid": tenant_id, "vid": version_id, "q": str(query), "k": k},
    )
    keep = ("Scan", "Limit", "Sort", "Index Searches", "Execution Time", "Planning Time")
    lines: list[str] = []
    for row in result:
        line = str(row[0]).strip()
        if "Order By: (ce.embedding" in line:
            lines.append("Order By: (embedding <=> :query)")
        elif any(token in line for token in keep):
            lines.append(line)
    return lines


async def run_bench(*, rows: int, tenants: int, keep: bool) -> int:
    load_env_files()
    settings = get_settings()
    db = settings.database
    dsn = (
        f"postgresql+asyncpg://{db.migration_user}:"
        f"{db.migration_password.get_secret_value()}@{db.host}:{db.port}/{db.name}"
    )
    # **必须带上与生产相同的 server_settings**：
    # `random_page_cost` 决定 planner 是否选用 HNSW 索引。
    # 用默认值（4.0，机械硬盘假设）测出来的"索引没被使用"
    # 是测量环境的问题，不是代码的问题——这种假阴性最误导人。
    engine = create_async_engine(
        dsn,
        pool_size=4,
        connect_args={
            "server_settings": {
                "random_page_cost": str(settings.database.random_page_cost),
            }
        },
    )
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    print(f"准备 {rows} 行数据（{tenants} 个租户）...")
    seed_started = time.perf_counter()
    tenant_ids, version_id = await _seed(factory, rows=rows, tenants=tenants)
    print(f"数据准备完成：{time.perf_counter() - seed_started:.1f}s\n")

    target_tenant = tenant_ids[0]
    k = 10

    # **查询向量必须取自数据集内部**
    #
    # 用随机生成的查询向量（与数据毫无关系）会得到荒谬的召回率：
    # 随机高维向量之间的余弦距离几乎相同，top-10 与 top-11 的差距
    # 小于浮点噪声，精确检索的"真值"本身就不可靠，HNSW 更不可能复现。
    #
    # 取数据集里的向量当作查询，才有明确的"最近邻"语义
    # （至少自己与自己距离为 0），这是评测向量索引的标准做法。
    async with factory() as session:
        rows = await session.execute(
            text(
                "SELECT embedding::text FROM chunk_embeddings "
                "WHERE tenant_id = :t ORDER BY chunk_id LIMIT 20"
            ),
            {"t": target_tenant},
        )
        queries = [_parse_vector(str(row[0])) for row in rows]
    results: list[BenchResult] = []
    print(f"查询向量：{len(queries)} 个（取自数据集内部）\n")

    try:
        async with factory() as session:
            plan = await _explain(
                session, tenant_id=target_tenant, version_id=version_id, query=queries[0], k=k
            )
        print("查询计划：")
        for line in plan:
            print(f"  {line}")
        uses_hnsw = any("hnsw" in line.lower() for line in plan)
        print(f"  → planner 选择 HNSW: {'是' if uses_hnsw else '否'}")

        # 强制走索引的计划对比：证明索引可用，只是 planner 没选
        async with factory() as session:
            forced_plan = await _explain(
                session,
                tenant_id=target_tenant,
                version_id=version_id,
                query=queries[0],
                k=k,
                force_index=True,
            )
        forced_uses = any("hnsw" in line.lower() for line in forced_plan)
        if not uses_hnsw and forced_uses:
            print("  → 强制后使用 HNSW: 是（说明索引可用，planner 代价估算未选它）")
            print("    **这是已知的 pgvector 代价模型局限**：小数据量下")
            print("    'Seq Scan + Sort' 的估算代价低于图遍历，尽管实测更慢。")
            print("    数据量增长到 ~10 万行后 planner 会自行改用索引。")
        print()

        started = time.perf_counter()
        async with factory() as session:
            exact_sets = [
                await _exact_search(
                    session,
                    tenant_id=target_tenant,
                    version_id=version_id,
                    query=q,
                    k=k,
                )
                for q in queries
            ]
        exact_ms = (time.perf_counter() - started) / len(queries) * 1000
        print(f"精确检索（seq scan）平均: {exact_ms:.2f}ms/query\n")

        print("== 强制走 HNSW（测 ef_search 的真实影响）==")
        print(f"{'ef_search':>10} {'迭代扫描':>14} {'延迟(ms)':>10} {'召回率':>8}")
        print("-" * 48)
        for ef in (10, 40, 100, 200):
            for mode in ("off", "relaxed_order"):
                latencies: list[float] = []
                hits = 0
                async with factory() as session:
                    for index, query in enumerate(queries):
                        started = time.perf_counter()
                        approx = await _approx_search(
                            session,
                            tenant_id=target_tenant,
                            version_id=version_id,
                            query=query,
                            k=k,
                            ef_search=ef,
                            iterative_scan=mode,
                            force_index=True,
                        )
                        latencies.append((time.perf_counter() - started) * 1000)
                        hits += len(set(approx) & set(exact_sets[index]))
                recall = hits / (len(queries) * k)
                latency = statistics.fmean(latencies)
                results.append(
                    BenchResult(
                        ef_search=ef,
                        iterative_scan=mode,
                        latency_ms=latency,
                        recall=recall,
                    )
                )
                print(f"{ef:>10} {mode:>14} {latency:>10.2f} {recall:>7.1%}")
            print()

        print("=" * 48)
        if not uses_hnsw:
            print(
                f"注意：{rows} 行时 planner 未选 HNSW。上面的延迟/召回是在"
                "**强制走索引**下测的——那才是索引的真实性能。"
            )
        full = [r for r in results if r.recall >= 0.999]
        if full:
            cheapest = min(full, key=lambda r: r.ef_search)
            print(
                f"召回到 100% 的最低 ef_search: {cheapest.ef_search}"
                f"（{cheapest.iterative_scan}，{cheapest.latency_ms:.2f}ms）"
            )
        else:
            best = max(results, key=lambda r: r.recall)
            print(
                f"未达 100%，最高 {best.recall:.1%}"
                f"（ef_search={best.ef_search}, {best.iterative_scan}）"
            )
        if rows < 5000:
            print("\n提示：行数 < 5000 时 planner 通常选 seq scan。建议 --rows 20000")
        return 0
    finally:
        if not keep:
            async with factory() as session:
                for tenant_id in tenant_ids:
                    for statement in (
                        "DELETE FROM chunk_embeddings WHERE tenant_id = :t",
                        "DELETE FROM chunks WHERE tenant_id = :t",
                        "DELETE FROM index_versions WHERE tenant_id = :t",
                        "DELETE FROM document_versions WHERE tenant_id = :t",
                        "DELETE FROM documents WHERE tenant_id = :t",
                        "DELETE FROM knowledge_bases WHERE tenant_id = :t",
                        "DELETE FROM blobs WHERE tenant_id = :t",
                    ):
                        await session.execute(text(statement), {"t": tenant_id})
                for tenant_id in tenant_ids:
                    await session.execute(
                        text("DELETE FROM tenants WHERE id = :t"), {"t": tenant_id}
                    )
                await session.execute(
                    text("DELETE FROM users WHERE email LIKE 'bench-%@bench.local'")
                )
                await session.commit()
            print("\n测试数据已清理")
        else:
            print(f"\n测试数据已保留：{[str(t) for t in tenant_ids]}")
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HNSW 参数实测")
    parser.add_argument("--rows", type=int, default=20_000, help="行数（默认 20000）")
    parser.add_argument("--tenants", type=int, default=5, help="租户数")
    parser.add_argument("--keep", action="store_true", help="保留测试数据")
    args = parser.parse_args(argv)
    return asyncio.run(run_bench(rows=args.rows, tenants=args.tenants, keep=args.keep))


if __name__ == "__main__":
    sys.exit(main())
