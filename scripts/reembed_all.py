"""换 embedding 提供商后，零停机重建索引向量。

## 为什么需要重建版本

不同 provider / 不同模型的向量属于不同向量空间，余弦距离不可比。
系统里有硬防线：`EmbedDocumentHandler` 拒绝把新模型的向量写进
"旧模型的 ACTIVE 版本"——混写不会报错，只会让排序变成噪声。

## 零停机流程（A10）

对每个知识库：

    1. 新建索引版本（status=building，模型 = 当前配置）
    2. 入队 embed 任务，payload 里带 index_version_id（**写入 building 版本**）
       —— 此时 ACTIVE 仍是旧版本，检索完全不受影响
    3. 轮询"新版本已写入的 chunk 数"，直到覆盖全部当前 chunk
    4. 校验通过后切换 ACTIVE（一条 UPDATE，秒级）

失败/超时时**不会切换**：旧版本继续提供服务，新版本停在 building，
排查后可重跑（或手动切换）。旧版本切换后置 `retired`，向量保留可回滚：

    UPDATE index_versions SET status='retired' WHERE knowledge_base_id=:kb AND status='active';
    UPDATE index_versions SET status='active'  WHERE id=:old_version_id;

## 为什么不用 `documents.status`

回填期间文档仍是 `ready`（它们通过旧索引可正常检索）。把状态改成
`embedding` 会让前端显示"处理中"、并且对用户没有任何实际收益。
状态机的 `embedding` 表示"文档还没有任何可用向量"，与"正在重建另一套向量"不同。

用法：

    python /app/scripts/reembed_all.py --dry-run
    python /app/scripts/reembed_all.py            # 回填 + 校验 + 切换
    python /app/scripts/reembed_all.py --no-wait  # 只建版本与入队，不动 ACTIVE
    python /app/scripts/reembed_all.py --tenant-id <uuid> --timeout 900
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from sqlalchemy import text

from rag_contracts.enums import DocumentStatus, TaskType
from rag_core.db import Database
from rag_core.logging import get_logger
from rag_core.settings import get_settings
from rag_db.repositories import TaskRepository
from rag_db.repositories.chunk import IndexVersionRepository

log = get_logger(__name__)


_PICK_DOCS = text(
    """
    SELECT d.id, d.tenant_id, d.knowledge_base_id, v.id AS version_id
    FROM documents d
    JOIN document_versions v
      ON v.document_id = d.id AND v.is_current = TRUE
    WHERE d.status IN (:ready, :embedding)
      AND d.knowledge_base_id IS NOT NULL
      -- 显式 CAST：PG 无法推断 `:tenant_id IS NULL` 中 NULL 字面量的类型。
      AND (CAST(:tenant_id AS uuid) IS NULL OR d.tenant_id = CAST(:tenant_id AS uuid))
    ORDER BY d.created_at
    """
)

_COUNT_EXPECTED = text(
    """
    SELECT count(*) FROM chunks
    WHERE knowledge_base_id = :kb AND is_current = TRUE AND chunk_level = 0
    """
)

_COUNT_BACKFILLED = text(
    """
    SELECT count(*) FROM chunk_embeddings e
    JOIN chunks c ON c.id = e.chunk_id
    WHERE e.index_version_id = :version
      AND c.is_current = TRUE AND c.chunk_level = 0
    """
)


async def _targets(
    db: Database, *, tenant_id: uuid.UUID | None
) -> list[tuple[uuid.UUID, uuid.UUID, uuid.UUID, uuid.UUID]]:
    async with db.transaction_bypass_rls() as session:
        result = await session.execute(
            _PICK_DOCS,
            {
                "ready": DocumentStatus.READY.value,
                "embedding": DocumentStatus.EMBEDDING.value,
                "tenant_id": tenant_id,
            },
        )
        return [(row[0], row[1], row[2], row[3]) for row in result]


async def _expected_chunks(db: Database, *, knowledge_base_id: uuid.UUID) -> int:
    async with db.transaction_bypass_rls() as session:
        result = await session.execute(_COUNT_EXPECTED, {"kb": knowledge_base_id})
        return int(result.scalar_one())


async def _backfilled_chunks(db: Database, *, version_id: uuid.UUID) -> int:
    async with db.transaction_bypass_rls() as session:
        result = await session.execute(_COUNT_BACKFILLED, {"version": version_id})
        return int(result.scalar_one())


async def _create_building_version(
    db: Database,
    *,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    model: str,
    dimension: int,
    expected: int,
    run_id: str,
) -> uuid.UUID:
    """新建一个 building 版本，**不切换 ACTIVE**。"""
    async with db.transaction_bypass_rls() as session:
        version = await IndexVersionRepository(session).create(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            # 唯一约束 (kb, tag)：带 run_id 防止同一秒内重跑撞约束
            version_tag=f"{model}-d{dimension}-{run_id}",
            embedding_model=model,
            embedding_dim=dimension,
            expected_chunk_count=expected,
        )
        return version.id


async def _active_matches(
    db: Database, *, knowledge_base_id: uuid.UUID, model: str, dimension: int
) -> bool:
    async with db.transaction_bypass_rls() as session:
        active = await IndexVersionRepository(session).get_active(
            knowledge_base_id=knowledge_base_id
        )
        return bool(
            active is not None
            and active.embedding_model == model
            and active.embedding_dim == dimension
        )


async def _enqueue_rebuild(
    db: Database,
    *,
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    document_version_id: uuid.UUID,
    index_version_id: uuid.UUID,
    run_id: str,
) -> None:
    """入队“把该版本文档的 chunk 写入指定索引版本”。

    不修改文档状态、不清空遗留列：回填是**旁路**，不应影响在线检索。
    """
    async with db.transaction_bypass_rls() as session:
        await TaskRepository(session).enqueue(
            tenant_id=tenant_id,
            task_type=TaskType.EMBED_DOCUMENT.value,
            # run_id + 目标版本：重跑会生成新任务，而不是复用历史已完成任务
            idempotency_key=f"reembed:{index_version_id}:{document_version_id}",
            payload={
                "document_id": str(document_id),
                "document_version_id": str(document_version_id),
                # 关键：写入 building 版本，而不是 ACTIVE
                "index_version_id": str(index_version_id),
            },
            document_id=document_id,
            document_version_id=document_version_id,
            knowledge_base_id=knowledge_base_id,
        )


async def _activate(db: Database, *, knowledge_base_id: uuid.UUID, version_id: uuid.UUID) -> None:
    async with db.transaction_bypass_rls() as session:
        await IndexVersionRepository(session).activate(
            knowledge_base_id=knowledge_base_id, version_id=version_id
        )


async def _wait_backfilled(
    db: Database,
    *,
    version_id: uuid.UUID,
    expected: int,
    timeout_s: int,
) -> bool:
    """轮询直到新版本覆盖全部 chunk（或超时）。"""
    elapsed = 0
    while elapsed <= timeout_s:
        done = await _backfilled_chunks(db, version_id=version_id)
        if done >= expected:
            return True
        if elapsed % 10 == 0:
            print(f"    回填中 {done}/{expected} …")
        await asyncio.sleep(2)
        elapsed += 2
    return False


async def reembed(
    *,
    tenant_id: uuid.UUID | None,
    dry_run: bool,
    wait: bool,
    timeout_s: int,
    force_new: bool = False,
) -> int:
    settings = get_settings()
    model = settings.embedding.model_name
    dimension = settings.embedding.dimension
    db = Database(settings)
    run_id = uuid.uuid4().hex[:12]
    try:
        targets = await _targets(db, tenant_id=tenant_id)
        print(
            f"provider={settings.embedding.provider} model={model} "
            f"dimension={dimension}（run_id={run_id}）"
        )
        print(f"待处理文档: {len(targets)}")

        # 按知识库分组；同一 KB 只建一个版本
        by_kb: dict[uuid.UUID, list[tuple[uuid.UUID, uuid.UUID, uuid.UUID]]] = {}
        tenants: dict[uuid.UUID, uuid.UUID] = {}
        for document_id, doc_tenant, kb_id, version_id in targets:
            by_kb.setdefault(kb_id, []).append((document_id, version_id, doc_tenant))
            tenants[kb_id] = doc_tenant

        activated = 0
        for kb_id, docs in by_kb.items():
            expected = await _expected_chunks(db, knowledge_base_id=kb_id)
            if expected == 0:
                print(f"  kb={kb_id} 没有 chunk，跳过（{len(docs)} 个文档）")
                continue

            if (
                await _active_matches(db, knowledge_base_id=kb_id, model=model, dimension=dimension)
                and not force_new
            ):
                # ACTIVE 已是当前模型：说明只需补齐（例如上次中断）。
                # 用 ACTIVE 自己作为目标版本，避免每跑一次就多建一个版本。
                active_id = await _active_version_id(db, knowledge_base_id=kb_id)
                if active_id is None:  # pragma: no cover - _active_matches 已保证存在
                    continue
                if await _backfilled_chunks(db, version_id=active_id) >= expected:
                    print(f"  kb={kb_id} 已是当前模型且回填完整，跳过")
                    continue
                if dry_run:
                    print(f"  [dry-run] kb={kb_id} 补齐 ACTIVE 版本 {active_id}")
                    continue
                target_version = active_id
                created = False
            else:
                if dry_run:
                    print(
                        f"  [dry-run] kb={kb_id} 新建版本 {model}/d{dimension}，"
                        f"预计回填 {expected} 个 chunk"
                    )
                    continue
                target_version = await _create_building_version(
                    db,
                    tenant_id=tenants[kb_id],
                    knowledge_base_id=kb_id,
                    model=model,
                    dimension=dimension,
                    expected=expected,
                    run_id=run_id,
                )
                created = True

            print(
                f"  kb={kb_id} 目标版本 {target_version}"
                f"{'（新建 building）' if created else '（补齐 ACTIVE）'} "
                f"expected={expected}"
            )

            for document_id, version_id, _tenant in docs:
                await _enqueue_rebuild(
                    db,
                    document_id=document_id,
                    tenant_id=tenants[kb_id],
                    knowledge_base_id=kb_id,
                    document_version_id=version_id,
                    index_version_id=target_version,
                    run_id=run_id,
                )
            print(f"    已入队 {len(docs)} 个文档")

            if not wait:
                print("    --no-wait：不校验、不切换 ACTIVE")
                continue

            ok = await _wait_backfilled(
                db, version_id=target_version, expected=expected, timeout_s=timeout_s
            )
            if not ok:
                print(
                    f"    ✗ 回填未完成（超时 {timeout_s}s）：ACTIVE 未切换，"
                    "旧索引继续服务；排查后重跑本脚本"
                )
                continue
            await _activate(db, knowledge_base_id=kb_id, version_id=target_version)
            activated += 1
            print("    ✓ 已切换 ACTIVE（旧版本保留为 retired，可回滚）")

        if dry_run:
            print("dry-run 结束，未修改数据")
        else:
            print(f"完成：切换了 {activated} 个知识库的索引版本")
        return len(targets)
    finally:
        await db.close()


async def _active_version_id(db: Database, *, knowledge_base_id: uuid.UUID) -> uuid.UUID | None:
    async with db.transaction_bypass_rls() as session:
        active = await IndexVersionRepository(session).get_active(
            knowledge_base_id=knowledge_base_id
        )
        return active.id if active is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description="换 embedding 提供商后零停机重建索引")
    parser.add_argument("--tenant-id", default=None, help="只处理某个租户")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不修改数据")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="只建版本并入队，不等待回填完成、不切换 ACTIVE",
    )
    parser.add_argument("--timeout", type=int, default=900, help="等待回填的最长秒数（默认 900）")
    parser.add_argument(
        "--force-new-version",
        action="store_true",
        help="即使 ACTIVE 已是当前模型也新建版本（模型未变时的索引重建/演练）",
    )
    args = parser.parse_args()
    tenant_id = uuid.UUID(args.tenant_id) if args.tenant_id else None
    asyncio.run(
        reembed(
            tenant_id=tenant_id,
            dry_run=args.dry_run,
            wait=not args.no_wait,
            timeout_s=args.timeout,
            force_new=args.force_new_version,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
