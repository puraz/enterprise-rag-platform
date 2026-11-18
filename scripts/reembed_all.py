"""换 embedding 提供商后，按 A10 的方式重建索引向量。

## 为什么必须重建版本，而不是原地覆盖

系统里有一道防护：`EmbedDocumentHandler._ensure_active_index_version`
会校验"当前 ACTIVE 版本使用的模型 == 当前 provider"。不一致就**永久失败**，
提示"必须先重建索引版本"。

这道防护是对的：不同模型的向量不可比，混在同一个版本里会让距离计算
失去意义——检索照样返回结果、不报错，只是排序全是噪声。

所以本脚本走版本化重建：

1. 每个知识库建一个**新**索引版本（`embedding_model` = 当前 provider），
   并切换 ACTIVE 指向它（旧版本自动置 `retired`，向量保留可回滚）；
2. 把文档状态重置为 `embedding`、清空遗留列 `chunks.embedding`；
3. 入队 `embed_document`，worker 会把向量写入**新 ACTIVE 版本**。

## 能回滚

旧版本只是状态变成 `retired`，`chunk_embeddings` 行还在。要回滚：

```sql
UPDATE index_versions SET status='retired' WHERE knowledge_base_id=:kb AND status='active';
UPDATE index_versions SET status='active'  WHERE id=:old_version_id;
```

## 已知窗口

切换 ACTIVE 到"空的新版本"后、worker 回填完成前，该知识库检索不到内容
（不是报错，是 0 命中）。这是**用可用性换正确性**：宁可暂时搜不到，
也不要让用户看到基于错误向量的排序。

真正零停机的做法是"新建 → 回填 → 校验条数 → 切换"（先把新版本填满再切），
需要后台回填任务与完成回调，属于 A10 的后续工作。当前脚本面向开发/演示
与"换 provider 后尽快恢复可用"。

用法：

    python /app/scripts/reembed_all.py --dry-run
    python /app/scripts/reembed_all.py
    python /app/scripts/reembed_all.py --tenant-id <uuid>
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
      -- 显式 CAST：PG 无法推断 `:tenant_id IS NULL` 中 NULL 字面量的类型，
      -- 会报 "could not determine data type of parameter"。
      AND (CAST(:tenant_id AS uuid) IS NULL OR d.tenant_id = CAST(:tenant_id AS uuid))
    ORDER BY d.created_at
    """
)


async def _targets(
    db: Database, *, tenant_id: uuid.UUID | None
) -> list[tuple[uuid.UUID, uuid.UUID, uuid.UUID, uuid.UUID]]:
    async with db.transaction_bypass_rls() as session:
        result = await session.execute(
            _PICK_DOCS,
            {
                # `embedding` 也要包含：上一次重建失败时文档会停在 embedding，
                # 只扫 ready 会永远漏掉它们。
                "ready": DocumentStatus.READY.value,
                "embedding": DocumentStatus.EMBEDDING.value,
                "tenant_id": tenant_id,
            },
        )
        return [(row[0], row[1], row[2], row[3]) for row in result]


async def _prepare_index_version(
    db: Database,
    *,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    model: str,
    dimension: int,
    run_id: str,
) -> tuple[uuid.UUID, bool]:
    """确保 ACTIVE 版本与当前模型匹配；不匹配就新建并切换。

    Returns:
        `(active_version_id, created_new)`
    """
    async with db.transaction_bypass_rls() as session:
        repo = IndexVersionRepository(session)
        active = await repo.get_active(knowledge_base_id=knowledge_base_id)
        if (
            active is not None
            and active.embedding_model == model
            and active.embedding_dim == dimension
        ):
            return active.id, False

        version = await repo.create(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            # 唯一约束 (kb, tag)：带 run_id 防止同一秒内重跑撞约束
            version_tag=f"{model}-d{dimension}-{run_id}",
            embedding_model=model,
            embedding_dim=dimension,
        )
        # 旧 ACTIVE 自动置 retired（向量保留，可回滚）
        await repo.activate(knowledge_base_id=knowledge_base_id, version_id=version.id)
        return version.id, True


async def _reset_and_enqueue(
    db: Database,
    *,
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    version_id: uuid.UUID,
    run_id: str,
) -> None:
    async with db.transaction_bypass_rls() as session:
        # `chunks.embedding` 是 A10 之前的遗留列（现在恒为 NULL），
        # 清空它只是保持旧数据一致；真正的"是否已向量化"看旁表。
        await session.execute(
            text(
                "UPDATE chunks SET embedding = NULL, embedding_model = NULL "
                "WHERE document_version_id = :v"
            ),
            {"v": version_id},
        )
        await session.execute(
            text(
                "UPDATE documents SET status = :s, error_code = NULL, error_message = NULL "
                "WHERE id = :d"
            ),
            {"s": DocumentStatus.EMBEDDING.value, "d": document_id},
        )
        await TaskRepository(session).enqueue(
            tenant_id=tenant_id,
            task_type=TaskType.EMBED_DOCUMENT.value,
            # run_id 保证"重跑一次就真的会再跑"，而不是复用历史已完成任务
            idempotency_key=f"reembed:{version_id}:{run_id}",
            payload={
                "document_id": str(document_id),
                "document_version_id": str(version_id),
            },
            document_id=document_id,
            document_version_id=version_id,
            knowledge_base_id=knowledge_base_id,
        )


async def reembed(*, tenant_id: uuid.UUID | None, dry_run: bool) -> int:
    settings = get_settings()
    model = settings.embedding.model_name
    dimension = settings.embedding.dimension
    db = Database(settings)
    run_id = uuid.uuid4().hex[:12]
    try:
        targets = await _targets(db, tenant_id=tenant_id)
        print(f"provider={settings.embedding.provider} model={model} dimension={dimension}")
        print(f"待重建文档: {len(targets)}（run_id={run_id}）")

        if dry_run:
            for document_id, _tenant, kb_id, version_id in targets:
                print(f"  [dry-run] document={document_id} version={version_id} kb={kb_id}")
            return len(targets)

        # 先按知识库准备索引版本：同一 KB 只建一次
        prepared: dict[uuid.UUID, uuid.UUID] = {}
        for _document_id, doc_tenant, kb_id, _version_id in targets:
            if kb_id in prepared:
                continue
            active_id, created = await _prepare_index_version(
                db,
                tenant_id=doc_tenant,
                knowledge_base_id=kb_id,
                model=model,
                dimension=dimension,
                run_id=run_id,
            )
            prepared[kb_id] = active_id
            print(
                f"  kb={kb_id} active_version={active_id} "
                f"{'（新建并切换）' if created else '（已匹配当前模型）'}"
            )

        for document_id, doc_tenant, kb_id, version_id in targets:
            await _reset_and_enqueue(
                db,
                document_id=document_id,
                tenant_id=doc_tenant,
                knowledge_base_id=kb_id,
                version_id=version_id,
                run_id=run_id,
            )
            print(f"  enqueued document={document_id} version={version_id}")

        print("完成：worker 会在后台重新向量化。用 make logs 观察进度。")
        return len(targets)
    finally:
        await db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="换 embedding 提供商后重建索引向量")
    parser.add_argument("--tenant-id", default=None, help="只处理某个租户")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要处理的文档，不修改数据")
    args = parser.parse_args()
    tenant_id = uuid.UUID(args.tenant_id) if args.tenant_id else None
    asyncio.run(reembed(tenant_id=tenant_id, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
