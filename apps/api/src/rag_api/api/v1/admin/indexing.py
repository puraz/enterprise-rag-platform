"""管理面：索引重建 / 版本切换 / 回滚（零停机重建的运维入口）。

## 为什么这是一个**三条独立接口**而不是一条"重建"接口

零停机重建是三段，每段的风险与确认要求不同：

| 阶段 | 接口 | 风险 | 谁该做 |
|---|---|---|---|
| ① 回填 | `POST .../index-rebuild` | 低（不切换，检索仍走旧版本） | 运营 |
| ② 切换 | `POST .../index-versions/{id}/activate` | **高**（改变检索结果） | 超管 / 租户管理员 |
| ③ 回滚 | `POST .../index-versions/{id}/rollback` | 中（切回旧版本） | 超管 / 租户管理员 |

把三段合成一条会让"回填完成后自动切换"成为默认——而回填完成
**不等于**质量合格（新模型可能召回更差）。人工看质量再切换是刻意的。

## 为什么触发重建返回 202 而不是 200

回填要跑几十分钟（十万 chunk × embedding 调用）。同步等待会：

- 占住一个 HTTP 连接（网关超时 30~60 秒，直接 504）；
- 用户界面卡死，且失败后无法重试。

所以返回 202 + `task_id` + `index_version_id`，让前端轮询。

## 为什么切换/回滚是同步的

切换只是一条 UPDATE（把 ACTIVE 指向新版本），**秒级**。
做成异步会让"切了没有"多一次不确定窗口——而这是个不该等的操作。

## 并发保护

同一知识库同时只允许一个 `building` 版本（`find_building` 检查 + 409）。
并行回填会争抢 GPU/API 配额，且两个版本都半成品时无法判断该切哪个。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.openapi import ADMIN_ENDPOINT, ASYNC_ENDPOINT
from rag_api.permissions import AdminCapability, AdminContext, get_admin_context
from rag_api.permissions.admin import get_admin_db, require_admin_capability
from rag_api.schemas.admin import (
    to_index_rebuild_response,
    to_index_version,
)
from rag_contracts.api.admin import (
    ActivateIndexVersionRequest,
    CreateIndexRebuildRequest,
    IndexRebuildResponse,
    IndexVersionItem,
    RollbackIndexVersionRequest,
)
from rag_contracts.enums import (
    AuditAction,
    AuditOutcome,
    AuditTargetType,
    IndexVersionStatus,
    KnowledgeBaseStatus,
    TaskType,
)
from rag_core.errors import ConflictError, NotFoundError, ValidationFailedError
from rag_db.repositories import (
    AuditRepository,
    IndexVersionRepository,
    KnowledgeBaseRepository,
    TaskRepository,
)

router = APIRouter(prefix="/knowledge-bases", tags=["admin"])


@router.post(
    "/{kb_id}/index-rebuild",
    response_model=IndexRebuildResponse,
    status_code=202,
    summary="发起索引重建（零停机）",
    description=(
        "创建一个新的 `building` 版本并异步入队回填。\n\n"
        "**这是零停机的关键**：回填期间向量写入**旁表的新版本行**，"
        "旧版本的向量完全不动，所以检索继续走旧 ACTIVE 版本，"
        "用户完全无感。\n\n"
        "回填完成后版本变成 `ready`，**需要人工调用 activate 才会切换**——"
        "回填完成不等于质量合格（新模型可能召回更差），人工验收是刻意的。\n\n"
        "**并发约束**：同一知识库同时只允许一个 `building` 版本，"
        "重复发起返回 409。并行回填会争抢 GPU/API 配额且无法判断该切哪个。\n\n"
        "响应 `202` + `task_id`（轮询进度）+ `index_version_id`（后续切换用）。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.OPERATE))],
    responses={**ADMIN_ENDPOINT, **ASYNC_ENDPOINT},
)
async def create_index_rebuild(
    kb_id: uuid.UUID,
    payload: CreateIndexRebuildRequest,
    ctx: AdminContext = Depends(get_admin_context),
    session: AsyncSession = Depends(get_admin_db),
) -> IndexRebuildResponse:
    """创建 building 版本并入队回填任务。"""
    kb = await KnowledgeBaseRepository(session).get(tenant_id=ctx.data_tenant_id, kb_id=kb_id)
    if kb is None:
        raise NotFoundError("知识库不存在")
    if kb.status != KnowledgeBaseStatus.ACTIVE.value:
        raise ConflictError(
            f"知识库当前状态为 {kb.status}，不能重建索引",
            details={"status": str(kb.status)},
        )

    versions = IndexVersionRepository(session)

    # 并发保护：同一库只允许一个 building 版本
    building = await versions.find_building(knowledge_base_id=kb_id)
    if building is not None:
        raise ConflictError(
            "该知识库已有正在构建的索引版本，请等待完成或等待其失败后再发起",
            details={
                "building_version_id": str(building.id),
                "version_tag": building.version_tag,
                "status": str(building.status),
            },
        )

    from rag_core.settings import get_settings

    settings = get_settings()

    # version_tag 由服务端生成：`v{序号}-{嵌入模型短名}`。
    #
    # 不用时间戳：tag 是给人看的，序号更容易沟通（"切到 v3"）。
    # 模型名带上是因为"换模型重建"是最常见的场景，tag 里直接可见。
    seq = await versions.count_chunks_by_version_tag(knowledge_base_id=kb_id) + 1
    model_short = settings.embedding.model_name.replace("/", "-")[:24]
    version_tag = f"v{seq}-{model_short}"

    # 当前参与检索的 chunk 数：作为"回填完成"的判定基准。
    # 用事实（COUNT）而不是缓存字段。
    expected = await _count_kb_chunks(session, tenant_id=ctx.data_tenant_id, kb_id=kb_id)

    version = await versions.create(
        tenant_id=ctx.data_tenant_id,
        knowledge_base_id=kb_id,
        version_tag=version_tag,
        embedding_model=settings.embedding.model_name,
        embedding_dim=settings.embedding.dimension,
        index_params={"m": 16, "ef_construction": 64},
        expected_chunk_count=expected,
    )

    # 入队回填任务（与版本创建同事务：只建版本不入队会让它永远停在 building）
    task, _created = await TaskRepository(session).enqueue(
        tenant_id=ctx.data_tenant_id,
        task_type=TaskType.REBUILD_INDEX.value,
        # 幂等键含版本 ID：同一个版本重复入队会被复用，
        # 不同版本天然得到独立任务。不用 uuid4——那会让"重试入队"
        # 产生第二个任务，两个 handler 同时回填同一版本。
        idempotency_key=f"rebuild:{version.id}",
        payload={
            "knowledge_base_id": str(kb_id),
            "index_version_id": str(version.id),
        },
        knowledge_base_id=kb_id,
    )

    await AuditRepository(session).record(
        tenant_id=ctx.data_tenant_id,
        action=AuditAction.ADMIN_INDEX_REBUILD,
        outcome=AuditOutcome.SUCCESS,
        actor_user_id=ctx.user_id,
        target_type=AuditTargetType.KNOWLEDGE_BASE,
        target_id=kb_id,
        target_name=kb.name,
        details={
            "index_version_id": str(version.id),
            "version_tag": version_tag,
            "task_id": str(task.id),
            "embedding_model": settings.embedding.model_name,
            "expected_chunk_count": expected,
            "reason": payload.reason or "",
            "platform_role": ctx.platform_role.value,
        },
    )

    return to_index_rebuild_response(
        task_id=task.id,
        index_version_id=version.id,
        version_tag=version_tag,
    )


@router.post(
    "/{kb_id}/index-versions/{version_id}/activate",
    response_model=IndexVersionItem,
    summary="切换 ACTIVE 索引版本",
    description=(
        "把指定版本切为当前生效版本（alias 切换）。**秒级完成**："
        "只改一行状态，在途查询继续用旧版本跑完，新请求用新版本。\n\n"
        "**切换前硬校验**：新版本的向量数必须 ≥ 预期 chunk 数。"
        "未回填完就切换会让检索**静默丢失内容**——用户会得到"
        "「知识库里查不到这份文档」，而实际上文档在库里，只是向量没算完。"
        "这比切换失败严重得多，所以这里拒绝而不是尽力而为。\n\n"
        "`confirm_version_tag` 必须与目标版本完全一致（防前端拿错 ID）。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.ADMIN))],
    responses=ADMIN_ENDPOINT,
)
async def activate_index_version(
    kb_id: uuid.UUID,
    version_id: uuid.UUID,
    payload: ActivateIndexVersionRequest,
    ctx: AdminContext = Depends(get_admin_context),
    session: AsyncSession = Depends(get_admin_db),
) -> IndexVersionItem:
    """切换 ACTIVE（含回填完成度硬校验）。"""
    versions = IndexVersionRepository(session)

    version = await versions.get(knowledge_base_id=kb_id, version_id=version_id)
    if version is None:
        raise NotFoundError("索引版本不存在")
    if version.version_tag != payload.confirm_version_tag:
        # 400 而不是 409：这不是状态冲突，是请求参数与目标不匹配
        raise ValidationFailedError(
            "confirm_version_tag 与目标版本不一致",
            details={
                "expected": version.version_tag,
                "provided": payload.confirm_version_tag,
            },
        )
    if version.status == IndexVersionStatus.ACTIVE.value:
        raise ConflictError("该版本已是当前生效版本")

    # 完成度校验：以旁表真实行数为准（不是 version.chunk_count 计数器）
    actual = await versions.count_version_chunks(index_version_id=version_id)
    expected = version.expected_chunk_count
    if expected is not None and actual < expected:
        raise ConflictError(
            f"回填未完成，拒绝切换：预期 {expected} 个向量，实际 {actual}。"
            "未完成就切换会导致检索静默丢失内容",
            details={"expected": expected, "actual": actual},
        )

    await versions.activate(knowledge_base_id=kb_id, version_id=version_id)

    await AuditRepository(session).record(
        tenant_id=ctx.data_tenant_id,
        action=AuditAction.ADMIN_INDEX_ACTIVATE,
        outcome=AuditOutcome.SUCCESS,
        actor_user_id=ctx.user_id,
        target_type=AuditTargetType.KNOWLEDGE_BASE,
        target_id=kb_id,
        details={
            "index_version_id": str(version_id),
            "version_tag": version.version_tag,
            "vector_count": actual,
            "expected": expected if expected is not None else -1,
            "reason": payload.reason or "",
            "platform_role": ctx.platform_role.value,
        },
    )

    refreshed = await versions.get(knowledge_base_id=kb_id, version_id=version_id)
    assert refreshed is not None  # 刚更新过，不可能不存在
    return to_index_version(refreshed)


@router.post(
    "/{kb_id}/index-versions/{version_id}/rollback",
    response_model=IndexVersionItem,
    summary="回滚到指定索引版本",
    description=(
        "把 ACTIVE 指回一个历史版本。**秒级生效**：旧版本的向量在切换时"
        "没有删除，只是把指针改回去。\n\n"
        "这也是为什么「回收旧版本」必须延迟观察——立刻回收会让回滚"
        "失去可能。\n\n"
        "目标版本必须是 `ready` / `active` / `retired`：`building` 未完成、"
        "`failed` 没数据，都不能作为回滚目标。\n\n"
        "`confirm_version_tag` 同样需要完全一致。"
    ),
    dependencies=[Depends(require_admin_capability(AdminCapability.ADMIN))],
    responses=ADMIN_ENDPOINT,
)
async def rollback_index_version(
    kb_id: uuid.UUID,
    version_id: uuid.UUID,
    payload: RollbackIndexVersionRequest,
    ctx: AdminContext = Depends(get_admin_context),
    session: AsyncSession = Depends(get_admin_db),
) -> IndexVersionItem:
    """回滚 ALIAS 指向（不删数据）。"""
    versions = IndexVersionRepository(session)

    version = await versions.get(knowledge_base_id=kb_id, version_id=version_id)
    if version is None:
        raise NotFoundError("索引版本不存在")
    if version.version_tag != payload.confirm_version_tag:
        raise ValidationFailedError(
            "confirm_version_tag 与目标版本不一致",
            details={
                "expected": version.version_tag,
                "provided": payload.confirm_version_tag,
            },
        )

    # 只有"有向量"的版本才能作为回滚目标。
    #
    # building 是半成品（切换等于丢失未回填部分）；
    # failed 可能一行向量都没有（切换等于清空检索）。
    usable = {
        IndexVersionStatus.READY.value,
        IndexVersionStatus.ACTIVE.value,
        IndexVersionStatus.RETIRED.value,
    }
    if version.status not in usable:
        raise ConflictError(
            f"目标版本状态为 {version.status}，不能作为回滚目标",
            details={
                "status": str(version.status),
                # details 只接受标量（契约约束）：用逗号拼接而不是 list
                "usable_statuses": ",".join(sorted(usable)),
            },
        )

    await versions.activate(knowledge_base_id=kb_id, version_id=version_id)

    await AuditRepository(session).record(
        tenant_id=ctx.data_tenant_id,
        action=AuditAction.ADMIN_INDEX_REBUILD,
        outcome=AuditOutcome.SUCCESS,
        actor_user_id=ctx.user_id,
        target_type=AuditTargetType.KNOWLEDGE_BASE,
        target_id=kb_id,
        details={
            "action": "rollback",
            "index_version_id": str(version_id),
            "version_tag": version.version_tag,
            "reason": payload.reason or "",
            "platform_role": ctx.platform_role.value,
        },
    )

    refreshed = await versions.get(knowledge_base_id=kb_id, version_id=version_id)
    assert refreshed is not None
    return to_index_version(refreshed)


async def _count_kb_chunks(session: AsyncSession, *, tenant_id: uuid.UUID, kb_id: uuid.UUID) -> int:
    """知识库当前参与检索的 chunk 数（叶子块）。

    这是"回填完成"的判定基准。**必须只算叶子块**：
    父块（`chunk_level > 0`）不向量化，算进去会让预期永远达不到。
    """
    from sqlalchemy import func, select

    from rag_db.models.chunk import Chunk

    result = await session.execute(
        select(func.count())
        .select_from(Chunk)
        .where(
            Chunk.tenant_id == tenant_id,
            Chunk.knowledge_base_id == kb_id,
            Chunk.is_current.is_(True),
            Chunk.chunk_level == 0,
        )
    )
    return int(result.scalar_one())


__all__ = ["router"]
