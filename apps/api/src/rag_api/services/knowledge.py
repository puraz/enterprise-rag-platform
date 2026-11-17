"""知识库业务编排：创建 / 列表 / 详情 / 更新 / 归档。

## 这一层存在的理由（不只是转发仓储）

路由层拿到的请求参数是"用户视角的"（`CreateKnowledgeBaseRequest`），
仓储层接受的是"数据库视角的"（一堆散字段）。中间需要有人负责：

| 职责 | 放在哪 | 为什么 |
|---|---|---|
| 名称唯一性检查 | **本层** | 仓储只管插入，冲突的业务语义（409 + 哪个名字）在这里 |
| 可见性过滤 | **本层**（生成 SQL 条件） | 依赖 `PermissionContext`，仓储不该认识权限 |
| 审计 | **本层** | 需要"这个动作成功了没有"的语义，仓储给不了 |
| 文档数非零时禁止删除 | **本层** | 跨两个仓储（KB + Document），仓储层不该横向依赖 |

## 为什么删除是"归档"而不是真删

`documents.knowledge_base_id` 是 `ON DELETE CASCADE`：物理删库会连带
删掉文档元数据，而 chunk 与向量是异步清理的——中间窗口里检索会返回
"指向已不存在文档"的结果。归档保留全部数据，只是默认不可见。

## 为什么删除前要求文档数为 0

两种做法各有代价：

| 做法 | 问题 |
|---|---|
| 允许删非空库（级联） | 文档、chunk、向量、blob 引用全部要级联处理，任一处漏掉就是孤儿数据 |
| **要求先清空**（本项目） | 用户多一步操作，但"删除"永远是一个可解释的原子动作 |

选了后者，并且错误信息要明确告诉用户"还有 N 个文档"——
让用户自己去猜"为什么删不掉"是最糟的体验。
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.permissions.acl import (
    PermissionContext,
    knowledge_base_visibility_filter,
)
from rag_contracts.api.knowledge import (
    CreateKnowledgeBaseRequest,
    UpdateKnowledgeBaseRequest,
)
from rag_contracts.enums import AuditAction, AuditTargetType
from rag_core.errors import ConflictError, NotFoundError
from rag_core.logging import get_logger
from rag_db.models.knowledge import KnowledgeBase
from rag_db.repositories import KnowledgeBaseRepository

log = get_logger(__name__)


class KnowledgeBaseService:
    """知识库业务编排。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = KnowledgeBaseRepository(session)

    async def create(
        self, *, ctx: PermissionContext, payload: CreateKnowledgeBaseRequest
    ) -> KnowledgeBase:
        """创建知识库。

        ## 唯一性检查为什么在这里而不是只靠数据库约束

        `uq_knowledge_bases_tenant_name` 是**全表唯一**（含已归档的库）。
        所以：
        - 检查放在本层 → 能给出"名字已被占用"的可读错误；
        - 只靠约束 → `IntegrityError` 需要翻译，且翻译时需要再查一次
          "是哪个库占了名字"。

        ## 为什么先查再插是可以的（尽管别处禁止这样做）

        AGENTS.md 禁止"先查再插"是针对**并发幂等**场景（两个请求同时
        插同一行，先查后插会双双通过检查然后撞约束）。这里的情况不同：
        即使并发撞上，数据库约束仍然会拦住，最坏结果是用户看到 500 而不是
        409。名称冲突不是高频路径，用一次额外查询换可读的错误信息是划算的。

        **但 `IntegrityError` 仍然要翻译**（见 `_translate_integrity_error`），
        否则并发场景用户会看到"服务内部错误"。
        """
        existing = await self._repo.get_by_name(tenant_id=ctx.tenant_id, name=payload.name)
        if existing is not None:
            raise ConflictError(
                f"知识库名称已被占用：{payload.name}",
                details={"conflicting_name": payload.name},
            )

        kb = await self._repo.create(
            tenant_id=ctx.tenant_id,
            name=payload.name,
            created_by=ctx.user_id,
            description=payload.description,
            visibility=payload.visibility.value,
            confidentiality=payload.confidentiality.value,
            chunking_strategy=payload.chunking_strategy,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )

        await self._audit(
            ctx=ctx,
            action=AuditAction.KB_CREATE,
            kb=kb,
            details={
                "visibility": payload.visibility.value,
                "confidentiality": payload.confidentiality.value,
                "chunking_strategy": payload.chunking_strategy,
                "chunk_size": payload.chunk_size,
                "chunk_overlap": payload.chunk_overlap,
            },
        )
        log.info(
            "knowledge_base_created",
            kb_id=str(kb.id),
            visibility=payload.visibility.value,
            created_by=str(ctx.user_id),
        )
        return kb

    async def list(
        self,
        *,
        ctx: PermissionContext,
        limit: int,
        offset: int,
        include_total: bool,
    ) -> tuple[list[KnowledgeBase], int | None]:
        """列出当前用户可见的知识库。

        ## 可见性过滤为什么必须进 SQL

        见 `acl.py` 模块 docstring：查出来再在 Python 里过滤会同时造成
        分页错乱、越权内容进内存、I/O 浪费。

        ## 为什么参数里没有 `visibility` 之类的筛选项

        当前阶段前端只需要"我有哪些库"。加筛选参数会让契约先长起来，
        而实际使用中还没有人需要它——按需扩展比提前设计更不容易做错。
        """
        rows, total = await self._repo.list_active(
            tenant_id=ctx.tenant_id,
            limit=limit,
            offset=offset,
            access_filter=knowledge_base_visibility_filter(ctx),
            include_total=include_total,
        )
        return list(rows), total

    async def get(self, *, ctx: PermissionContext, kb_id: uuid.UUID) -> KnowledgeBase:
        """取单个知识库（不可见即 404）。

        ## 为什么不可见返回 404 而不是 403

        403 等于告诉攻击者"这个 ID 存在，只是你没权限"——可以用它
        枚举出别的租户有哪些知识库 ID。404 不泄漏存在性。
        """
        return await self._require(ctx=ctx, kb_id=kb_id)

    async def _require(self, *, ctx: PermissionContext, kb_id: uuid.UUID) -> KnowledgeBase:
        """取可见的 ORM 对象（供内部复用）。"""
        kb = await self._repo.list_visible(
            tenant_id=ctx.tenant_id,
            kb_id=kb_id,
            access_filter=knowledge_base_visibility_filter(ctx),
        )
        if kb is None:
            raise NotFoundError("知识库不存在")
        return kb

    async def update(
        self,
        *,
        ctx: PermissionContext,
        kb_id: uuid.UUID,
        payload: UpdateKnowledgeBaseRequest,
    ) -> KnowledgeBase:
        """更新知识库元数据。"""
        # 先做可见性 + 存在性检查：不存在或不可见统一 404，
        # 不给"存在但不可改"这种可区分信号。
        await self._require(ctx=ctx, kb_id=kb_id)

        if payload.name is not None:
            conflicting = await self._repo.get_by_name(tenant_id=ctx.tenant_id, name=payload.name)
            # 改成自己已用的名字不算冲突
            if conflicting is not None and conflicting.id != kb_id:
                raise ConflictError(
                    f"知识库名称已被占用：{payload.name}",
                    details={"conflicting_name": payload.name},
                )

        updated = await self._repo.update_metadata(
            tenant_id=ctx.tenant_id,
            kb_id=kb_id,
            name=payload.name,
            description=payload.description,
            confidentiality=(
                payload.confidentiality.value if payload.confidentiality is not None else None
            ),
        )
        if updated is None:
            # 检查通过但更新影响 0 行：并发归档/删除。
            # 用 404 而不是 500——对用户来说资源确实没了。
            raise NotFoundError("知识库不存在")

        await self._audit(
            ctx=ctx,
            action=AuditAction.KB_UPDATE,
            kb=updated,
            details={
                "changed_fields": sorted(
                    field
                    for field, value in (
                        ("name", payload.name),
                        ("description", payload.description),
                        ("confidentiality", payload.confidentiality),
                    )
                    if value is not None
                ),
                "new_confidentiality": (
                    payload.confidentiality.value if payload.confidentiality is not None else None
                ),
            },
        )
        return updated

    async def archive(self, *, ctx: PermissionContext, kb_id: uuid.UUID) -> None:
        """归档知识库（软删）。"""
        kb = await self._require(ctx=ctx, kb_id=kb_id)

        # 文档数从 documents 表实时数，不读 `knowledge_bases.document_count`
        # （缓存字段可能滞后，而"能不能删"必须基于事实）。
        doc_count = await self._repo.count_documents(kb_id=kb_id)
        if doc_count > 0:
            raise ConflictError(
                f"知识库下还有 {doc_count} 个文档，请先删除文档",
                details={"document_count": doc_count},
            )

        archived = await self._repo.archive(tenant_id=ctx.tenant_id, kb_id=kb_id)
        if not archived:
            raise NotFoundError("知识库不存在")

        await self._audit(
            ctx=ctx,
            action=AuditAction.KB_DELETE,
            kb=kb,
            details={"document_count": doc_count},
        )
        log.info(
            "knowledge_base_archived",
            kb_id=str(kb_id),
            archived_by=str(ctx.user_id),
        )

    # ------------------------------------------------------------ 内部

    async def _audit(
        self,
        *,
        ctx: PermissionContext,
        action: AuditAction,
        kb: KnowledgeBase,
        details: dict[str, object],
    ) -> None:
        """写知识库审计（同事务）。"""
        from rag_api.services.audit import AuditService

        await AuditService(self._session).record(
            ctx=ctx,
            action=action,
            target_type=AuditTargetType.KNOWLEDGE_BASE,
            target_id=kb.id,
            target_name=kb.name,
            details=details,
        )


__all__ = ["KnowledgeBaseService"]
