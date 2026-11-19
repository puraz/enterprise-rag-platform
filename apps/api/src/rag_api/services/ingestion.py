"""文档入库编排：把已上传的 blob 变成可处理的文档。

## 这是 A6 与 A8 之间缺失的那一环

两条上游路径最终汇到同一点：

    # 首次上传
    POST /uploads → PUT parts × N → POST /uploads/{id}/complete
                                                        │
                                                        ▼ blob_id
    # 秒传（内容相同，无需上传）
    POST /uploads → (deduplicated=true) ─────────────▶ blob_id
                                                        │
                                                        ▼
                                    [本模块] IngestionService.create_document
                                                        │
                            ┌───────────────────────────┼──────────────────┐
                            ▼                           ▼                  ▼
                     documents(PENDING)        document_versions      tasks(parse_document)
                                                                           │
                                                                           ▼
                                                              worker: parse → chunk → embed → ready

在 A6 与 A8 都各自完成、测试也通过的情况下，这条链**仍然是断的**：
`UploadService.complete()` 只返回 `Blob`，`apps/` 下没有任何代码
enqueue `PARSE_DOCUMENT`。表现是"上传成功但文档永远不出现"，
且没有任何报错——这是最难排查的一类缺口（每个组件都对，组装错了）。

## 为什么这些写入必须在**同一个事务**里

五件事彼此依赖，缺任何一件都会留下坏状态：

| 缺哪一步 | 后果 |
|---|---|
| 建文档不建版本 | 文档存在但没有 blob 指针，worker 无法解析 |
| 建版本不设 current | `is_current` 部分唯一索引找不到它，检索拿不到 |
| 不入队任务 | 文档永远停在 pending，且没有任务能推进它 |
| blob 引用计数不加 | 原文档删除时 blob 被提前回收 → "文档还在但文件没了" |
| 文档计数不加 | 知识库列表显示"0 个文档"，用户以为上传失败 |

所以路由层用 `get_db` 的单一事务覆盖全部，本模块**只 flush 不 commit**
（与 worker handler 的约定一致，见 AGENTS.md 4.11）。

## 为什么任务用 outbox 之外的方式入队

`docs/architecture` 里 outbox 的作用是"保证消息最终会被发出"。
但本项目的 worker **每 2 秒轮询 `tasks` 表兜底**（见 AGENTS.md 4.11：
"数据库是任务的权威来源，队列只是通知"）。所以只要 `tasks` 行与业务
数据同事务写入，任务就不会丢——outbox 是"降低延迟"的优化，
不是正确性的必要条件。

当前 `dispatch_outbox` handler 尚未实现（`handlers/__init__.py` 里
会 warning）。在这个前提下仍然**写 outbox 行**会让表持续堆积，
所以这里只写 `tasks`，等 outbox 转发实装后再补。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.permissions.acl import LEVEL_RANK, PermissionContext
from rag_api.services.upload_security import document_format_for
from rag_contracts.enums import (
    AuditAction,
    AuditTargetType,
    BlobStatus,
    ConfidentialityLevel,
    DocumentStatus,
    DocumentVersionStatus,
    KnowledgeBaseStatus,
    TaskType,
)
from rag_core.errors import (
    ClassificationDeniedError,
    ConflictError,
    NotFoundError,
    ValidationFailedError,
)
from rag_core.logging import get_logger
from rag_db.models.blob import Blob
from rag_db.models.document import Document
from rag_db.models.knowledge import KnowledgeBase
from rag_db.repositories import (
    BlobRepository,
    DocumentRepository,
    KnowledgeBaseRepository,
    TaskRepository,
)

log = get_logger(__name__)

# 文件名去掉扩展名后的标题长度上限。
# 与 `documents.title` 的 String(500) 一致——超长会在 flush 时被数据库拒绝，
# 而那时错误信息是"value too long for type character varying(500)"，
# 用户看不出是文件名太长。
_MAX_TITLE_LENGTH = 500


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """建文档的结果。"""

    document: Document
    version_id: uuid.UUID
    task_id: uuid.UUID
    """`PARSE_DOCUMENT` 任务 ID。前端据此轮询，也是排查卡住的入口。"""

    deduplicated_blob: bool
    """该 blob 是否已被其它文档引用（同一文件被多次登记）。"""


class IngestionService:
    """文档入库编排。

    依赖注入 `session`，其余仓储内部构造——它们都是同一 session 上的
    薄包装，拆成构造参数只会让调用方写五行一样的代码。
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._kbs = KnowledgeBaseRepository(session)
        self._docs = DocumentRepository(session)
        self._blobs = BlobRepository(session)
        self._tasks = TaskRepository(session)

    async def create_document(
        self,
        *,
        ctx: PermissionContext,
        blob_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        original_filename: str,
        title: str | None = None,
        confidentiality: ConfidentialityLevel | None = None,
    ) -> IngestionResult:
        """把 blob 登记为文档并触发解析。

        Args:
            ctx: 权限上下文。`user_id` 记入 `created_by` 与审计，
                `tenant_id` 用于隔离。
            blob_id: 上传完成的物理文件。
            knowledge_base_id: 目标库。
            original_filename: 原始文件名，决定格式与审计记录。
            title: 省略时用文件名去扩展名。
            confidentiality: 省略时继承库级密级。

        Raises:
            NotFoundError: 知识库或 blob 不存在（或不属于本租户）。
            ConflictError: 知识库已归档。
            ClassificationDeniedError: 文档密级低于库级密级。
            ValidationFailedError: 文件格式不支持。
        """
        kb = await self._require_kb(tenant_id=ctx.tenant_id, kb_id=knowledge_base_id)
        blob = await self._require_blob(tenant_id=ctx.tenant_id, blob_id=blob_id)

        effective_level = self._resolve_confidentiality(requested=confidentiality, kb=kb)
        doc_format = self._resolve_format(original_filename)
        doc_title = _derive_title(title, original_filename)

        # ---- 建文档 ----
        document = await self._docs.create(
            tenant_id=ctx.tenant_id,
            knowledge_base_id=kb.id,
            title=doc_title,
            original_filename=original_filename,
            format=doc_format,
            created_by=ctx.user_id,
            confidentiality=effective_level.value,
            status=DocumentStatus.PENDING,
        )
        # size_bytes 来自 blob（文件真实大小）。放在 create 之后单独赋值
        # 而不是扩 `create` 的参数：只有上传来源才有 blob，
        # 将来 URL/API 来源的文档没有 size，扩参数会让签名变脏。
        document.size_bytes = blob.size_bytes

        # ---- 建版本 ----
        #
        # 版本号必须走 `allocate_version_no`（原子 UPDATE ... RETURNING）：
        # 并发给同一文档加版本时，Python 侧 `latest + 1` 会拿到重复号。
        version_no = await self._docs.allocate_version_no(doc_id=document.id)
        version = await self._docs.create_version(
            tenant_id=ctx.tenant_id,
            document_id=document.id,
            version_no=version_no,
            blob_id=blob.id,
            created_by=ctx.user_id,
            # **切分参数在此快照**：从库的当前配置复制到版本上。
            # 之后改库配置不影响这个版本——否则"重跑同一版本"会产出
            # 不同的 chunk，A10 的版本回滚就失去意义。
            chunking_strategy=kb.default_chunking_strategy,
            chunking_config={
                "chunk_size": kb.default_chunk_size,
                "chunk_overlap": kb.default_chunk_overlap,
            },
            status=DocumentVersionStatus.PROCESSING,
        )
        await self._docs.mark_version_current(
            document_id=document.id, version_id=version.id, version_no=version_no
        )

        # ---- 引用计数与统计 ----
        #
        # blob 的引用计数决定它能否被物理回收。少加一次 →
        # 删掉其它文档后这个 blob 被回收，本文档就变成"有元数据无文件"。
        #
        # **计数基准不是 0**：`UploadService.complete()` 已经 +1（那次是
        # "上传完成持有这个 blob"）。所以第一次建文档后计数是 2，
        # 用 `new_ref_count > 1` 判断"是否重复"会永远为真——实测抓到。
        await self._blobs.increment_ref(blob_id=blob.id)
        await self._kbs.increment_document_count(kb_id=kb.id, delta=1)

        # ---- 是否重复引用同一个 blob ----
        #
        # 直接数"有多少文档指向这个 blob"：建完当前文档后 > 1 就说明
        # 之前已有别的文档用了同一份内容（秒传或重复上传）。
        #
        # 为什么不用 ref_count：它把"上传会话持有" "文档持有" 混在一起，
        # 数值本身没有业务含义。文档侧计数才是用户理解的"重复"。
        deduplicated = (
            await self._count_documents_for_blob(tenant_id=ctx.tenant_id, blob_id=blob.id) > 1
        )

        # ---- 入队解析任务 ----
        #
        # 幂等键**必须含 version_id**：
        # - 用 document_id 会让"同一文档传新版本"复用一个已有任务，
        #   新版本永远不被解析；
        # - 含 version 则重试同一版本时不会产生第二个任务（幂等），
        #   而新版本天然得到独立任务。
        task, created = await self._tasks.enqueue(
            tenant_id=ctx.tenant_id,
            task_type=TaskType.PARSE_DOCUMENT.value,
            idempotency_key=f"parse:{version.id}",
            payload={
                "document_id": str(document.id),
                "document_version_id": str(version.id),
            },
            document_id=document.id,
            document_version_id=version.id,
            knowledge_base_id=kb.id,
        )

        # ---- 审计（同事务）----
        # 上传是合规关注点（"谁把什么文件放进了知识库"），必须留痕。
        # details 只放 ID 与格式，**不放文件名**——文件名可能含
        # 人命/项目代号，而审计表保留 180 天。
        await self._audit_upload(
            ctx=ctx,
            document=document,
            kb=kb,
            blob=blob,
            deduplicated=deduplicated,
        )

        log.info(
            "document_ingested",
            document_id=str(document.id),
            version_id=str(version.id),
            task_id=str(task.id),
            task_created=created,
            knowledge_base_id=str(kb.id),
            format=doc_format,
            size_bytes=blob.size_bytes,
            deduplicated=deduplicated,
        )

        return IngestionResult(
            document=document,
            version_id=version.id,
            task_id=task.id,
            deduplicated_blob=deduplicated,
        )

    # ------------------------------------------------------------ 校验步骤

    async def _count_documents_for_blob(self, *, tenant_id: uuid.UUID, blob_id: uuid.UUID) -> int:
        """数有多少个未删除文档引用这个 blob。

        "重复上传同一文件"的判据。**从文档侧数**而不是看 `ref_count`：
        后者把"上传会话持有"与"文档持有"混在一起，数值本身不可解释
        （上传完成后 ref_count 就已经是 1）。详见调用点的说明。
        """
        from sqlalchemy import func, select

        from rag_db.models.document import DocumentVersion

        result = await self._session.execute(
            select(func.count(func.distinct(DocumentVersion.document_id)))
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                DocumentVersion.blob_id == blob_id,
                Document.tenant_id == tenant_id,
                Document.deleted_at.is_(None),
            )
        )
        return int(result.scalar_one())

    async def _require_kb(self, *, tenant_id: uuid.UUID, kb_id: uuid.UUID) -> KnowledgeBase:
        """取知识库并校验可用。

        ## 为什么"不存在"与"已归档"分别报错

        - 不存在 → 404（可能 ID 写错，也可能跨租户探测——不泄漏区别）；
        - 已归档 → 409（库确实存在，只是状态不允许新增文档）。

        合并成 404 会让用户对着一张确实存在的库反复刷新，
        以为界面出了问题。
        """
        kb = await self._kbs.get(tenant_id=tenant_id, kb_id=kb_id)
        if kb is None:
            raise NotFoundError("知识库不存在")
        if kb.status != KnowledgeBaseStatus.ACTIVE.value:
            raise ConflictError(
                "知识库已归档，无法新增文档",
                details={"status": str(kb.status)},
            )
        return kb

    async def _require_blob(self, *, tenant_id: uuid.UUID, blob_id: uuid.UUID) -> Blob:
        """取 blob 并校验它确实可读。

        ## 为什么要校验 `status == complete`

        `blobs.status` 有 `pending`（分片会话已建、数据未完整）这个状态。
        拿一个 pending 的 blob 去建文档，worker 解析时会读到半个文件——
        PyMuPDF 会抛"文件损坏"，任务变成 PERMANENT 失败，
        而真正的根因（上传没完成）已经被埋在错误信息里。

        在这里拦住，错误信息就能直指"上传尚未完成"。

        ## 为什么 `tenant_id` 必须进 WHERE

        只用 `blob_id` 查询等于把租户隔离交给调用方自觉。跨租户拿别人的
        blob 建文档 = 直接的数据泄漏。RLS 是最后一道防线，查询本身
        也要写对。
        """
        from sqlalchemy import select

        result = await self._session.execute(
            select(Blob).where(Blob.id == blob_id, Blob.tenant_id == tenant_id).limit(1)
        )
        blob = result.scalar_one_or_none()
        if blob is None:
            raise NotFoundError("文件不存在")
        if blob.status == BlobStatus.MISSING.value:
            raise ValidationFailedError("文件已在对象存储中丢失，请重新上传")
        if blob.status != BlobStatus.COMPLETE.value:
            raise ValidationFailedError(
                f"文件尚未上传完成（当前状态：{blob.status}）",
                details={"blob_status": str(blob.status)},
            )
        return blob

    def _resolve_confidentiality(
        self, *, requested: ConfidentialityLevel | None, kb: KnowledgeBase
    ) -> ConfidentialityLevel:
        """确定文档密级，并校验它不松于库级。

        ## 不变量：文档密级 ≥ 库级密级

        这个约束写在 `models/document.py` 的注释里（"不得低于所属库的密级"），
        但此前没有任何代码实现它。

        不校验的后果：库里标着"机密"，但某人上传文档时填了"公开"，
        于是这份文档对密级只有 internal 的用户可见——**库的密级被单条
        文档绕过**。用户以为"这个库整体是机密的"，实际不是。

        ## 为什么用 `LEVEL_RANK` 而不是枚举定义顺序或字符串比较

        `confidential` < `internal` < `public` < `secret` 是按字母排的，
        与敏感程度无关。按字符串比较会把 `secret` 当成最低级——权限
        完全错，且不报错。`LEVEL_RANK` 是唯一的正确比较依据。
        """
        kb_level = ConfidentialityLevel(str(kb.confidentiality))
        if requested is None:
            return kb_level

        if LEVEL_RANK[requested] < LEVEL_RANK[kb_level]:
            raise ClassificationDeniedError(
                f"文档密级（{requested.value}）不能低于知识库密级（{kb_level.value}）"
            )
        return requested

    def _resolve_format(self, filename: str) -> str:
        """按文件名推导格式。不支持时抛 415 语义的异常。"""
        return document_format_for(filename)

    async def _audit_upload(
        self,
        *,
        ctx: PermissionContext,
        document: Document,
        kb: KnowledgeBase,
        blob: Blob,
        deduplicated: bool,
    ) -> None:
        """写上传审计（与业务同事务）。

        用局部 import：`AuditService` 依赖 `PermissionContext`，
        而 `permissions` 包又依赖 `rag_db`——顶层导入会在某些
        装配顺序下形成环。
        """
        from rag_api.services.audit import AuditService

        await AuditService(self._session).record(
            ctx=ctx,
            action=AuditAction.DOC_UPLOAD,
            target_type=AuditTargetType.DOCUMENT,
            target_id=document.id,
            target_name=document.title,
            details={
                "knowledge_base_id": str(kb.id),
                "format": str(document.format),
                "content_hash_prefix": blob.content_hash[:12],
                "size_bytes": blob.size_bytes,
                "confidentiality": str(document.confidentiality),
                "deduplicated": deduplicated,
            },
        )


def _derive_title(title: str | None, filename: str) -> str:
    """推导文档标题。

    默认用文件名去掉扩展名：用户上传 `2026年差旅制度.pdf`，
    标题就该是"2026年差旅制度"，而不是带 `.pdf` 的原始文件名。
    扩展名是格式信息，已经在 `format` 字段里表达过了。

    无扩展名（或只有 `foo.` 这种）时整串当标题。
    """
    if title is not None and title.strip():
        return title.strip()[:_MAX_TITLE_LENGTH]

    dot = filename.rfind(".")
    stem = filename[:dot] if dot > 0 else filename
    stem = stem.strip() or filename
    return stem[:_MAX_TITLE_LENGTH]


__all__ = ["IngestionResult", "IngestionService"]
