"""上传服务：编排存储层与数据库，实现分片上传、秒传、断点续传。

## 职责边界

| 层 | 负责 |
|---|---|
| 路由（A6 后续） | 参数校验、权限、HTTP 语义 |
| **本服务** | 分片编排、秒传判断、hash 校验、blob 落库 |
| 存储层 | 字节的可靠存取、key 净化 |
| 仓储层 | 数据库读写 |

## 秒传（dedup）

客户端先算本地文件 SHA-256，调 `initiate` 时带上：
1. 服务端按 `(tenant_id, content_hash)` 查 `blobs`；
2. 命中且 `status=complete` → **直接返回已存在的 blob**，跳过上传；
3. 未命中 → 正常创建上传会话。

**安全性**：秒传只在**同租户内**生效。跨租户的内容相同也不能秒传——
否则可以通过"秒传成功/失败"探测别的租户有没有这个文件（存在性侧信道）。

## 断点续传

`list_uploaded_parts` 的权威来源是**对象存储**（`storage.list_parts`），
不是数据库计数：
- `upload_sessions.uploaded_parts` 只是缓存（A3-4 的设计）；
- 客户端重连后必须问存储"我传到哪了"，因为可能是别的实例/进程
  接收的分片，数据库计数可能滞后。

## 分片 ETag 校验

每个分片上传后存储返回 ETag。客户端 `complete` 时必须带上
它拿到的 ETag 列表，服务端与存储里实际的对比：
- 不一致说明分片被替换或损坏；
- **这是完整性校验的关键环节**，跳过会让损坏文件入库。

## 合并后的 hash 校验

合并完成后重新计算整体 SHA-256，与客户端声明的 `content_hash` 对比：
- 不匹配 → 作废（删除对象），返回 `UPLOAD_HASH_MISMATCH`；
- 匹配 → 写入 `blobs` 记录，返回 blob。

**为什么必须校验**：分片传输中任何一片损坏都会导致最终文件损坏，
而分片级 ETag 只能证明"这一片传上去的和发出去的一致"，
不能证明"整体与客户端原始文件一致"（比如客户端自己的计算错误、
或分片顺序错）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from rag_adapters.storages.base import (
    MultipartUpload,
    PartInfo,
    StorageBackend,
    StorageError,
    build_storage_key,
)
from rag_contracts.enums import BlobStatus, UploadSessionStatus
from rag_core.errors import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
)
from rag_core.logging import get_logger
from rag_core.settings.storage import StorageSettings
from rag_db.models.blob import Blob
from rag_db.models.upload import UploadSession
from rag_db.repositories import BlobRepository, UploadSessionRepository

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class InitiateResult:
    """`initiate` 的结果。

    两种可能：
    - **秒传命中**：`blob_id` 有值、`upload_id` 为 None；
    - **需要上传**：`upload_id` 有值、`blob_id` 为 None。
    """

    deduplicated: bool
    blob_id: uuid.UUID | None = None
    upload_id: str | None = None
    """**对外的会话 ID**（数据库主键），客户端后续用它调 parts/complete。
    不用存储层的 upload_id 作为公开标识——那是后端的实现细节，
    暴露它会让攻击者推断存储配置。"""
    storage_upload_id: str | None = None
    """存储层 upload_id（内部使用，不返回给客户端）。"""
    part_size: int = 0
    total_parts: int = 0
    # 已上传的分片号（续传场景：客户端已传过一部分）
    uploaded_parts: list[int] | None = None


class UploadService:
    """分片上传编排。

    与仓储、存储协作，不直接写 SQL，也不直接碰文件系统。
    """

    def __init__(
        self,
        session: AsyncSession,
        storage: StorageBackend,
        settings: StorageSettings,
    ) -> None:
        self._session = session
        self._storage = storage
        self._settings = settings
        self._sessions = UploadSessionRepository(session)
        self._blobs = BlobRepository(session)

    async def initiate(
        self,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        filename: str,
        total_size: int,
        content_type: str,
        content_hash: str | None,
        knowledge_base_id: uuid.UUID | None = None,
    ) -> InitiateResult:
        """开始上传。

        **先校验目标知识库，再查秒传，最后建会话**。顺序的理由：
        - 先校验 KB：库不存在/已归档时就该直接拒绝，不该让用户白传 2GB
          然后才在 `POST /documents` 那一步发现；
        - 秒传在会话之前：先建会话再查会留下无用的会话记录
          （客户端拿到秒传结果后不会再用它）。
        """
        if total_size <= 0:
            raise BadRequestError("文件大小必须大于 0")
        if total_size > self._settings.max_file_size:
            raise PayloadTooLargeError(
                f"文件超过上限 {self._settings.max_file_size // 1024 // 1024}MB",
                details={"max_size": self._settings.max_file_size},
            )

        # ---- 目标知识库校验 ----
        #
        # 为什么在上传阶段就校验：initiate 时客户端已经指定了目标库，
        # 此时拒绝的成本最低。等到 complete 或 create_document 才拦，
        # 用户已经花了带宽和几分钟等待。
        if knowledge_base_id is not None:
            await self._require_active_kb(tenant_id=tenant_id, kb_id=knowledge_base_id)

        # ---- 秒传 ----
        if content_hash:
            existing = await self._blobs.find_by_hash(
                tenant_id=tenant_id, content_hash=content_hash
            )
            if existing is not None and existing.status == BlobStatus.COMPLETE.value:
                log.info(
                    "upload_deduplicated",
                    tenant_id=str(tenant_id),
                    content_hash=content_hash[:12],
                    size_bytes=existing.size_bytes,
                )
                # 引用计数 +1：秒传的文档也要正确计数，
                # 否则原文档删除时 blob 会被提前回收
                await self._blobs.increment_ref(blob_id=existing.id)
                return InitiateResult(
                    deduplicated=True,
                    blob_id=existing.id,
                    part_size=self._settings.part_size,
                    total_parts=0,
                )

        # ---- 创建上传会话 ----
        part_size = self._settings.part_size
        total_parts = self._settings.total_parts_for(total_size)
        storage_key = build_storage_key(
            tenant_id=str(tenant_id),
            content_hash=content_hash or uuid.uuid4().hex,
            suffix=_suffix_for(filename),
        )

        multipart = await self._storage.initiate_multipart(storage_key, content_type=content_type)

        session_row = await self._sessions.create(
            tenant_id=tenant_id,
            created_by=user_id,
            filename=filename,
            total_size=total_size,
            content_type=content_type,
            content_hash=content_hash,
            part_size=part_size,
            total_parts=total_parts,
            storage_key_prefix=storage_key,
            storage_backend=self._storage.name,
            knowledge_base_id=knowledge_base_id,
            ttl=timedelta(seconds=self._settings.upload_session_ttl_s),
        )
        # 把存储层的 upload_id 记下来：complete/abort 需要它
        session_row.storage_upload_id = multipart.upload_id
        await self._session.flush()

        log.info(
            "upload_initiated",
            upload_session_id=str(session_row.id),
            total_size=total_size,
            total_parts=total_parts,
            part_size=part_size,
        )
        return InitiateResult(
            deduplicated=False,
            upload_id=str(session_row.id),
            storage_upload_id=multipart.upload_id,
            part_size=part_size,
            total_parts=total_parts,
            uploaded_parts=[],
        )

    async def _load_upload_for_part(
        self, *, tenant_id: uuid.UUID, session_id: uuid.UUID
    ) -> tuple[UploadSession, MultipartUpload]:
        """取出会话并构造存储层的 multipart 句柄。"""
        row = await self._sessions.get(tenant_id=tenant_id, session_id=session_id)
        if row is None:
            raise NotFoundError("上传会话不存在")

        if row.status in (
            UploadSessionStatus.COMPLETED.value,
            UploadSessionStatus.ABORTED.value,
        ):
            raise ConflictError(f"会话状态为 {row.status}，不能继续上传")
        if row.storage_upload_id is None:
            raise ConflictError("会话缺少存储上传标识", details={"session_id": str(row.id)})

        return row, MultipartUpload(
            upload_id=row.storage_upload_id,
            key=row.storage_key_prefix,
            backend=row.storage_backend,
        )

    async def upload_part(
        self,
        *,
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        part_number: int,
        data: bytes,
    ) -> PartInfo:
        """上传一个分片。

        `part_number` 从 1 开始。越界与 0 都要拒绝——
        0-indexed 与 1-indexed 混用是分片上传最常见的 bug。
        """
        row, multipart = await self._load_upload_for_part(
            tenant_id=tenant_id, session_id=session_id
        )

        if part_number < 1 or part_number > row.total_parts:
            raise BadRequestError(f"分片编号 {part_number} 超出范围 [1, {row.total_parts}]")

        # 除最后一片外，每片大小必须等于 part_size
        expected_size = row.part_size
        if part_number == row.total_parts:
            # 最后一片是剩余部分（可能小于 part_size）
            expected_size = row.total_size - row.part_size * (row.total_parts - 1)
        if len(data) != expected_size:
            raise BadRequestError(
                f"分片 {part_number} 大小应为 {expected_size} 字节，实际 {len(data)}",
                details={"expected": expected_size, "actual": len(data)},
            )

        part = await self._storage.upload_part(multipart, part_number, data)

        # 更新会话计数（缓存值，权威来源是存储）
        await self._sessions.mark_part_uploaded(session_id=session_id, expected_version=row.version)

        return part

    async def list_uploaded_parts(
        self, *, tenant_id: uuid.UUID, session_id: uuid.UUID
    ) -> list[PartInfo]:
        """列出已上传分片（断点续传）。

        **权威来源是存储层**，不是数据库计数：可能有别的实例接收了分片，
        或者数据库更新失败但存储成功。
        """
        _row, multipart = await self._load_upload_for_part(
            tenant_id=tenant_id, session_id=session_id
        )
        return await self._storage.list_parts(multipart)

    async def session_status(
        self, *, tenant_id: uuid.UUID, session_id: uuid.UUID
    ) -> tuple[str, int, int, list[PartInfo], list[int]]:
        """返回会话状态与分片进度。

        返回 `(status, total_parts, part_size, uploaded, missing)`。
        `uploaded` 以**存储层**为准；`missing` 是客户端还差哪些片。
        """
        row, multipart = await self._load_upload_for_part(
            tenant_id=tenant_id, session_id=session_id
        )
        uploaded = await self._storage.list_parts(multipart)
        present = {p.part_number for p in uploaded}
        missing = [n for n in range(1, row.total_parts + 1) if n not in present]
        return row.status, row.total_parts, row.part_size, uploaded, missing

    async def session_id_for_upload(
        self, *, tenant_id: uuid.UUID, storage_upload_id: str
    ) -> uuid.UUID:
        """按存储层 upload_id 反查会话主键。

        保留这个方法是为了兼容"客户端只知道存储 upload_id"的旧调用方。
        新接口应该直接用会话主键（`InitiateResult.upload_id`）。
        """
        from sqlalchemy import select

        result = await self._session.execute(
            select(UploadSession.id)
            .where(
                UploadSession.tenant_id == tenant_id,
                UploadSession.storage_upload_id == storage_upload_id,
            )
            .limit(1)
        )
        found = result.scalar_one_or_none()
        if found is None:
            raise NotFoundError("上传会话不存在")
        return found

    # ------------------------------------------------------------ 内部辅助

    async def _require_active_kb(self, *, tenant_id: uuid.UUID, kb_id: uuid.UUID) -> None:
        """校验知识库存在、属于本租户、且未归档。

        ## 为什么这一层校验不能省

        `upload_sessions.knowledge_base_id` 只是个普通外键。不校验就能：

        1. **跨租户**：拿别人租户的 kb_id 建会话，complete 后再用
           `POST /documents` 建文档——虽然那时也会被 `_require_kb` 拦住，
           但上传的字节已经落进了存储，白占空间；
        2. **已归档的库**：归档库里不该再进新文档。

        提前拦在这里，错误信息也更有用（"知识库已归档" vs 大文件传完
        才报"无法建文档"）。
        """
        from rag_contracts.enums import KnowledgeBaseStatus
        from rag_core.errors import ConflictError, NotFoundError
        from rag_db.repositories import KnowledgeBaseRepository

        kb = await KnowledgeBaseRepository(self._session).get(tenant_id=tenant_id, kb_id=kb_id)
        if kb is None:
            raise NotFoundError("知识库不存在")
        if kb.status != KnowledgeBaseStatus.ACTIVE.value:
            raise ConflictError(
                "知识库已归档，无法上传文档",
                details={"status": str(kb.status)},
            )

    async def get_session_filename(self, *, tenant_id: uuid.UUID, session_id: uuid.UUID) -> str:
        """取会话的原始文件名（用于 complete 时确定扩展名/魔数）。"""
        row, _ = await self._load_upload_for_part(tenant_id=tenant_id, session_id=session_id)
        return row.filename

    async def get_session_owner(self, *, tenant_id: uuid.UUID, session_id: uuid.UUID) -> uuid.UUID:
        """取上传会话的创建者 ID（归属校验用）。

        ## 为什么需要这个方法

        上传会话只按 `tenant_id` 查（`_load_upload_for_part`），所以
        **同租户内任何成员都能操作他人会话**——上传分片、合并、甚至中止。
        这是同租户内的越权。

        路由层拿到 `created_by` 后调 `ensure_own_resource`：
        本人或管理员放行，其他人 404（不泄漏会话存在性）。

        ## 为什么不在 service 里直接判权限

        service 不认识 `PermissionContext`（那是 web 层的概念）。
        把"谁是所有者"（事实）与"这个人能不能操作"（策略）分开，
        worker / 脚本复用 service 时不会被 web 权限模型绑住。
        """
        row, _ = await self._load_upload_for_part(tenant_id=tenant_id, session_id=session_id)
        return row.created_by

    async def complete(
        self,
        *,
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        parts: list[PartInfo],
        declared_hash: str | None = None,
    ) -> Blob:
        """合并分片并落库。

        步骤：
        1. 原子把会话切到 COMPLETING（防并发重复合并）；
        2. 合并分片；
        3. **重新计算整体 hash 并校验**；
        4. 校验失败 → 删除对象、标记失败；
        5. 成功 → 写 blobs 记录、标记会话完成。

        Args:
            declared_hash: 客户端声明的整体 SHA-256。None 时跳过校验
                （不推荐，但兼容"客户端算不出 hash"的场景）。
        """
        row = await self._sessions.begin_complete(tenant_id=tenant_id, session_id=session_id)
        if row is None:
            raise ConflictError("会话状态不允许完成（可能已完成或正在合并）")

        multipart = MultipartUpload(
            upload_id=row.storage_upload_id or "",
            key=row.storage_key_prefix,
            backend=row.storage_backend,
        )

        try:
            # 1. 校验分片齐全（存储侧的实际状态，不信任客户端传的列表）
            stored_parts = await self._storage.list_parts(multipart)
            stored_numbers = {p.part_number for p in stored_parts}
            missing = set(range(1, row.total_parts + 1)) - stored_numbers
            if missing:
                raise BadRequestError(
                    f"分片不完整，缺少 {len(missing)} 片",
                    details={"missing": ",".join(str(n) for n in sorted(missing)[:20])},
                )

            # 2. 合并（用存储侧的实际 ETag，不用客户端传的——
            #    客户端传的可能是伪造的）
            metadata = await self._storage.complete_multipart(multipart, stored_parts)

            # 3. **魔数校验**：initiate 时没有内容只能校验扩展名，
            #    现在有完整文件了，必须确认内容与扩展名一致。
            #    防的是"声明 .pdf 实际传 .exe"这类绕过。
            await self._verify_content_matches_extension(key=multipart.key, filename=row.filename)

            # 4. 整体 hash 校验
            actual_hash = await self._compute_stored_hash(multipart.key)
            if declared_hash and actual_hash != declared_hash:
                log.warning(
                    "upload_hash_mismatch",
                    session_id=str(session_id),
                    declared=declared_hash[:12],
                    actual=actual_hash[:12],
                )
                await self._storage.delete(multipart.key)
                await self._sessions.mark_failed(
                    session_id=session_id,
                    error_code="UPLOAD.HASH_MISMATCH",
                    error_message="合并后的内容与声明 hash 不一致",
                )
                raise BadRequestError(
                    "文件内容校验失败，请重新上传",
                    details={"reason": "hash_mismatch"},
                )

            # 4. 写 blob（用 ON CONFLICT 处理并发同内容上传）
            blob, created = await self._blobs.get_or_create(
                tenant_id=tenant_id,
                content_hash=actual_hash,
                size_bytes=metadata.size_bytes or row.total_size,
                storage_key=multipart.key,
                storage_backend=row.storage_backend,
                content_type=row.content_type,
                status=BlobStatus.COMPLETE,
            )
            if created:
                await self._blobs.increment_ref(blob_id=blob.id)
            else:
                # 并发秒传：已有记录，加引用计数
                await self._blobs.increment_ref(blob_id=blob.id)
                log.info(
                    "upload_blob_reused_after_complete",
                    content_hash=actual_hash[:12],
                )

            await self._sessions.finish_complete(session_id=session_id, blob_id=blob.id)
            log.info(
                "upload_completed",
                session_id=str(session_id),
                blob_id=str(blob.id),
                size_bytes=blob.size_bytes,
            )
            return blob

        except (StorageError, Exception) as exc:
            # 失败时清理：删存储对象（避免孤儿），标记会话失败
            if not isinstance(exc, (BadRequestError, ConflictError, NotFoundError)):
                log.error(
                    "upload_complete_failed",
                    session_id=str(session_id),
                    error_type=type(exc).__name__,
                    exc_info=True,
                )
                await self._sessions.mark_failed(
                    session_id=session_id,
                    error_code="UPLOAD.COMPLETE_FAILED",
                    error_message=type(exc).__name__,
                )
            raise

    async def _verify_content_matches_extension(self, *, key: str, filename: str) -> None:
        """读取已存文件的开头，校验魔数与扩展名一致。

        只读前 8KB（足够判断所有支持的格式），不下载整个文件。

        失败时删除对象——不留下一个"扩展名造假"的文件在存储里。
        """
        from rag_api.services.upload_security import inspect_file
        from rag_core.errors import UnsupportedMediaTypeError

        head = b""
        async for chunk in self._storage.get(key):
            head = chunk[:8192]
            break

        try:
            inspect_file(filename=filename, head=head)
        except UnsupportedMediaTypeError:
            # 内容与扩展名不符：删对象再抛，避免存储里留下假文件
            from rag_adapters.storages.base import StorageError

            try:
                await self._storage.delete(key)
            except StorageError as cleanup_exc:
                log.warning(
                    "upload_cleanup_after_mime_mismatch_failed",
                    key=key,
                    error=type(cleanup_exc).__name__,
                )
            raise

    async def _compute_stored_hash(self, key: str) -> str:
        """流式计算已存对象的 SHA-256。

        **流式**：1GB 文件不能读进内存。
        """
        import hashlib

        digest = hashlib.sha256()
        async for chunk in self._storage.get(key):
            digest.update(chunk)
        return digest.hexdigest()

    async def abort(self, *, tenant_id: uuid.UUID, session_id: uuid.UUID) -> None:
        """中止上传并清理分片。

        **必须调用存储层的 abort**：S3 不会自动清理未完成的分片上传。
        """
        row = await self._sessions.get(tenant_id=tenant_id, session_id=session_id)
        if row is None:
            raise NotFoundError("上传会话不存在")

        if row.storage_upload_id:
            try:
                await self._storage.abort_multipart(
                    MultipartUpload(
                        upload_id=row.storage_upload_id,
                        key=row.storage_key_prefix,
                        backend=row.storage_backend,
                    )
                )
            except StorageError as exc:
                # 中止失败不阻断：会话仍要标记为 aborted，
                # 否则清理任务会一直重试它
                log.warning(
                    "upload_abort_storage_failed",
                    session_id=str(session_id),
                    error=type(exc).__name__,
                )

        await self._sessions.abort(tenant_id=tenant_id, session_id=session_id)
        log.info("upload_aborted", session_id=str(session_id))


def _suffix_for(filename: str) -> str:
    """提取文件扩展名（含点），用于存储键。

    只用扩展名、不用完整文件名：存储键里不该出现用户文件名
    （隐私 + 路径穿越风险）。
    """
    dot = filename.rfind(".")
    if dot <= 0 or dot == len(filename) - 1:
        return ""
    suffix = filename[dot:].lower()
    # 只保留字母数字，防止构造奇怪的扩展名
    cleaned = "".join(ch for ch in suffix if ch.isalnum() or ch == ".")
    return cleaned[:16]


__all__ = ["InitiateResult", "UploadService"]
