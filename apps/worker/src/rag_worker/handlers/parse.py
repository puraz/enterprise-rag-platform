"""解析任务处理器：下载原始文件 → 解析 → 落解析产物 → 推进状态机。

## 管线位置

    上传完成（API 写 documents/versions/blobs + 任务 + outbox，同事务）
      │
      └─▶ [本处理器]  download → parse → verify offsets → 存 parsed 文本
            │                                        │
            │  documents.status: PENDING → PARSING → CHUNKING
            ▼
          切分任务（A9）从这里接手

## 四个容易做错的地方

### 1. 关系必须 eager load

`version.blob` / `version.document` 是 ORM 关系。在 async 会话里
**隐式懒加载会抛 `MissingGreenlet`**——不是"慢一点"，是直接崩。

所以用一条 `select(...).options(selectinload(...))` 把需要的关系一次取全，
而不是 `session.get()` 之后再点属性。

### 2. 状态推进必须带乐观锁

`documents.status` 有并发读者（管理端看板、重试接口、删除流程），
用 `update_status(..., expected_version=...)` 而不是裸 UPDATE：
冲突时返回 False，由调用方重新决策，而不是两个写者互相覆盖。

### 3. 解析产物不进数据库

解析出的全文可能有几十 MB。写进 `parse_metadata`(JSONB) 会让该行
TOAST 膨胀、备份暴涨、切分时整块反序列化进内存。

所以全文写到对象存储的派生键 `parsed/{tenant}/{version}.txt`，
数据库只存键 + hash + 统计。切分任务按 `parse_metadata.parsed_key` 去读。

**派生键是确定性的**（不含时间戳/随机数）：任务被回收重派时重写同一键，
不会每重试一次留一个孤儿对象。

### 4. 偏移校验失败不能静默

`verify_char_offsets()` 对不上说明解析器有 bug。此时必须让任务失败：
偏移错的后果是引用高亮显示错误的文字——答案对、高亮错，
端到端测试几乎发现不了，只能靠入库前拦截。

用 `ParseError(retryable=False)` → 分类为 PERMANENT → 直接死信。
解析器的 bug 不会因为重试 3 次而修好。

## 临时文件

下载到临时文件而不是内存：几百 MB 的 PDF 整块入内存，并发 4 个就 OOM。
`finally` 里必删——不删会让 `/tmp` 涨满，最后**所有**任务一起失败。
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from rag_adapters.parsers import ParsedDocument, ParseError, parse_document, verify_char_offsets
from rag_adapters.parsers.artifact import (
    PARSED_JSONL_SUFFIX,
    PARSED_TEXT_SUFFIX,
    BlockRecord,
    PageRecord,
    dump_parsed,
)
from rag_contracts.enums import DocumentFormat, DocumentStatus
from rag_core.errors import NotFoundError
from rag_core.logging import get_logger
from rag_db.models.document import Document, DocumentVersion
from rag_db.repositories import DocumentRepository
from rag_worker.engine import require_session

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from rag_adapters.storages import StorageBackend

log = get_logger(__name__)

# 派生存储键前缀。放常量而不是散落在代码里：清理任务要按前缀扫孤儿产物。
PARSED_KEY_PREFIX = "parsed/"


class ParseDocumentHandler:
    """`TaskType.PARSE_DOCUMENT` 的处理器。

    payload 约定（API 入队时写入）：

        {"document_id": "...", "document_version_id": "..."}

    **两者都要**：只有 document_id 无法定位是哪个版本的 blob
    （重解析旧版本是合法需求，A10 的索引重建会用到）。
    """

    # 单文档处理：引擎会包在一个事务里（失败整体回滚）。
    # 长任务（索引重建）才需要 owns_transaction = True。
    owns_transaction = False

    def __init__(self, *, storages: dict[str, StorageBackend]) -> None:
        """
        Args:
            storages: **全部**已配置后端（名字 → 实例），不是单个实例。

                `blobs.storage_backend` 记录每个对象存在哪个后端。
                迁移期（local → s3）会同时存在两个后端：老 blob 在 local、
                新 blob 在 s3。只用"当前主后端"去读老数据会报"对象不存在"，
                而且错误信息看起来像数据丢了。

                所以按 blob 记录的 backend **选**后端（见 `_storage_for`），
                而不是假设"配置里那个就是唯一的"。
        """
        if not storages:
            raise ValueError("至少需要一个存储后端")
        self._storages = storages
        # 派生产物的写入目标。约定注册表里第一个可用后端是主后端
        # （`build_all_storages` 保证主后端排在最前，见其实现）。
        self._primary_storage = next(iter(storages.values()))

    async def __call__(
        self,
        *,
        session: AsyncSession | None,
        task: Any,
        progress: Any,
        session_factory: Any = None,
    ) -> None:
        # 本 handler 不自管事务（owns_transaction = False），引擎保证传 session。
        # `session_factory` 只为满足 TaskHandler 协议签名，这里不使用。
        session = require_session(session)
        payload = dict(task.payload or {})
        document_id = _require_uuid(payload, "document_id")
        version_id = _require_uuid(payload, "document_version_id")

        await progress(5, "loading_metadata")
        version = await self._load_version(session, version_id=version_id)
        if version is None:
            # 版本被删了（用户删文档，但任务已入队）。不可重试：
            # 重试也查不到，只会白等 3 次退避。
            raise NotFoundError("文档版本不存在（可能已被删除）")
        if version.document_id != document_id:
            # payload 自相矛盾：入队时写错了，不是数据延迟。
            # 不能"按 version 猜一个 document"——猜错会改错文档的状态。
            raise ParseError(
                f"payload 不一致: version {version_id} 不属于 document {document_id}",
                retryable=False,
            )

        document = version.document
        blob = version.blob
        file_format = _parse_format(document.format, version_id=version_id)

        await progress(15, "downloading")
        # 按 blob 记录的 backend 选存储（迁移期两个后端并存）
        storage = self._storage_for(blob.storage_backend, version_id=version_id)
        tmp_path = await self._download(storage, blob.storage_key)

        try:
            await progress(30, "parsing")
            parsed = await self._parse(tmp_path, file_format)

            await progress(70, "verifying")
            self._verify_offsets(parsed, version_id=version_id)

            await progress(80, "persisting")
            await self._persist(
                session=session,
                version=version,
                document=document,
                parsed=parsed,
                document_id=document_id,
            )

            await progress(95, "finalizing")
        finally:
            # 无论成功失败都删：磁盘分会把无关任务一起拖死。
            # 放线程池：大文件 unlink 在慢盘上可能阻塞。
            await asyncio.to_thread(tmp_path.unlink, missing_ok=True)

    # ------------------------------------------------------------ 内部步骤

    async def _load_version(
        self, session: AsyncSession, *, version_id: uuid.UUID
    ) -> DocumentVersion | None:
        """取版本，并 eager load 它引用的 blob 与 document。

        **不能写 `session.get()` 然后点 `version.blob`**：async 会话里
        隐式懒加载会抛 `MissingGreenlet`（关系加载需要同步 IO，
        而当前上下文没有 greenlet 可供切换）。
        """
        result = await session.execute(
            select(DocumentVersion)
            .options(
                selectinload(DocumentVersion.blob),
                selectinload(DocumentVersion.document),
            )
            .where(DocumentVersion.id == version_id)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _download(self, storage: StorageBackend, storage_key: str) -> Path:
        """流式下载到临时文件。

        `StorageBackend.get` 是**普通 def 返回异步迭代器**（不是协程），
        所以是 `async for ... in storage.get(key)`，不加 `await`。
        """
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - 需要显式控制生命周期
            delete=False, prefix="rag-parse-", suffix=".bin"
        )
        path = Path(handle.name)
        try:
            async for chunk in storage.get(storage_key):
                # 写盘放线程池：大文件可能阻塞几十毫秒，不该占用事件循环
                await asyncio.to_thread(handle.write, chunk)
        except Exception:
            await asyncio.to_thread(handle.close)
            await asyncio.to_thread(path.unlink, missing_ok=True)
            raise
        else:
            await asyncio.to_thread(handle.close)
        return path

    def _storage_for(self, backend_name: str, *, version_id: uuid.UUID) -> StorageBackend:
        """按 blob 记录的后端名取存储实例。

        取不到说明配置被改过（比如原来用 s3，现在只配了 local）。
        这是**配置错误，不可重试**：重试还是取不到。必须人工补配置，
        然后把死信任务手动重放。
        """
        backend = self._storages.get(backend_name)
        if backend is None:
            raise ParseError(
                f"存储后端 {backend_name!r} 未初始化（version {version_id}）；"
                f"可用：{sorted(self._storages)}",
                retryable=False,
            )
        return backend

    async def _parse(self, path: Path, file_format: DocumentFormat) -> ParsedDocument:
        """在线程池里解析。

        解析器是同步阻塞的（PyMuPDF 的 C 扩展）。直接调用会阻塞
        **worker 进程的事件循环**——进度回写与并发任务全部停摆。

        为什么用线程池而不是进程池：worker 本身就是独立进程，
        再开进程池会让 storage client 等资源重复加载；PyMuPDF 的文本
        提取大部分在 C 层（会释放 GIL），线程池够用。
        """
        return await asyncio.to_thread(parse_document, path, format=file_format)

    def _verify_offsets(self, parsed: ParsedDocument, *, version_id: uuid.UUID) -> None:
        """校验解析结果自洽。失败即永久失败（解析器 bug，重试无用）。"""
        problems = verify_char_offsets(parsed)
        if not problems:
            return

        log.error(
            "parse_offset_invalid",
            version_id=str(version_id),
            problem_count=len(problems),
            # 只记第一条前 120 字符：问题可能上百条，日志不该被刷屏；
            # 该 message 只含偏移数字与文本片段，不含全文。
            first_problem=problems[0][:120],
        )
        raise ParseError(
            f"解析结果偏移不一致（{len(problems)} 处），解析器实现有 bug",
            retryable=False,
        )

    async def _persist(
        self,
        *,
        session: AsyncSession,
        version: DocumentVersion,
        document: Document,
        parsed: ParsedDocument,
        document_id: uuid.UUID,
    ) -> None:
        """落解析产物 + 回写统计 + 推进状态。"""
        base_key = f"{PARSED_KEY_PREFIX}{version.tenant_id}/{version.id}"
        text_key = base_key + PARSED_TEXT_SUFFIX
        artifact_key = base_key + PARSED_JSONL_SUFFIX

        encoded_text = parsed.text.encode("utf-8")
        content_hash = hashlib.sha256(encoded_text).hexdigest()

        # 派生产物写回**主**后端（blob 所在的可能是个正在退役的旧后端）。
        # 确定性键 + 内容寻址语义：重派任务重写同一对象，内容不变。
        #
        # 写两份：
        # 1. .txt —— 纯文本，供全文检索/调试/人工比对，人眼可读；
        # 2. .jsonl —— 结构化（块 + 页 + 偏移 + bbox + 标题层级），
        #    切分器按它填 chunk 的位置字段。**没有它就做不出引用高亮**。
        await self._primary_storage.put(
            text_key,
            encoded_text,
            content_type="text/plain; charset=utf-8",
        )

        artifact = dump_parsed(
            text=parsed.text,
            blocks=[
                BlockRecord(
                    text=block.text,
                    block_type=block.block_type,
                    char_start=block.char_start,
                    char_end=block.char_end,
                    page_no=block.page_no,
                    bbox=block.bbox,
                    level=block.level,
                    metadata=dict(block.metadata),
                )
                for block in parsed.blocks
            ],
            pages=[
                PageRecord(
                    page_no=page.page_no,
                    width=page.width,
                    height=page.height,
                    char_start=page.char_start,
                    char_end=page.char_end,
                    is_scanned=page.is_scanned,
                )
                for page in parsed.pages
            ],
            parser_name=parsed.parser_name,
            document_format=parsed.format.value,
            has_scanned_pages=parsed.has_scanned_pages,
            metadata=dict(parsed.metadata),
        )
        await self._primary_storage.put(
            artifact_key,
            artifact,
            content_type="application/x-ndjson",
        )

        # ---- 版本级：解析统计 + 产物指针 ----
        # 直接改 ORM 对象（事务内）：少写一个字段不会静默漏掉，
        # mypy 也能检查字段名（裸 UPDATE 的字符串键名查不出来）。
        version.parser_name = parsed.parser_name
        version.page_count = parsed.page_count or None
        version.parse_metadata = {
            **(version.parse_metadata or {}),
            # parsed_key 保留指向 .txt（已有的调试/比对工具按它找）
            "parsed_key": text_key,
            # artifact_key 是切分器的输入（结构化，含位置信息）
            "artifact_key": artifact_key,
            "parsed_hash": content_hash,
            "block_count": len(parsed.blocks),
            "page_count": len(parsed.pages),
            "char_count": len(parsed.text),
            "has_scanned_pages": parsed.has_scanned_pages,
            # 文档级元数据（标题/作者等）：解析器已按白名单过滤，不含正文
            **{f"doc_{key}": value for key, value in parsed.metadata.items()},
        }

        # ---- 文档级：状态推进 + 统计 ----
        updated = await DocumentRepository(session).update_status(
            tenant_id=version.tenant_id,
            doc_id=document_id,
            new_status=DocumentStatus.CHUNKING,
            expected_version=document.version,
            page_count=parsed.page_count or None,
            # 清掉上一次失败留下的错误信息：否则修好后仍显示"失败原因"
            error_code=None,
            error_message=None,
        )
        if not updated:
            # 乐观锁冲突：文档被并发修改（删除 / 重传新版本 / 另一 worker 抢先）。
            # 用可重试类型：另一方完成后再来一次通常就成功了。
            raise ParseError("文档状态被并发修改，解析结果未回写（将重试）", retryable=True)

        # ---- 入队切分任务（同一事务）----
        #
        # **必须与状态推进同事务**：分两处写会出现"状态已 CHUNKING 但任务
        # 没入队"的窗口——文档永远卡在 CHUNKING，且没有任务会推进它。
        # 这类卡死没有任何报错，只能靠人发现"这份文档一直是处理中"。
        #
        # 管线在这里串起来：parse → chunk → embed → ready。
        # 每一环都由上一环的 handler 在同一事务里入队，不依赖外部编排。
        from rag_contracts.enums import TaskType
        from rag_db.repositories import TaskRepository

        await TaskRepository(session).enqueue(
            tenant_id=version.tenant_id,
            task_type=TaskType.CHUNK_DOCUMENT.value,
            # 幂等键含 version：重解析同一版本不会产生第二个切分任务；
            # 但产生新版本时会有独立的任务（这是期望的）
            idempotency_key=f"chunk:{document_id}:{version.id}",
            payload={
                "document_id": str(document_id),
                "document_version_id": str(version.id),
            },
            document_id=document_id,
            document_version_id=version.id,
            knowledge_base_id=document.knowledge_base_id,
        )

        log.info(
            "document_parsed",
            document_id=str(document_id),
            version_id=str(version.id),
            parser=parsed.parser_name,
            page_count=parsed.page_count,
            block_count=len(parsed.blocks),
            char_count=len(parsed.text),
            has_scanned_pages=parsed.has_scanned_pages,
            parsed_key=text_key,
            artifact_key=artifact_key,
        )


# ---------------------------------------------------------------- 辅助函数


def _require_uuid(payload: dict[str, Any], key: str) -> uuid.UUID:
    """从 payload 取 UUID，缺失或非法即永久失败。

    用 `ParseError(retryable=False)` 而不是 ValueError：ValueError 会被
    分类成 FATAL（也不重试），但错误码会变成 `UNHANDLED_ValueError`；
    ParseError 能让死信里显示"任务参数错误"，一眼看出是入队方的问题。
    """
    raw = payload.get(key)
    if not raw:
        raise ParseError(f"任务 payload 缺少 {key}", retryable=False)
    try:
        return uuid.UUID(str(raw))
    except ValueError as exc:
        raise ParseError(f"任务 payload 的 {key} 不是合法 UUID", retryable=False) from exc


def _parse_format(raw: Any, *, version_id: uuid.UUID) -> DocumentFormat:
    """把库里存的格式字符串转成枚举。

    `documents.format` 存的是字符串（DB 不该依赖 Python 枚举的成员名）。
    值不认识说明：要么迁移没跑全，要么有人手改了库。
    这不是可重试错误——重试读到的还是同一个非法值。
    """
    try:
        return DocumentFormat(str(raw))
    except ValueError as exc:
        raise ParseError(
            f"未知的文档格式 {raw!r}（version {version_id}）", retryable=False
        ) from exc


__all__ = ["PARSED_KEY_PREFIX", "ParseDocumentHandler"]
