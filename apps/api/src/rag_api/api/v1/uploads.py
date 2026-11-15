"""上传路由：分片上传的四个端点。

## 端点设计

| 方法 | 路径 | 作用 | 状态码 |
|---|---|---|---|
| POST | `/uploads` | 初始化（含秒传判断） | 200 / **201** |
| PUT | `/uploads/{id}/parts/{n}` | 上传一个分片 | 200 |
| GET | `/uploads/{id}` | 查进度（断点续传） | 200 |
| POST | `/uploads/{id}/complete` | 合并完成 | 200 / 202 |
| DELETE | `/uploads/{id}` | 中止 | **204** |

**为什么分片用 PUT 而不是 POST**：PUT 是幂等的——同一分片重传
结果相同。客户端网络抖动时可以安全重试，不需要服务端做额外幂等处理。

## 分片体直接读原始字节，不走 multipart

```python
data = await request.body()
```

不用 `UploadFile` / `File(...)` 的原因：
- multipart 编码会给每个分片加上 boundary 与头部（几十字节到几百字节），
  1000 个分片就白传几百 KB；
- multipart 解析器会缓冲整个请求，大分片（8MB）时内存翻倍；
- 分片是**二进制块**，不是"文件上传"语义，用原始体更直接。

**代价**：必须手动检查 `Content-Length`，防止恶意超大请求。
路由里用 `max_part_size` 做上限校验（在读取前就拒绝）。

## complete 的 202 vs 200

- 小文件（已合并完）：200 直接返回 blob；
- 如果将来加"合并后异步建文档"（A8），应返回 202 + task_id。

当前实现是同步合并，返回 200。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import Principal, get_current_principal, get_db, get_storage
from rag_api.openapi import (
    DELETE_ENDPOINT,
    TENANT_ENDPOINT,
    TENANT_WRITE_ENDPOINT,
)
from rag_api.services.upload import UploadService
from rag_api.services.upload_security import inspect_file
from rag_api.storages.base import PartInfo, StorageBackend
from rag_contracts.api.uploads import (
    CompleteUploadRequest,
    CompleteUploadResponse,
    InitiateUploadRequest,
    InitiateUploadResponse,
    UploadPartResponse,
    UploadSessionStatusResponse,
)
from rag_core.errors import BadRequestError, PayloadTooLargeError
from rag_core.logging import get_logger
from rag_core.settings import get_settings

log = get_logger(__name__)

router = APIRouter(prefix="/uploads", tags=["uploads"])

# 分片请求体的硬上限（防止恶意超大 body 撑爆内存）。
# 用配置的 part_size 的 1.5 倍留余量：客户端可能因编码差异略微超长。
_PART_OVERHEAD_FACTOR = 1.5


def _service(session: AsyncSession, storage: StorageBackend) -> UploadService:
    return UploadService(session, storage, get_settings().storage)


@router.post(
    "",
    response_model=InitiateUploadResponse,
    status_code=201,
    summary="初始化上传（含秒传判断）",
    description=(
        "提供 `content_hash` 时启用秒传：同租户内已有相同内容则直接返回 blob_id，"
        "无需上传。否则返回 upload_id 与分片规划。\n\n"
        "**秒传只在同租户内生效**：跨租户秒传会泄漏「某个文件是否存在」的信息。"
    ),
    responses=TENANT_WRITE_ENDPOINT,
)
async def initiate_upload(
    payload: InitiateUploadRequest,
    principal: Principal = Depends(get_current_principal),  # noqa: B008
    session: AsyncSession = Depends(get_db),  # noqa: B008
    storage: StorageBackend = Depends(get_storage),  # noqa: B008
) -> InitiateUploadResponse:
    # **文件名与扩展名在这里校验**（不是等到 complete）：
    # 让客户端尽早知道"这个格式不支持"，而不是传完 2GB 才发现
    # initiate 阶段还没有文件内容，只校验文件名与扩展名。
    # 魔数校验留到 complete：那时才有完整文件可读。
    safe_name = inspect_file(filename=payload.filename, head=None)

    service = _service(session, storage)
    result = await service.initiate(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        filename=safe_name.safe_filename,
        total_size=payload.total_size,
        content_type=payload.content_type,
        content_hash=payload.content_hash,
        knowledge_base_id=payload.knowledge_base_id,
    )

    return InitiateUploadResponse(
        deduplicated=result.deduplicated,
        upload_id=result.upload_id,  # 会话主键，不是存储层 upload_id
        blob_id=result.blob_id,
        part_size=result.part_size,
        total_parts=result.total_parts,
        uploaded_parts=result.uploaded_parts or [],
    )


@router.put(
    "/{upload_id}/parts/{part_number}",
    response_model=UploadPartResponse,
    summary="上传分片",
    description=(
        "请求体是**原始二进制**（`application/octet-stream`），不是 multipart。\n\n"
        "分片编号从 **1** 开始。除最后一片外，每片大小必须等于 `part_size`。\n\n"
        "PUT 是幂等的：同一分片重传结果相同，客户端可安全重试。"
    ),
    responses=TENANT_WRITE_ENDPOINT,
)
async def upload_part(
    upload_id: uuid.UUID,
    part_number: int,
    request: Request,
    principal: Principal = Depends(get_current_principal),  # noqa: B008
    session: AsyncSession = Depends(get_db),  # noqa: B008
    storage: StorageBackend = Depends(get_storage),  # noqa: B008
) -> UploadPartResponse:
    # 先看 Content-Length 再读体：超长请求直接拒绝，不浪费内存
    settings = get_settings()
    max_bytes = int(settings.storage.part_size * _PART_OVERHEAD_FACTOR)
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise PayloadTooLargeError(
                    f"分片超过上限 {max_bytes} 字节",
                    details={"max_bytes": max_bytes, "declared": int(declared)},
                )
        except ValueError as exc:
            raise BadRequestError("Content-Length 非法") from exc

    data = await request.body()
    if not data:
        raise BadRequestError("分片内容为空")
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"分片超过上限 {max_bytes} 字节", details={"actual": len(data)})

    # upload_id 是存储层的 upload_id（hex 字符串），不是数据库的 session id。
    # 这里通过 service 反查 session —— 需要 tenant 隔离，不能直接用 upload_id 查库。
    service = _service(session, storage)

    part: PartInfo = await service.upload_part(
        tenant_id=principal.tenant_id,
        session_id=upload_id,
        part_number=part_number,
        data=data,
    )
    return UploadPartResponse(
        part_number=part.part_number, etag=part.etag, size_bytes=part.size_bytes
    )


@router.get(
    "/{upload_id}",
    response_model=UploadSessionStatusResponse,
    summary="查询上传进度（断点续传）",
    description=(
        "返回已上传与缺失的分片。**已上传列表以对象存储为准**，"
        "不是数据库计数——可能有别的实例接收了分片，或数据库更新失败但存储成功。"
    ),
    responses=TENANT_ENDPOINT,
)
async def get_upload_status(
    upload_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),  # noqa: B008
    session: AsyncSession = Depends(get_db),  # noqa: B008
    storage: StorageBackend = Depends(get_storage),  # noqa: B008
) -> UploadSessionStatusResponse:
    service = _service(session, storage)
    status, total_parts, part_size, uploaded, missing = await service.session_status(
        tenant_id=principal.tenant_id, session_id=upload_id
    )
    return UploadSessionStatusResponse(
        upload_id=str(upload_id),
        status=status,
        total_parts=total_parts,
        part_size=part_size,
        uploaded_parts=[
            UploadPartResponse(part_number=p.part_number, etag=p.etag, size_bytes=p.size_bytes)
            for p in uploaded
        ],
        missing_parts=missing,
    )


@router.post(
    "/{upload_id}/complete",
    response_model=CompleteUploadResponse,
    summary="完成上传（合并分片）",
    description=(
        "合并所有分片并落库为 blob。\n\n"
        "提供 `content_hash` 时服务端会**重新计算**合并后文件的 SHA-256 并比对，"
        "不匹配则作废（删除对象）。分片级 ETag 只能证明单片正确，"
        "不能证明整体与客户端原始文件一致。"
    ),
    responses=TENANT_WRITE_ENDPOINT,
)
async def complete_upload(
    upload_id: uuid.UUID,
    payload: CompleteUploadRequest,
    principal: Principal = Depends(get_current_principal),  # noqa: B008
    session: AsyncSession = Depends(get_db),  # noqa: B008
    storage: StorageBackend = Depends(get_storage),  # noqa: B008
) -> CompleteUploadResponse:
    service = _service(session, storage)

    # 合并前用存储层的实际分片列表（不信任客户端传的）
    _status, _total, _size, uploaded, missing = await service.session_status(
        tenant_id=principal.tenant_id, session_id=upload_id
    )
    if missing:
        raise BadRequestError(
            f"还有 {len(missing)} 个分片未上传",
            details={"missing": ",".join(str(n) for n in missing[:20])},
        )

    blob = await service.complete(
        tenant_id=principal.tenant_id,
        session_id=upload_id,
        parts=uploaded,
        declared_hash=payload.content_hash,
    )
    return CompleteUploadResponse(
        blob_id=blob.id,
        size_bytes=blob.size_bytes,
        content_hash=blob.content_hash,
        content_type=blob.content_type,
    )


@router.delete(
    "/{upload_id}",
    status_code=204,
    summary="中止上传",
    description=(
        "中止上传并清理已上传的分片。**必须调用**："
        "S3 不会自动清理未完成的分片上传，它们会一直占用存储并计费。"
    ),
    responses=DELETE_ENDPOINT,
)
async def abort_upload(
    upload_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),  # noqa: B008
    session: AsyncSession = Depends(get_db),  # noqa: B008
    storage: StorageBackend = Depends(get_storage),  # noqa: B008
) -> None:
    service = _service(session, storage)
    await service.abort(tenant_id=principal.tenant_id, session_id=upload_id)
