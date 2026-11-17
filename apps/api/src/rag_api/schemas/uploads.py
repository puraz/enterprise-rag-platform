"""上传请求/响应实体与映射。

契约形状来自 `rag_contracts.api.uploads`；本模块只做 re-export 与
domain → 响应实体的映射。
"""

from __future__ import annotations

import uuid

from rag_adapters.storages.base import PartInfo
from rag_api.services.upload import InitiateResult
from rag_contracts.api.uploads import (
    CompleteUploadRequest,
    CompleteUploadResponse,
    InitiateUploadRequest,
    InitiateUploadResponse,
    UploadPartResponse,
    UploadSessionStatusResponse,
)
from rag_db.models.blob import Blob


def to_initiate_response(result: InitiateResult) -> InitiateUploadResponse:
    """初始化结果 → 响应实体。

    `upload_id` 是对外的会话主键（数据库 UUID），不是存储层 upload_id。
    """
    return InitiateUploadResponse(
        deduplicated=result.deduplicated,
        upload_id=result.upload_id,  # 会话主键，不是存储层 upload_id
        blob_id=result.blob_id,
        part_size=result.part_size,
        total_parts=result.total_parts,
        uploaded_parts=result.uploaded_parts or [],
    )


def to_part_response(part: PartInfo) -> UploadPartResponse:
    """分片信息 → 响应实体。"""
    return UploadPartResponse(
        part_number=part.part_number,
        etag=part.etag,
        size_bytes=part.size_bytes,
    )


def to_status_response(
    *,
    upload_id: uuid.UUID,
    status: str,
    total_parts: int,
    part_size: int,
    uploaded: list[PartInfo],
    missing: list[int],
) -> UploadSessionStatusResponse:
    """会话状态 → 响应实体。"""
    return UploadSessionStatusResponse(
        upload_id=str(upload_id),
        status=status,
        total_parts=total_parts,
        part_size=part_size,
        uploaded_parts=[to_part_response(p) for p in uploaded],
        missing_parts=missing,
    )


def to_complete_response(blob: Blob) -> CompleteUploadResponse:
    """blob → 合并完成响应实体。"""
    return CompleteUploadResponse(
        blob_id=blob.id,
        size_bytes=blob.size_bytes,
        content_hash=blob.content_hash,
        content_type=blob.content_type,
    )


__all__ = [
    "CompleteUploadRequest",
    "CompleteUploadResponse",
    "InitiateUploadRequest",
    "InitiateUploadResponse",
    "UploadPartResponse",
    "UploadSessionStatusResponse",
    "to_complete_response",
    "to_initiate_response",
    "to_part_response",
    "to_status_response",
]
