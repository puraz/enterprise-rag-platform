"""上传相关的 HTTP 契约。

## 设计要点

**分片上传的响应不返回分片内容**：客户端只需要知道 ETag 以便 complete 时带上。
返回整个分片数据是浪费（客户端刚发上来的）。

**`PartInfo` 必须是契约模型而不是内部 dataclass**：
路由的 `response_model` 需要它，而 OpenAPI 生成 TS 时也要看到它。
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class InitiateUploadRequest(BaseModel):
    """初始化上传请求。"""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(description="原始文件名（服务端会净化）", min_length=1, max_length=500)
    # **不在契约层设上限**：超过限制应返回 413（Payload Too Large），
    # 而不是 422（参数错误）。由 service 层校验，前端据此提示"文件过大"。
    total_size: int = Field(description="文件总字节数", ge=1)
    content_type: str = Field(
        description="MIME 类型（仅供参考，服务端以魔数为准）",
        min_length=1,
        max_length=200,
    )
    content_hash: str | None = Field(
        default=None,
        description=(
            "整体 SHA-256（小写十六进制）。提供时启用秒传："
            "同租户内已有相同内容的文件则直接返回 blob_id，无需上传"
        ),
        pattern=r"^[0-9a-f]{64}$",
    )
    knowledge_base_id: uuid.UUID | None = Field(
        default=None, description="目标知识库（可稍后在 complete 时指定）"
    )


class InitiateUploadResponse(BaseModel):
    """初始化上传响应。

    两种形态（用 `deduplicated` 区分）：
    - **秒传**：`deduplicated=true`、`blob_id` 有值、`upload_id` 为 null；
    - **需上传**：`deduplicated=false`、`upload_id` 有值、`blob_id` 为 null。
    """

    model_config = ConfigDict(frozen=True)

    deduplicated: bool = Field(description="是否秒传命中（无需上传）")
    upload_id: str | None = Field(default=None, description="存储层上传会话 ID")
    blob_id: uuid.UUID | None = Field(default=None, description="秒传命中的 blob ID")
    part_size: int = Field(description="建议的分片大小（字节）")
    total_parts: int = Field(description="总分片数")
    uploaded_parts: list[int] = Field(
        default_factory=list,
        description="已上传的分片号（续传场景：客户端只需补传缺失的）",
    )


class UploadPartResponse(BaseModel):
    """分片上传响应。"""

    model_config = ConfigDict(frozen=True)

    part_number: int = Field(description="分片编号（从 1 开始）")
    etag: str = Field(description="分片 ETag，complete 时需带回")
    size_bytes: int = Field(description="分片字节数")


class UploadSessionStatusResponse(BaseModel):
    """上传会话状态（断点续传查询）。"""

    model_config = ConfigDict(frozen=True)

    upload_id: str = Field(description="上传会话 ID")
    status: str = Field(description="会话状态：initiated/uploading/completing/completed/...")
    total_parts: int
    part_size: int
    uploaded_parts: list[UploadPartResponse] = Field(
        description="已上传的分片（**以对象存储为准**，不是数据库计数）"
    )
    missing_parts: list[int] = Field(description="尚未上传的分片号，客户端只需补这些")


class CompleteUploadRequest(BaseModel):
    """完成上传请求。"""

    model_config = ConfigDict(extra="forbid")

    content_hash: str | None = Field(
        default=None,
        description=(
            "整体 SHA-256。提供时服务端会重新计算已合并文件的 hash 并比对；"
            "不匹配则作废上传。**强烈建议提供**——分片级 ETag 无法证明整体正确"
        ),
        pattern=r"^[0-9a-f]{64}$",
    )
    knowledge_base_id: uuid.UUID | None = Field(
        default=None, description="目标知识库（覆盖 initiate 时的值）"
    )


class CompleteUploadResponse(BaseModel):
    """完成上传响应。"""

    model_config = ConfigDict(frozen=True)

    blob_id: uuid.UUID = Field(description="落库的物理文件 ID")
    size_bytes: int = Field(description="文件大小")
    content_hash: str = Field(description="服务端计算的内容 SHA-256")
    content_type: str = Field(description="服务端确认的 MIME 类型")
