"""对象存储配置。

## 为什么后端选择要配置化

`docs/PROMPT.md` 要求"换厂商只改配置不改代码"：本地开发用磁盘、
生产用 S3（或兼容的 MinIO/OSS/COS），切换只是改 `backend` 与环境变量。

## 分片大小的双重约束

`part_size` 同时受两端限制：
- **S3 要求**：除最后一片外每片 ≥ 5MB（`MIN_PART_SIZE`）；
- **用户体验**：片太大时单片重传成本高（网络抖动要重传 100MB）。

默认 8MB：满足 S3 下限，且单片重传代价可接受。
**不能设小于 5MB**：S3 会在 complete 时拒绝（
`EntityTooSmall`），而且错误发生在**所有分片传完之后**——最浪费的失败点。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

# S3 的硬性要求：除最后一片，每片至少 5MB
S3_MIN_PART_SIZE = 5 * 1024 * 1024


class StorageSettings(BaseModel):
    """对象存储配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    backend: Literal["local", "s3"] = "local"
    """存储后端。本机无 MinIO，默认用本地磁盘。"""

    # ---- 本地后端 ----
    local_root: str = ".data/storage"
    """本地存储根目录（相对仓库根或绝对路径）。"""

    read_chunk_size: int = Field(default=1024 * 1024, ge=4096)
    """流式读取的块大小。1MB 是吞吐与内存的平衡点。"""

    # ---- S3 / 兼容后端 ----
    s3_bucket: str = "rag-documents"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    """自定义端点。None 用 AWS 官方；MinIO/OSS/COS 填各自地址。

    这是"一套代码兼容多家"的关键：S3 协议兼容的厂商只需改这个值。
    """

    s3_access_key: str | None = None
    s3_secret_key: SecretStr | None = None
    s3_use_ssl: bool = True

    s3_addressing_style: Literal["auto", "path", "virtual"] = "auto"
    """URL 寻址方式。

    - `virtual`：`https://bucket.s3.amazonaws.com/key`（AWS 默认）
    - `path`：`https://s3.amazonaws.com/bucket/key`（MinIO 必须用这个）

    分不清时用 `auto`，boto3 会按 endpoint 类型判断。
    """

    # ---- 分片上传 ----
    part_size: int = Field(default=8 * 1024 * 1024, ge=1024)
    """分片大小。

    **S3 要求 ≥ 5MB**（最后一片除外）。小于这个值会在 complete 时
    报 `EntityTooSmall` —— 而且是在所有分片传完之后才失败。
    """

    multipart_threshold: int = Field(default=100 * 1024 * 1024, ge=1024)
    """超过这个大小才走分片上传。小文件直接 put 更快。"""

    max_file_size: int = Field(default=2 * 1024**3, ge=1024)
    """单文件大小上限（业务限制，在 service 层校验）。"""

    upload_session_ttl_s: int = Field(default=24 * 3600, ge=60)
    """上传会话有效期。过期后分片由清理任务回收。"""

    @model_validator(mode="after")
    def _check_part_size(self) -> StorageSettings:
        """分片大小必须 ≥ S3 下限。

        本地后端不受此限制，但统一约束避免"本地能传、上了 S3 就失败"。
        """
        if self.backend == "s3" and self.part_size < S3_MIN_PART_SIZE:
            raise ValueError(
                f"S3 要求分片至少 {S3_MIN_PART_SIZE // 1024 // 1024}MB，"
                f"当前配置 {self.part_size // 1024 // 1024}MB"
            )
        if self.part_size > self.max_file_size:
            raise ValueError("分片大小不能超过文件大小上限")
        return self

    def total_parts_for(self, size_bytes: int) -> int:
        """按文件大小计算分片数（向上取整）。"""
        import math

        return max(1, math.ceil(size_bytes / self.part_size))
