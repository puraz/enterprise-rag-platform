"""S3 兼容存储后端（boto3）。

## 一套代码兼容多家

S3 协议被 AWS S3 / MinIO / 阿里 OSS / 腾讯 COS / 七牛 / Backblaze 广泛兼容。
换厂商只需改 `endpoint_url` + 凭证，**代码零改动**。

两个必须注意的兼容差异：

| 差异 | AWS | MinIO | 处理 |
|---|---|---|---|
| URL 寻址 | virtual-hosted | **path-style** | `s3_addressing_style` 配置 |
| SSL | 必须 | 通常关闭 | `s3_use_ssl` 配置 |

## boto3 是同步库 —— 必须切线程

boto3 没有官方异步版本。直接用会**阻塞事件循环**（`AGENTS.md` 禁止）。

每次调用都 `asyncio.to_thread(...)`。这带来一个额外好处：
boto3 的 `StreamingBody` 可以在**线程里**分块读取，
主循环只负责转发——大文件下载不会撑爆内存。

## 不使用 aioboto3 的理由

`aioboto3` 是第三方封装，主要问题：
- 连接池与 boto3 的 session 生命周期不一致，长时间运行会泄漏连接；
- 版本滞后于 boto3，某些 API（如 `upload_part_copy`）支持不全；
- 多一层抽象意味着出错时栈更深。

`to_thread` 的线程开销对 S3 调用可以忽略（网络往返是毫秒级，
线程切换是微秒级）。

## 流式上传的取舍

`put_object` 接受 file-like 对象。异步迭代器无法直接传给它，
所以流式上传的实现是：**先落临时文件再上传**。
代价是双倍磁盘 I/O，但：
- 大文件上传本来就是慢速网络 I/O，磁盘不是瓶颈；
- 换来的是内存恒定（不会因为 2GB 文件 OOM）。

如果这个开销不可接受，生产上应该让客户端直接走分片上传
（分片天然是流式的，每片独立 `upload_part`）。
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from collections.abc import AsyncIterator, Iterator
from typing import Any, BinaryIO

from rag_adapters.storages.base import (
    MultipartUpload,
    ObjectMetadata,
    ObjectNotFoundError,
    PartInfo,
    StorageBackend,
    StorageCapabilities,
    StorageError,
    sanitize_key,
)
from rag_core.logging import get_logger
from rag_core.settings.storage import StorageSettings

log = get_logger(__name__)


class S3StorageBackend(StorageBackend):
    """S3 及所有 S3 协议兼容的存储。"""

    name = "s3"

    def __init__(self, settings: StorageSettings, *, client: Any = None) -> None:
        self._settings = settings
        self._bucket = settings.s3_bucket
        self._client = client  # 允许注入 fake client（测试用）
        self._owns_client = client is None

    def _ensure_client(self) -> Any:
        """延迟创建 boto3 client。

        延迟的原因：创建时 boto3 会读取凭证链（环境变量、
        ~/.aws/credentials、IMDS 元数据服务）。如果凭证配置有误，
        在 import 期就抛异常会让模块无法加载，调试困难。
        """
        if self._client is None:
            import boto3
            from botocore.config import Config

            settings = self._settings
            self._client = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint_url,
                region_name=settings.s3_region,
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=(
                    settings.s3_secret_key.get_secret_value()
                    if settings.s3_secret_key is not None
                    else None
                ),
                use_ssl=settings.s3_use_ssl,
                config=Config(
                    s3={"addressing_style": settings.s3_addressing_style},
                    # 重试交给上层（ResilientCaller），关掉 boto3 自己的
                    # 否则会出现 "boto3 重试 × 业务重试" 的乘数效应
                    retries={"max_attempts": 1, "mode": "standard"},
                    connect_timeout=5,
                    read_timeout=60,
                ),
            )
        return self._client

    @property
    def capabilities(self) -> StorageCapabilities:
        return StorageCapabilities(
            supports_multipart=True,
            supports_presigned_url=True,  # S3 的签名 URL
            supports_server_side_encryption=True,
            max_object_size_bytes=5 * 1024**4,  # S3 单对象 5TB
            is_local=False,
        )

    def _translate_error(self, exc: BaseException, key: str) -> StorageError:
        """把 boto3 异常翻译成带可重试标记的异常。"""
        code = ""
        if hasattr(exc, "response"):
            error: dict[str, Any] = getattr(exc, "response", {}).get("Error", {})
            code = str(error.get("Code", ""))

        if code in ("NoSuchKey", "404", "NotFound"):
            return ObjectNotFoundError(key)
        if code in ("AccessDenied", "403", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
            # 凭证问题：重试无用，但要能让熔断器感知
            return StorageError(f"S3 认证失败（{code}）", retryable=False)
        if code in ("SlowDown", "503", "500", "InternalError"):
            return StorageError(f"S3 暂时不可用（{code}）", retryable=True)
        return StorageError(f"S3 操作失败（{code or type(exc).__name__}）", retryable=True)

    # ---- 基本操作 ----

    async def put(
        self,
        key: str,
        data: AsyncIterator[bytes] | bytes,
        *,
        content_type: str | None = None,
    ) -> ObjectMetadata:
        """写入对象。

        bytes 直接上传；异步迭代器先落临时文件（见模块 docstring 的取舍）。
        """
        sanitize_key(key)
        if isinstance(data, bytes):
            return await asyncio.to_thread(self._put_bytes_sync, key, data, content_type)

        # 流式：落临时文件后上传
        tmp_path = await self._spill_to_tempfile(data)
        try:
            return await asyncio.to_thread(self._put_file_sync, key, tmp_path, content_type)
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _spill_to_tempfile(self, chunks: AsyncIterator[bytes]) -> Any:
        """把异步流落到临时文件，返回路径。

        用磁盘换内存：2GB 文件上传时内存占用恒定在 1MB 左右。

        **这里必须手动管理临时文件**（不能用 `with` 上下文）：
        需要在函数返回后继续使用该路径来上传。所以：
        - 成功：交出路径，由调用方负责删除；
        - 失败：立即删除，不泄漏。
        """
        from pathlib import Path

        # delete=False 才有路径可用（with 语义会自动删）。SIM115 的告警
        # 在这里是误报：下面是显式的 try/except 管理。
        tmp = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
        path = Path(tmp.name)
        try:
            async for chunk in chunks:
                if chunk:
                    await asyncio.to_thread(tmp.write, chunk)
            await asyncio.to_thread(tmp.flush)
            await asyncio.to_thread(tmp.close)
        except Exception:
            await asyncio.to_thread(tmp.close)
            # 删除也必须切线程：pathlib 的 unlink 是同步 I/O
            await asyncio.to_thread(path.unlink, True)
            raise
        return path

    def _put_bytes_sync(self, key: str, data: bytes, content_type: str | None) -> ObjectMetadata:
        client = self._ensure_client()
        extra = {"ContentType": content_type} if content_type else {}
        try:
            response = client.put_object(Bucket=self._bucket, Key=key, Body=data, **extra)
        except Exception as exc:
            raise self._translate_error(exc, key) from exc

        return ObjectMetadata(
            key=key,
            size_bytes=len(data),
            content_type=content_type,
            etag=str(response.get("ETag", "")).strip('"') or None,
        )

    def _put_file_sync(self, key: str, path: Any, content_type: str | None) -> ObjectMetadata:
        client = self._ensure_client()
        extra = {"ContentType": content_type} if content_type else {}
        try:
            with path.open("rb") as fh:
                response = client.put_object(Bucket=self._bucket, Key=key, Body=fh, **extra)
        except Exception as exc:
            raise self._translate_error(exc, key) from exc

        return ObjectMetadata(
            key=key,
            size_bytes=path.stat().st_size,
            content_type=content_type,
            etag=str(response.get("ETag", "")).strip('"') or None,
        )

    def get(self, key: str) -> AsyncIterator[bytes]:
        """流式读取（返回异步迭代器，本身不是协程）。

        boto3 的 `StreamingBody` 支持在线程里分块读，主循环只转发。
        """
        sanitize_key(key)
        return self._iter_stream(key)

    async def _iter_stream(self, key: str) -> AsyncIterator[bytes]:
        """真正的分块读取。"""
        body = await asyncio.to_thread(self._open_stream, key)
        try:
            while True:
                chunk = await asyncio.to_thread(body.read, 1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(body.close)

    def _open_stream(self, key: str) -> Any:
        client = self._ensure_client()
        try:
            response = client.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            raise self._translate_error(exc, key) from exc
        return response["Body"]

    async def delete(self, key: str) -> None:
        """删除对象。不存在时不报错（S3 的 delete 本身幂等）。"""
        sanitize_key(key)
        await asyncio.to_thread(self._delete_sync, key)

    def _delete_sync(self, key: str) -> None:
        client = self._ensure_client()
        try:
            client.delete_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            raise self._translate_error(exc, key) from exc

    async def exists(self, key: str) -> bool:
        try:
            await self.head(key)
            return True
        except ObjectNotFoundError:
            return False

    async def head(self, key: str) -> ObjectMetadata:
        sanitize_key(key)
        return await asyncio.to_thread(self._head_sync, key)

    def _head_sync(self, key: str) -> ObjectMetadata:
        client = self._ensure_client()
        try:
            response = client.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            raise self._translate_error(exc, key) from exc

        return ObjectMetadata(
            key=key,
            size_bytes=int(response.get("ContentLength", 0)),
            content_type=str(response.get("ContentType") or "") or None,
            etag=str(response.get("ETag", "")).strip('"') or None,
            last_modified=(
                response["LastModified"].isoformat() if response.get("LastModified") else None
            ),
        )

    # ---- 分片上传 ----

    async def initiate_multipart(
        self, key: str, *, content_type: str | None = None
    ) -> MultipartUpload:
        sanitize_key(key)
        return await asyncio.to_thread(self._initiate_sync, key, content_type)

    def _initiate_sync(self, key: str, content_type: str | None) -> MultipartUpload:
        client = self._ensure_client()
        extra = {"ContentType": content_type} if content_type else {}
        try:
            response = client.create_multipart_upload(Bucket=self._bucket, Key=key, **extra)
        except Exception as exc:
            raise self._translate_error(exc, key) from exc

        return MultipartUpload(
            upload_id=str(response["UploadId"]),
            key=key,
            backend=self.name,
            bucket_hint=self._bucket,
        )

    async def upload_part(
        self,
        upload: MultipartUpload,
        part_number: int,
        data: bytes,
    ) -> PartInfo:
        """上传分片。part_number 从 1 开始（S3 规定）。"""
        if part_number < 1:
            raise StorageError(f"分片编号必须 ≥ 1（S3 语义），收到 {part_number}", retryable=False)
        return await asyncio.to_thread(self._upload_part_sync, upload, part_number, data)

    def _upload_part_sync(self, upload: MultipartUpload, part_number: int, data: bytes) -> PartInfo:
        client = self._ensure_client()
        try:
            response = client.upload_part(
                Bucket=self._bucket,
                Key=upload.key,
                UploadId=upload.upload_id,
                PartNumber=part_number,
                Body=data,
            )
        except Exception as exc:
            raise self._translate_error(exc, upload.key) from exc

        return PartInfo(
            part_number=part_number,
            etag=str(response.get("ETag", "")).strip('"'),
            size_bytes=len(data),
        )

    async def list_parts(self, upload: MultipartUpload) -> list[PartInfo]:
        """列出已上传分片。

        **必须处理分页**：S3 的 `list_parts` 每次最多返回 1000 个，
        大文件（>1000 片）不翻页会漏。
        """
        return await asyncio.to_thread(self._list_parts_sync, upload)

    def _list_parts_sync(self, upload: MultipartUpload) -> list[PartInfo]:
        client = self._ensure_client()
        parts: list[PartInfo] = []
        marker = 0
        while True:
            try:
                response = client.list_parts(
                    Bucket=self._bucket,
                    Key=upload.key,
                    UploadId=upload.upload_id,
                    PartNumberMarker=marker,
                )
            except Exception as exc:
                raise self._translate_error(exc, upload.key) from exc

            for item in response.get("Parts") or []:
                parts.append(
                    PartInfo(
                        part_number=int(item["PartNumber"]),
                        etag=str(item.get("ETag", "")).strip('"'),
                        size_bytes=int(item.get("Size", 0)),
                    )
                )
            if not response.get("IsTruncated"):
                break
            marker = int(response.get("NextPartNumberMarker", 0))

        return sorted(parts, key=lambda p: p.part_number)

    async def complete_multipart(
        self, upload: MultipartUpload, parts: list[PartInfo]
    ) -> ObjectMetadata:
        """合并分片。`parts` 必须按 part_number 升序。"""
        if not parts:
            raise StorageError("没有可合并的分片", retryable=False)
        ordered = sorted(parts, key=lambda p: p.part_number)
        return await asyncio.to_thread(self._complete_sync, upload, ordered)

    def _complete_sync(self, upload: MultipartUpload, parts: list[PartInfo]) -> ObjectMetadata:
        client = self._ensure_client()
        try:
            response = client.complete_multipart_upload(
                Bucket=self._bucket,
                Key=upload.key,
                UploadId=upload.upload_id,
                MultipartUpload={
                    "Parts": [{"PartNumber": p.part_number, "ETag": p.etag} for p in parts]
                },
            )
        except Exception as exc:
            raise self._translate_error(exc, upload.key) from exc

        return ObjectMetadata(
            key=upload.key,
            size_bytes=0,  # S3 的 complete 不返回总大小，调用方用 head 取
            etag=str(response.get("ETag", "")).strip('"') or None,
        )

    async def abort_multipart(self, upload: MultipartUpload) -> None:
        """中止并清理分片。

        **必须调用**：S3 不会自动清理未完成的分片上传，
        它们会一直占用存储并产生费用。
        """
        await asyncio.to_thread(self._abort_sync, upload)

    def _abort_sync(self, upload: MultipartUpload) -> None:
        client = self._ensure_client()
        try:
            client.abort_multipart_upload(
                Bucket=self._bucket,
                Key=upload.key,
                UploadId=upload.upload_id,
            )
        except Exception as exc:
            # 中止失败不抛异常：会话已经不完整，抛错只会让清理任务卡住
            log.warning(
                "s3_abort_multipart_failed",
                upload_id=upload.upload_id,
                error=type(exc).__name__,
            )

    async def presigned_url(self, key: str, *, expires_in_s: int = 900) -> str:
        """签发临时下载直链。

        默认 15 分钟。**签发前必须校验权限**（service 层职责）：
        签名 URL 一旦发出，在有效期内任何人都能访问。
        """
        sanitize_key(key)
        return await asyncio.to_thread(self._presign_sync, key, expires_in_s)

    def _presign_sync(self, key: str, expires_in_s: int) -> str:
        client = self._ensure_client()
        try:
            url: str = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in_s,
            )
        except Exception as exc:
            raise self._translate_error(exc, key) from exc
        return url

    async def health(self) -> bool:
        """检查 bucket 可访问。"""
        try:
            await asyncio.to_thread(self._health_sync)
            return True
        except Exception as exc:
            log.warning("s3_health_failed", error=type(exc).__name__)
            return False

    def _health_sync(self) -> None:
        client = self._ensure_client()
        client.head_bucket(Bucket=self._bucket)

    async def aclose(self) -> None:
        """关闭 boto3 client。

        boto3 的 client 内部维护连接池，长期运行需要显式关闭
        以避免连接泄漏（尤其在频繁创建/销毁 client 的场景）。
        """
        if self._client is not None and self._owns_client:
            closer = getattr(self._client, "close", None)
            if callable(closer):
                await asyncio.to_thread(closer)


def iter_file(path: Any) -> Iterator[bytes]:
    """把文件读成字节迭代器（供测试与调试用）。"""
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            yield chunk


def compute_sha256(data: bytes) -> str:
    """计算内容 SHA-256（与 blob 表的 content_hash 对齐）。"""
    return hashlib.sha256(data).hexdigest()


__all__ = ["BinaryIO", "S3StorageBackend", "compute_sha256", "iter_file"]
