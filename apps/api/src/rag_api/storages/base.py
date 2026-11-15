"""对象存储抽象。

## 为什么需要这一层

`docs/PROMPT.md` 要求"换厂商只改配置不改代码"。S3 协议已被 AWS S3、
MinIO、阿里 OSS、腾讯 COS、七牛、Backblaze B2 广泛兼容，所以一个
`S3StorageBackend` + `endpoint_url` 就能覆盖绝大多数场景；
本地开发用 `LocalStorageBackend`（无需 MinIO）。

## 接口设计的三个关键点

### 1. 全部异步，但 boto3 是同步的

boto3 **没有**官方异步版本（`aioboto3` 是第三方封装，且有连接池复用问题）。
直接用 boto3 会**阻塞事件循环**——`AGENTS.md` 明确禁止。

本层要求所有实现的方法都是 `async`：
- `LocalStorageBackend` 用 `asyncio.to_thread` 包文件 I/O；
- `S3StorageBackend` 用 `asyncio.to_thread` 包 boto3 调用。

**不要**用 `run_in_executor` 但忘记传播 contextvars（A2-3 的
`contextual_call` 就是为此存在）。这里用 `asyncio.to_thread`，
它内部由 asyncio 管理线程池，且默认会复制 context。

### 2. 流式读写，不整文件入内存

1GB 的文件一次性读进内存会导致：单请求吃掉 1GB、并发几个就 OOM。
所以 `download` 返回 `AsyncIterator[bytes]`，`upload` 接受
`AsyncIterator[bytes]` 或可 seek 的 file-like。

### 3. 分片上传的两套语义

| 后端 | multipart 实现 |
|---|---|
| S3 | 原生 `create_multipart_upload` / `upload_part` / `complete` |
| 本地 | 把每个分片写成独立文件，complete 时按序拼接 |

**接口统一成 S3 的语义**（upload_id + part_number + ETag），
因为它是事实标准；本地实现模拟它。这样换后端时上层代码不变。

## 安全边界

**存储层不做业务校验**（文件大小上限、格式白名单、病毒扫描都不在这里）。
那些属于 service 层：它们需要数据库、配置、审计上下文。
存储层只负责"把字节可靠地放进/取出某个 key"。

唯一属于存储层的安全职责是 **key 净化**：拒绝 `../` 这类路径穿越，
因为它是存储实现自身的安全边界（一个恶意 key 可能写到文件系统任意位置）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    """对象元信息（不下载内容就能拿到）。"""

    key: str
    size_bytes: int
    content_type: str | None = None
    etag: str | None = None
    """S3 的 ETag。本地实现用内容 SHA-256 模拟。"""

    last_modified: str | None = None


@dataclass(frozen=True, slots=True)
class MultipartUpload:
    """分片上传会话（S3 语义）。"""

    upload_id: str
    key: str
    backend: str
    """后端标识。完成时需要知道用哪个后端，因为会话可能跨越重启。"""

    bucket_hint: str | None = None


@dataclass(frozen=True, slots=True)
class PartInfo:
    """已上传的分片信息。"""

    part_number: int
    etag: str
    size_bytes: int = 0


class StorageError(Exception):
    """存储操作失败。不继承 AppError（与 Provider 层同样的理由）。"""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class ObjectNotFoundError(StorageError):
    """对象不存在。**不可重试**——重试也不会出现。"""

    def __init__(self, key: str) -> None:
        super().__init__(f"对象不存在: {key}", retryable=False)
        self.key = key


@dataclass(frozen=True, slots=True)
class StorageCapabilities:
    """后端能力声明。上层据此决定用哪条路径，不做名字判断。"""

    supports_multipart: bool = True
    supports_presigned_url: bool = False
    """能否签发带时效的直链（S3 支持，本地不支持）。"""

    supports_server_side_encryption: bool = False
    max_object_size_bytes: int = 5 * 1024**3
    """单对象上限。S3 是 5TB，本地受文件系统限制。"""

    is_local: bool = False


class StorageBackend(ABC):
    """对象存储接口。

    实现约定：
    - 所有方法必须**异步**且不阻塞事件循环；
    - 大文件必须**流式**处理，不得整文件入内存；
    - key 必须经过 `sanitize_key` 校验；
    - 不做业务校验（大小上限、格式白名单在 service 层）。
    """

    name: str = "unknown"

    @property
    @abstractmethod
    def capabilities(self) -> StorageCapabilities:
        raise NotImplementedError

    # ---- 基本操作 ----

    @abstractmethod
    async def put(
        self,
        key: str,
        data: AsyncIterator[bytes] | bytes,
        *,
        content_type: str | None = None,
    ) -> ObjectMetadata:
        """写入对象（小文件或不走分片的直接上传）。

        Args:
            data: 字节串或异步迭代器。**用迭代器时不能回退读取**——
                流式写入不做缓冲，传了就消费掉。
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str) -> AsyncIterator[bytes]:
        """读取对象，流式返回。

        **声明为普通 def 而不是 async def**：它返回的是异步迭代器，
        本身不是协程。写成 `async def ... -> AsyncIterator` 会让调用方
        得到 `Coroutine[...]`，`async for` 会报 "not async iterable"
        ——mypy 能抓到，但运行时错误信息很难懂。
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        """删除对象。

        对象不存在时**不报错**（幂等）：清理任务重跑时不该因为
        对象已被删掉而失败。
        """
        raise NotImplementedError

    @abstractmethod
    async def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def head(self, key: str) -> ObjectMetadata:
        """取对象元信息。不存在时抛 `ObjectNotFoundError`。"""
        raise NotImplementedError

    # ---- 分片上传（S3 语义）----

    @abstractmethod
    async def initiate_multipart(
        self, key: str, *, content_type: str | None = None
    ) -> MultipartUpload:
        """创建分片上传会话。"""
        raise NotImplementedError

    @abstractmethod
    async def upload_part(
        self,
        upload: MultipartUpload,
        part_number: int,
        data: bytes,
    ) -> PartInfo:
        """上传一个分片。

        `part_number` 从 1 开始（S3 规定）。传 0 会被拒绝——
        这个 off-by-one 是最常见的分片上传 bug。
        """
        raise NotImplementedError

    @abstractmethod
    async def complete_multipart(
        self, upload: MultipartUpload, parts: list[PartInfo]
    ) -> ObjectMetadata:
        """合并分片。

        **`parts` 必须按 part_number 升序**：S3 要求有序，
        本地实现也按这个顺序拼接。
        """
        raise NotImplementedError

    @abstractmethod
    async def abort_multipart(self, upload: MultipartUpload) -> None:
        """中止分片上传并清理已上传的分片（避免存储泄漏）。"""
        raise NotImplementedError

    @abstractmethod
    async def list_parts(self, upload: MultipartUpload) -> list[PartInfo]:
        """列出已上传的分片。

        **这是断点续传的权威数据源**：客户端重连后问"我传到哪了"，
        答案必须来自存储本身，而不是 Redis 缓存或数据库计数
        （那些可能与实际不一致）。
        """
        raise NotImplementedError

    # ---- 可选能力 ----

    async def presigned_url(self, key: str, *, expires_in_s: int = 900) -> str:
        """签发临时下载直链。

        默认抛 `NotImplementedError`：本地后端没有这个概念。
        调用方应先检查 `capabilities.supports_presigned_url`。
        """
        raise NotImplementedError(f"{self.name} 不支持预签名 URL")

    async def health(self) -> bool:
        """后端是否可用。"""
        return True


# ---------------------------------------------------------------- key 净化


def sanitize_key(key: str) -> str:
    """校验并净化对象键。

    **这是存储层唯一的安全职责**：恶意的 key（含 `../`）在文件系统
    后端可能写到任意位置，在 S3 后端虽然不会但有其他风险
    （比如覆盖他人对象）。

    规则：
    - 禁止绝对路径与前导 `/`；
    - 禁止 `..` 段（路径穿越）；
    - 禁止反斜杠（Windows 路径分隔符可能被解释为目录）；
    - 禁止空字节与控制字符；
    - 禁止以 `.` 开头的段（避免生成隐藏文件 / S3 的相对引用）。
    """
    if not key:
        raise StorageError("对象键不能为空", retryable=False)

    if key.startswith("/"):
        raise StorageError("对象键不能以 / 开头", retryable=False)

    # 空字节会截断 C 层的字符串处理，是经典的文件系统攻击向量
    if "\x00" in key:
        raise StorageError("对象键含空字节", retryable=False)

    if "\\" in key:
        raise StorageError("对象键不能含反斜杠", retryable=False)

    segments = key.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise StorageError(f"对象键含非法路径段: {segment!r}", retryable=False)
        if segment.startswith("."):
            raise StorageError("对象键的路径段不能以 . 开头", retryable=False)
        # 控制字符（含换行）会破坏日志与 HTTP 头
        if any(ord(ch) < 32 for ch in segment):
            raise StorageError("对象键含控制字符", retryable=False)

    return key


def build_storage_key(*, tenant_id: str, content_hash: str, suffix: str = "") -> str:
    """生成标准存储键。

    格式：`{tenant}/{hash前2位}/{hash}{suffix}`

    **为什么用 hash 分片**：S3 在单一前缀下大量对象时列目录会变慢
    （虽然 S3 已优化，但某些兼容实现仍有此问题）。用 hash 前两位
    分散到 256 个前缀。

    **为什么不含原始文件名**：
    - 文件名可能含用户隐私信息（合同.pdf、张三的体检报告.pdf）；
    - 文件名可能含恶意字符（虽然会净化，但不进 key 更安全）；
    - 重命名文档不应改变存储位置。
    """
    prefix = content_hash[:2] if len(content_hash) >= 2 else "00"
    return f"{tenant_id}/{prefix}/{content_hash}{suffix}"
