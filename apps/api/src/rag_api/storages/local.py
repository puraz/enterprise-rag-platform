"""本地磁盘存储后端。

## 用在哪

- **本地开发/测试**（本机没有 MinIO、Docker）；
- 集成测试（不需要起容器，测试快）；
- 单机部署的小规模场景。

## 关键设计

### 1. 所有文件 I/O 都切线程

本地 I/O 看起来快，但：
- 1GB 文件写入会阻塞事件循环几百毫秒到几秒；
- 机械盘/网络盘（NFS）更慢；
- 并发读写时互相争抢，延迟不可预测。

所以每个 I/O 操作都 `asyncio.to_thread`。这不是过度设计——
`AGENTS.md` 明确禁止同步阻塞调用进事件循环。

### 2. multipart 模拟 S3 语义

本地没有原生 multipart，用**分片目录**模拟：

    {root}/mpu/{upload_id}/part-00001
    {root}/mpu/{upload_id}/part-00002
    ...
    complete 时按序拼接 → {root}/{key}，然后删掉分片目录

这样上层代码（service 层）对本地和 S3 用完全相同的调用序列。

### 3. 并发安全的两个点

**同一 upload_id 的不同分片并发上传**：每个分片写独立文件，
天然不冲突。

**complete 与 upload_part 竞态**：如果 complete 正在拼接时又来了一个分片，
拼接结果会漏掉它。用每个 upload 的 `asyncio.Lock` 保护——
`complete`/`abort` 拿写锁，`upload_part` 拿读锁（简单起见都用同一把锁）。
**注意锁只在本进程有效**：多 worker 部署时本地后端本来就不该用
（应该用 S3）。

### 4. ETag 用内容 SHA-256 模拟

S3 的 ETag 是 MD5（分片时是特殊算法）。本地用 SHA-256 的十六进制
前 32 位模拟，够用且更快（不用引 MD5 的 FIPS 问题）。
**上层不应依赖 ETag 的具体算法**——只用于校验一致性。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from rag_api.storages.base import (
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

# 分片编号从 1 开始（S3 语义）。超过这个值 S3 也会拒绝。
MAX_PARTS = 10_000
# 单个分片最小 5MB（S3 要求，最后一片除外）。本地不限但保持一致。
MIN_PART_SIZE = 5 * 1024 * 1024


class LocalStorageBackend(StorageBackend):
    """本地文件系统存储。"""

    name = "local"

    def __init__(self, settings: StorageSettings) -> None:
        self._settings = settings
        self._root = Path(settings.local_root).resolve()
        self._mpu_root = self._root / "mpu"
        # 每个 upload 一把锁：保护 complete/abort 与 upload_part 的竞态。
        # 字典本身在多协程下访问安全（CPython GIL），锁的创建用 setdefault 幂等。
        self._locks: dict[str, asyncio.Lock] = {}
        self._init_dirs()

    def _init_dirs(self) -> None:
        """创建根目录。用同步 I/O——它在 __init__ 里，进程还没开始服务。

        也可以用 to_thread，但构造对象时没有事件循环更简单，
        且目录创建只需几毫秒。
        """
        self._root.mkdir(parents=True, exist_ok=True)
        self._mpu_root.mkdir(parents=True, exist_ok=True)

    @property
    def capabilities(self) -> StorageCapabilities:
        return StorageCapabilities(
            supports_multipart=True,
            supports_presigned_url=False,  # 本地没有这个概念
            supports_server_side_encryption=False,
            # 本地受文件系统限制；单机通常不需要超过这个
            max_object_size_bytes=100 * 1024**3,
            is_local=True,
        )

    # ---- 路径 ----

    def _path_for(self, key: str) -> Path:
        """把对象键映射到文件路径。

        **必须经过 sanitize_key**：恶意的 key（`../../etc/passwd`）
        在拼接后会逃逸出 root。这是本地后端最关键的安全检查。
        """
        sanitized = sanitize_key(key)
        path = (self._root / sanitized).resolve()
        # 双重保险：resolve 后再确认仍在 root 下
        # （符号链接可能让 resolve 后的路径跳出 root）
        if not str(path).startswith(str(self._root) + os.sep) and path != self._root:
            raise StorageError("对象键逃逸出存储根目录", retryable=False)
        return path

    def _mpu_dir(self, upload_id: str) -> Path:
        """分片目录。upload_id 由自己生成（UUID），不需净化。"""
        return self._mpu_root / upload_id

    def _lock_for(self, upload_id: str) -> asyncio.Lock:
        return self._locks.setdefault(upload_id, asyncio.Lock())

    # ---- 基本操作 ----

    async def put(
        self,
        key: str,
        data: AsyncIterator[bytes] | bytes,
        *,
        content_type: str | None = None,
    ) -> ObjectMetadata:
        """写入对象。

        - `bytes`：一次性写入（小文件，或调用方已经拼好在内存里）；
        - 异步迭代器：**流式写入**，逐块落盘，不整文件入内存。

        两条路径都走同一套"临时文件 + 原子重命名"，避免失败时
        留下半个文件（`head` 会误认为对象完整——这是静默数据损坏）。
        """
        path = self._path_for(key)

        if isinstance(data, bytes):
            return await asyncio.to_thread(self._put_bytes_sync, path, key, data)
        return await self._put_stream_async(path, key, data)

    def _put_bytes_sync(self, path: Path, key: str, data: bytes) -> ObjectMetadata:
        """同步写入字节串（在线程池里跑）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex[:8]}")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)  # 原子替换
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return ObjectMetadata(
            key=key,
            size_bytes=len(data),
            content_type=None,
            etag=hashlib.sha256(data).hexdigest()[:32],
        )

    async def _put_stream_async(
        self, path: Path, key: str, chunks: AsyncIterator[bytes]
    ) -> ObjectMetadata:
        """流式写入异步迭代器。

        **不能把异步迭代器交给线程池**（线程里没有事件循环）。
        做法：在主循环里逐块 `await`，每块写盘用 `to_thread`。
        块大小由生产方决定（通常是 64KB-1MB）。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex[:8]}")
        digest = hashlib.sha256()
        size = 0

        handle = await asyncio.to_thread(tmp.open, "wb")
        try:
            async for chunk in chunks:
                if not chunk:
                    continue
                digest.update(chunk)
                size += len(chunk)
                await asyncio.to_thread(handle.write, chunk)
            await asyncio.to_thread(handle.flush)
        except Exception:
            await asyncio.to_thread(handle.close)
            await asyncio.to_thread(tmp.unlink, True)
            raise
        else:
            await asyncio.to_thread(handle.close)
            await asyncio.to_thread(os.replace, tmp, path)

        return ObjectMetadata(
            key=key,
            size_bytes=size,
            content_type=None,
            etag=digest.hexdigest()[:32],
        )

    def get(self, key: str) -> AsyncIterator[bytes]:
        """流式读取（返回异步迭代器，本身不是协程）。

        按固定块大小读，避免大文件整块入内存。
        实际读逻辑在内部 `async def` 里：外层是普通 def，
        因为「返回异步迭代器」与「本身是协程」是两件事。
        """
        path = self._path_for(key)
        # 同步的存在性检查：is_file 是内存态的 stat，微秒级，
        # 不值得为它切线程。不存在时直接抛（调用方通常用 head 先探）。
        if not path.is_file():
            raise ObjectNotFoundError(key)
        return self._iter_file(path)

    async def _iter_file(self, path: Path) -> AsyncIterator[bytes]:
        """真正的分块读取。"""
        chunk_size = self._settings.read_chunk_size
        # 用同步文件句柄 + to_thread 逐块读：
        # 每次 read 都是小 I/O，切线程的开销可接受，
        # 且比 aiofiles 少一个依赖。
        handle = await asyncio.to_thread(path.open, "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(handle.read, chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(handle.close)

    async def delete(self, key: str) -> None:
        """删除对象。不存在时不报错（幂等）。"""
        path = self._path_for(key)
        await asyncio.to_thread(path.unlink, True)  # missing_ok=True

    async def exists(self, key: str) -> bool:
        path = self._path_for(key)
        return await asyncio.to_thread(path.is_file)

    async def head(self, key: str) -> ObjectMetadata:
        path = self._path_for(key)
        stat = await asyncio.to_thread(self._stat_or_raise, path, key)
        return ObjectMetadata(
            key=key,
            size_bytes=stat[0],
            content_type=None,
            etag=stat[1],
            last_modified=stat[2],
        )

    def _stat_or_raise(self, path: Path, key: str) -> tuple[int, str, str]:
        if not path.is_file():
            raise ObjectNotFoundError(key)
        stat = path.stat()
        # ETag 用文件内容 hash（小文件直接读，大文件用 mtime+size 近似）
        if stat.st_size <= 8 * 1024 * 1024:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:32]
        else:
            digest = hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:32]
        from datetime import UTC, datetime

        modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        return stat.st_size, digest, modified

    # ---- 分片上传 ----

    async def initiate_multipart(
        self, key: str, *, content_type: str | None = None
    ) -> MultipartUpload:
        """创建分片上传会话。

        key 只做校验（防止后续拼接时越界），实际路径在 complete 时才用。
        """
        sanitize_key(key)  # 提前校验：非法 key 不该占用 upload_id
        upload_id = uuid.uuid4().hex
        mpu_dir = self._mpu_dir(upload_id)
        await asyncio.to_thread(mpu_dir.mkdir, parents=True, exist_ok=False)
        return MultipartUpload(upload_id=upload_id, key=key, backend=self.name)

    async def upload_part(
        self,
        upload: MultipartUpload,
        part_number: int,
        data: bytes,
    ) -> PartInfo:
        """上传一个分片。

        `part_number` 必须 ≥ 1（S3 语义）。传 0 是经典 off-by-one bug。
        """
        if part_number < 1:
            raise StorageError(f"分片编号必须 ≥ 1（S3 语义），收到 {part_number}", retryable=False)
        if part_number > MAX_PARTS:
            raise StorageError(f"分片编号超过上限 {MAX_PARTS}", retryable=False)

        mpu_dir = self._mpu_dir(upload.upload_id)
        if not await asyncio.to_thread(mpu_dir.is_dir):
            raise StorageError(f"上传会话不存在: {upload.upload_id}", retryable=False)

        part_path = mpu_dir / f"part-{part_number:05d}"
        digest = hashlib.sha256(data).hexdigest()[:32]

        async with self._lock_for(upload.upload_id):
            await asyncio.to_thread(part_path.write_bytes, data)

        return PartInfo(part_number=part_number, etag=digest, size_bytes=len(data))

    async def list_parts(self, upload: MultipartUpload) -> list[PartInfo]:
        """列出已上传分片（断点续传的权威数据源）。"""
        mpu_dir = self._mpu_dir(upload.upload_id)
        return await asyncio.to_thread(self._list_parts_sync, mpu_dir)

    def _list_parts_sync(self, mpu_dir: Path) -> list[PartInfo]:
        if not mpu_dir.is_dir():
            raise StorageError("上传会话不存在", retryable=False)
        parts: list[PartInfo] = []
        for entry in sorted(mpu_dir.glob("part-*")):
            try:
                number = int(entry.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            content = entry.read_bytes()
            parts.append(
                PartInfo(
                    part_number=number,
                    etag=hashlib.sha256(content).hexdigest()[:32],
                    size_bytes=len(content),
                )
            )
        return parts

    async def complete_multipart(
        self, upload: MultipartUpload, parts: list[PartInfo]
    ) -> ObjectMetadata:
        """按 part_number 升序拼接分片。

        **必须排序**：客户端可能乱序上传，但拼装必须按序号。
        不排序会得到内容错乱的文件——且大小正确，极难发现。
        """
        if not parts:
            raise StorageError("没有可合并的分片", retryable=False)

        numbers = [p.part_number for p in parts]
        expected = list(range(1, max(numbers) + 1))
        if sorted(numbers) != expected:
            missing = sorted(set(expected) - set(numbers))
            raise StorageError(f"分片不连续，缺少 {missing[:10]}", retryable=False)

        target = self._path_for(upload.key)
        async with self._lock_for(upload.upload_id):
            return await asyncio.to_thread(
                self._complete_sync, upload, sorted(parts, key=lambda p: p.part_number), target
            )

    def _complete_sync(
        self, upload: MultipartUpload, parts: list[PartInfo], target: Path
    ) -> ObjectMetadata:
        mpu_dir = self._mpu_dir(upload.upload_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + f".tmp-{uuid.uuid4().hex[:8]}")

        digest = hashlib.sha256()
        size = 0
        try:
            with tmp.open("wb") as out:
                for part in parts:
                    part_path = mpu_dir / f"part-{part.part_number:05d}"
                    if not part_path.is_file():
                        raise StorageError(f"分片 {part.part_number} 文件缺失", retryable=False)
                    with part_path.open("rb") as src:
                        while chunk := src.read(1024 * 1024):
                            out.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        # 合并成功后清理分片目录（否则存储泄漏）
        import shutil

        shutil.rmtree(mpu_dir, ignore_errors=True)
        self._locks.pop(upload.upload_id, None)

        return ObjectMetadata(
            key=upload.key,
            size_bytes=size,
            etag=digest.hexdigest()[:32],
        )

    async def abort_multipart(self, upload: MultipartUpload) -> None:
        """中止并清理分片目录。"""
        mpu_dir = self._mpu_dir(upload.upload_id)
        async with self._lock_for(upload.upload_id):
            import shutil

            await asyncio.to_thread(shutil.rmtree, mpu_dir, True)
            self._locks.pop(upload.upload_id, None)

    async def health(self) -> bool:
        """检查根目录可写。"""
        try:
            probe = self._root / f".health-{uuid.uuid4().hex[:8]}"
            await asyncio.to_thread(probe.write_text, "ok")
            await asyncio.to_thread(probe.unlink)
            return True
        except Exception as exc:
            log.warning("storage_health_failed", error=type(exc).__name__)
            return False
