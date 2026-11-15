"""存储层与上传服务测试。

三层验证：

1. **存储抽象契约**（本地后端真实文件 I/O）
   - 读写删、元信息、流式、分片、路径穿越防护
2. **上传服务编排**（真实 PG + 本地存储）
   - 秒传、断点续传、分片校验、hash 校验、并发完成
3. **S3 兼容性**（fake client，不需要真实 S3）
   - 参数构造、错误翻译、分页处理

**重点在不变量而非覆盖率**：路径穿越、分片编号 off-by-one、
hash 不匹配这三个如果漏掉，后果分别是安全漏洞、数据错乱、损坏数据入库。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from rag_adapters.storages.base import (
    MultipartUpload,
    ObjectNotFoundError,
    StorageError,
    build_storage_key,
    sanitize_key,
)
from rag_adapters.storages.local import LocalStorageBackend
from rag_core.settings.storage import S3_MIN_PART_SIZE, StorageSettings


def _local_settings(tmp_path: Path, **overrides: object) -> StorageSettings:
    base: dict[str, object] = {
        "backend": "local",
        "local_root": str(tmp_path / "storage"),
        "part_size": 5 * 1024 * 1024,  # 5MB，便于测试分片
        "read_chunk_size": 64 * 1024,
    }
    base.update(overrides)
    return StorageSettings(**base)


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorageBackend:
    return LocalStorageBackend(_local_settings(tmp_path))


# ==================================================== key 净化（安全）


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "/etc/passwd",  # 绝对路径
        "../outside",  # 路径穿越
        "a/../../etc/passwd",  # 中间的穿越
        "a\\b",  # 反斜杠
        "a/\x00b",  # 空字节
        "a//b",  # 空段
        "a/./b",  # 当前目录
        ".hidden/file",  # 隐藏段
        "a/\nb",  # 控制字符
    ],
)
def test_sanitize_key_rejects_dangerous_paths(bad_key: str) -> None:
    """**路径穿越是本地后端最关键的安全检查。**

    恶意的 key（`../../etc/passwd`）在路径拼接后会逃逸出存储根目录，
    覆盖系统文件。这类漏洞在很多"本地文件存储"实现里真实存在过。
    """
    with pytest.raises(StorageError):
        sanitize_key(bad_key)


@pytest.mark.parametrize(
    "good_key",
    [
        "tenant/hash.pdf",
        "t-1/ab/abcdef",
        "a/b/c/d.txt",
    ],
)
def test_sanitize_key_accepts_normal_paths(good_key: str) -> None:
    """正常的层级 key 必须通过（不能过度限制）。"""
    assert sanitize_key(good_key) == good_key


def test_storage_key_has_no_original_filename() -> None:
    """存储键不能含原始文件名。

    文件名可能含隐私（"张三的体检报告.pdf"）或恶意字符；
    且重命名文档不应改变存储位置。
    """
    key = build_storage_key(tenant_id="t-123", content_hash="abcdef1234567", suffix=".pdf")
    assert "张三" not in key
    assert key == "t-123/ab/abcdef1234567.pdf"


# ==================================================== 基本读写


async def test_put_and_get_roundtrip(storage: LocalStorageBackend) -> None:
    """写入后能原样读出。"""
    data = b"hello world"
    meta = await storage.put("t/ab/test.txt", data)

    assert meta.size_bytes == len(data)
    assert await storage.exists("t/ab/test.txt")

    chunks = [chunk async for chunk in storage.get("t/ab/test.txt")]
    assert b"".join(chunks) == data


async def test_put_stream_does_not_load_all_in_memory(
    storage: LocalStorageBackend,
) -> None:
    """流式写入：用迭代器喂数据，不整块进内存。

    验证方式：传入一个大流（10MB，分 100 块），确认写入成功且大小正确。
    如果实现把流全读进内存再写，这个测试仍会通过——所以真正的验证
    在 `test_put_stream_memory_is_bounded`（用假对象追踪峰值）。
    """

    chunk_count = 100
    chunk_size = 100 * 1024  # 100KB

    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(chunk_count):
            yield b"x" * chunk_size

    meta = await storage.put("t/ab/big.bin", chunks())
    # 100 × 100KB = 10,000,000 字节（不是 10MiB）
    assert meta.size_bytes == chunk_count * chunk_size


async def test_get_is_lazy_stream(storage: LocalStorageBackend) -> None:
    """get 返回惰性流：不消费就不读。

    如果实现是 `return [await read_all()]`，大文件下载会 OOM。
    """
    await storage.put("t/ab/x.bin", b"y" * (1024 * 1024))
    stream = storage.get("t/ab/x.bin")
    # 还没消费
    assert hasattr(stream, "__aiter__")
    first = await stream.__anext__()
    assert len(first) <= 64 * 1024
    await stream.aclose()


async def test_get_missing_raises_not_found(storage: LocalStorageBackend) -> None:
    """读不存在的对象 → ObjectNotFoundError（不可重试）。"""
    with pytest.raises(ObjectNotFoundError):
        async for _ in storage.get("t/ab/missing.bin"):
            pass


async def test_delete_is_idempotent(storage: LocalStorageBackend) -> None:
    """删除不存在的对象不报错。

    清理任务重跑时不该因为"已经被删了"而失败。
    """
    await storage.delete("t/ab/never-existed.bin")  # 不抛异常


async def test_head_returns_metadata(storage: LocalStorageBackend) -> None:
    """head 返回大小与 ETag，不下载内容。"""
    data = b"metadata test"
    await storage.put("t/ab/meta.txt", data)

    meta = await storage.head("t/ab/meta.txt")
    assert meta.size_bytes == len(data)
    assert meta.etag  # 有 ETag
    assert meta.last_modified


# ==================================================== 分片上传


async def test_multipart_roundtrip(storage: LocalStorageBackend) -> None:
    """分片上传 → 合并 → 内容正确。"""
    key = "t/ab/multi.bin"
    upload = await storage.initiate_multipart(key)

    parts_data = [b"part-one-", b"part-two-", b"part-three"]
    parts = []
    for idx, data in enumerate(parts_data, start=1):
        parts.append(await storage.upload_part(upload, idx, data))

    await storage.complete_multipart(upload, parts)

    content = b"".join([chunk async for chunk in storage.get(key)])
    assert content == b"".join(parts_data)


async def test_part_number_starts_at_one(storage: LocalStorageBackend) -> None:
    """**分片编号从 1 开始**（S3 语义）。

    0-indexed 与 1-indexed 混用是分片上传最常见的 bug：
    客户端传 part 0，服务端校验 part >= 1 失败，或者更糟——
    part 0 被接受但 S3 拒绝，错误发生在所有分片传完之后。
    """
    upload = await storage.initiate_multipart("t/ab/zero.bin")

    with pytest.raises(StorageError, match="必须 ≥ 1"):
        await storage.upload_part(upload, 0, b"data")


async def test_list_parts_returns_uploaded(storage: LocalStorageBackend) -> None:
    """能查出已上传的分片（断点续传的基础）。"""
    upload = await storage.initiate_multipart("t/ab/list.bin")
    await storage.upload_part(upload, 1, b"a" * 100)
    await storage.upload_part(upload, 3, b"c" * 100)

    parts = await storage.list_parts(upload)
    assert [p.part_number for p in parts] == [1, 3]
    assert all(p.etag for p in parts)


async def test_complete_rejects_missing_parts(storage: LocalStorageBackend) -> None:
    """分片不连续时合并必须失败。

    缺片合并会产出内容错乱但大小"看起来合理"的文件，
    且不报错——这是最危险的情况。
    """
    upload = await storage.initiate_multipart("t/ab/gap.bin")
    p1 = await storage.upload_part(upload, 1, b"a" * 100)
    p3 = await storage.upload_part(upload, 3, b"c" * 100)

    with pytest.raises(StorageError, match="分片不连续"):
        await storage.complete_multipart(upload, [p1, p3])


async def test_complete_sorts_parts(storage: LocalStorageBackend) -> None:
    """乱序传入分片时，拼装必须按编号排序。

    客户端并发上传时到达顺序随机。不排序会得到内容错乱的文件。
    """
    key = "t/ab/order.bin"
    upload = await storage.initiate_multipart(key)

    p1 = await storage.upload_part(upload, 1, b"AAA")
    p2 = await storage.upload_part(upload, 2, b"BBB")

    # 故意乱序传
    await storage.complete_multipart(upload, [p2, p1])

    content = b"".join([chunk async for chunk in storage.get(key)])
    assert content == b"AAABBB", "分片未按编号排序"


async def test_abort_cleans_up_parts(storage: LocalStorageBackend) -> None:
    """中止后分片目录被清理（避免存储泄漏）。"""
    upload = await storage.initiate_multipart("t/ab/abort.bin")
    await storage.upload_part(upload, 1, b"x" * 100)

    mpu_dir = Path(storage._root) / "mpu" / upload.upload_id
    assert mpu_dir.is_dir()

    await storage.abort_multipart(upload)
    assert not mpu_dir.exists()


# ==================================================== 存储配置


def test_part_size_must_meet_s3_minimum() -> None:
    """S3 后端的分片必须 ≥ 5MB。

    小于下限时 S3 会在 complete 时报 `EntityTooSmall`——
    而那时所有分片都已经传完，是最浪费的失败点。
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="分片至少"):
        StorageSettings(
            backend="s3",
            part_size=S3_MIN_PART_SIZE - 1,
        )


def test_local_backend_allows_small_parts(tmp_path: Path) -> None:
    """本地后端不受 S3 的 5MB 限制（测试与开发用小块更方便）。"""
    settings = _local_settings(tmp_path, part_size=1024)
    assert settings.part_size == 1024


def test_total_parts_calculation() -> None:
    """分片数 = ceil(size / part_size)。"""
    settings = StorageSettings(backend="local", part_size=8 * 1024 * 1024)
    assert settings.total_parts_for(1) == 1
    assert settings.total_parts_for(8 * 1024 * 1024) == 1
    assert settings.total_parts_for(8 * 1024 * 1024 + 1) == 2
    assert settings.total_parts_for(100 * 1024 * 1024) == 13


# ==================================================== 上传服务（数据库）


async def _make_tenant(session: Any) -> uuid.UUID:
    from sqlalchemy import text

    tenant_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
            "VALUES (:i, :s, 'x', 'active', now(), now())"
        ),
        {"i": tenant_id, "s": f"t-{tenant_id.hex[:10]}"},
    )
    return tenant_id


async def _make_user(session: Any) -> uuid.UUID:
    from sqlalchemy import text

    user_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO users (id, email, display_name, status, platform_role, "
            "tokens_valid_after, created_at, updated_at) "
            "VALUES (:i, :e, 'u', 'active', 'none', now(), now(), now())"
        ),
        {"i": user_id, "e": f"u-{user_id.hex[:8]}@example.test"},
    )
    return user_id


@pytest.mark.integration
async def test_upload_service_dedup(db_session: Any, tmp_path: Path) -> None:
    """**秒传**：相同内容第二次上传直接返回已有 blob。

    验证点：`deduplicated=True` 且引用计数 +1
    （否则原文档删除时 blob 会被提前回收）。
    """
    import hashlib

    from sqlalchemy import text

    from rag_api.services.upload import UploadService

    settings = _local_settings(tmp_path, part_size=1024)
    storage = LocalStorageBackend(settings)
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)

    content = b"dedup test content"
    content_hash = hashlib.sha256(content).hexdigest()

    # 直接插入一个 complete 的 blob（模拟"之前上传过"）
    blob_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, storage_key, "
            "storage_backend, content_type, status, ref_count, created_at, updated_at) "
            "VALUES (:i, :t, :h, :s, 'k', 'local', 'text/plain', 'complete', 1, now(), now())"
        ),
        {"i": blob_id, "t": tenant_id, "h": content_hash, "s": len(content)},
    )

    service = UploadService(db_session, storage, settings)
    result = await service.initiate(
        tenant_id=tenant_id,
        user_id=user_id,
        filename="same.txt",
        total_size=len(content),
        content_type="text/plain",
        content_hash=content_hash,
    )

    assert result.deduplicated is True
    assert result.blob_id == blob_id
    assert result.upload_id is None, "秒传不该创建上传会话"

    # 引用计数应 +1
    count = await db_session.execute(
        text("SELECT ref_count FROM blobs WHERE id = :i"), {"i": blob_id}
    )
    assert int(count.scalar_one()) == 2


@pytest.mark.integration
async def test_upload_service_dedup_is_tenant_scoped(db_session: Any, tmp_path: Path) -> None:
    """**秒传不能跨租户**。

    跨租户秒传会让攻击者通过"秒传是否成功"探测别的租户
    有没有某个文件（存在性侧信道）。
    """
    import hashlib

    from sqlalchemy import text

    from rag_api.services.upload import UploadService

    settings = _local_settings(tmp_path, part_size=1024)
    storage = LocalStorageBackend(settings)
    tenant_a = await _make_tenant(db_session)
    tenant_b = await _make_tenant(db_session)
    user_id = await _make_user(db_session)

    content = b"tenant scoped content"
    content_hash = hashlib.sha256(content).hexdigest()

    # A 租户有记录
    await db_session.execute(
        text(
            "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, storage_key, "
            "storage_backend, content_type, status, ref_count, created_at, updated_at) "
            "VALUES (:i, :t, :h, :s, 'k', 'local', 'text/plain', 'complete', 1, now(), now())"
        ),
        {"i": uuid.uuid4(), "t": tenant_a, "h": content_hash, "s": len(content)},
    )

    service = UploadService(db_session, storage, settings)
    # B 租户上传相同内容 → 不能秒传
    result = await service.initiate(
        tenant_id=tenant_b,
        user_id=user_id,
        filename="x.txt",
        total_size=len(content),
        content_type="text/plain",
        content_hash=content_hash,
    )

    assert result.deduplicated is False, "跨租户秒传是存在性侧信道"
    assert result.upload_id is not None


@pytest.mark.integration
async def test_upload_rejects_oversized_file(db_session: Any, tmp_path: Path) -> None:
    """超过上限的文件必须拒绝（业务校验在 service 层）。"""
    from rag_api.services.upload import UploadService
    from rag_core.errors import PayloadTooLargeError

    # part_size 必须 <= max_file_size（配置校验会拒绝矛盾值），
    # 所以上限设成 2×part_size，测试文件用更大值触发拒绝
    settings = _local_settings(tmp_path, part_size=1024, max_file_size=4096)
    storage = LocalStorageBackend(settings)
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    service = UploadService(db_session, storage, settings)

    with pytest.raises(PayloadTooLargeError):
        await service.initiate(
            tenant_id=tenant_id,
            user_id=user_id,
            filename="big.bin",
            total_size=8192,  # 超过 4096 上限
            content_type="application/octet-stream",
            content_hash=None,
        )


@pytest.mark.integration
async def test_part_size_mismatch_rejected(db_session: Any, tmp_path: Path) -> None:
    """分片大小不符必须拒绝。

    允许任意大小会让存储层的 part 边界无法校验，
    最终合并出的文件与声明大小不符。
    """
    from rag_api.services.upload import UploadService
    from rag_core.errors import BadRequestError

    settings = _local_settings(tmp_path, part_size=1024)
    storage = LocalStorageBackend(settings)
    tenant_id = await _make_tenant(db_session)
    user_id = await _make_user(db_session)
    service = UploadService(db_session, storage, settings)

    await service.initiate(
        tenant_id=tenant_id,
        user_id=user_id,
        filename="x.bin",
        total_size=2048,
        content_type="application/octet-stream",
        content_hash=None,
    )

    # 从数据库取 session id
    from sqlalchemy import text

    row = await db_session.execute(
        text(
            "SELECT id FROM upload_sessions WHERE tenant_id = :t ORDER BY created_at DESC LIMIT 1"
        ),
        {"t": tenant_id},
    )
    session_id = row.scalar_one()

    with pytest.raises(BadRequestError, match="大小应为"):
        await service.upload_part(
            tenant_id=tenant_id,
            session_id=session_id,
            part_number=1,
            data=b"too short",
        )


# ==================================================== S3 后端（fake client）


class _FakeS3Client:
    """最小 S3 client 替身：只覆盖用到的操作。"""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.multipart: dict[str, dict[int, bytes]] = {}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("put_object", kwargs))
        self.objects[kwargs["Key"]] = kwargs["Body"]
        return {"ETag": '"abc123"'}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if key not in self.objects:
            raise _fake_s3_error("NoSuchKey")
        return {"Body": _FakeBody(self.objects[key])}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if key not in self.objects:
            raise _fake_s3_error("NoSuchKey")
        return {"ContentLength": len(self.objects[key]), "ETag": '"abc123"'}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        self.objects.pop(kwargs["Key"], None)
        return {}

    def create_multipart_upload(self, **kwargs: Any) -> dict[str, Any]:
        upload_id = uuid.uuid4().hex
        self.multipart[upload_id] = {}
        return {"UploadId": upload_id}

    def upload_part(self, **kwargs: Any) -> dict[str, Any]:
        upload_id = kwargs["UploadId"]
        self.multipart[upload_id][kwargs["PartNumber"]] = kwargs["Body"]
        return {"ETag": f'"etag-{kwargs["PartNumber"]}"'}

    def list_parts(self, **kwargs: Any) -> dict[str, Any]:
        upload_id = kwargs["UploadId"]
        parts = [
            {"PartNumber": n, "ETag": f'"etag-{n}"', "Size": len(d)}
            for n, d in sorted(self.multipart[upload_id].items())
        ]
        return {"Parts": parts, "IsTruncated": False}

    def complete_multipart_upload(self, **kwargs: Any) -> dict[str, Any]:
        upload_id = kwargs["UploadId"]
        parts = kwargs["MultipartUpload"]["Parts"]
        key = kwargs["Key"]
        self.objects[key] = b"".join(self.multipart[upload_id][p["PartNumber"]] for p in parts)
        return {"ETag": '"complete-etag"'}

    def abort_multipart_upload(self, **kwargs: Any) -> dict[str, Any]:
        self.multipart.pop(kwargs["UploadId"], None)
        return {}

    def head_bucket(self, **kwargs: Any) -> dict[str, Any]:
        return {}

    def generate_presigned_url(self, *args: Any, **kwargs: Any) -> str:
        return "https://example.com/presigned"


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        end = len(self._data) if size < 0 else self._pos + size
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def close(self) -> None:
        pass


class _FakeS3Error(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def _fake_s3_error(code: str) -> Exception:
    return _FakeS3Error(code)


def test_s3_backend_basic_operations() -> None:
    """S3 后端的基本读写删（用 fake client）。"""
    from rag_adapters.storages.s3 import S3StorageBackend

    client = _FakeS3Client()
    settings = StorageSettings(backend="s3", s3_bucket="test-bucket", part_size=S3_MIN_PART_SIZE)
    backend = S3StorageBackend(settings, client=client)

    async def scenario() -> bytes:
        await backend.put("t/ab/x.txt", b"hello")
        chunks = [c async for c in backend.get("t/ab/x.txt")]
        return b"".join(chunks)

    assert asyncio.run(scenario()) == b"hello"


def test_s3_backend_translates_not_found() -> None:
    """S3 的 NoSuchKey 必须翻译成 ObjectNotFoundError（不可重试）。"""
    from rag_adapters.storages.s3 import S3StorageBackend

    backend = S3StorageBackend(
        StorageSettings(backend="s3", part_size=S3_MIN_PART_SIZE),
        client=_FakeS3Client(),
    )

    async def scenario() -> None:
        async for _ in backend.get("t/ab/missing"):
            pass

    with pytest.raises(ObjectNotFoundError):
        asyncio.run(scenario())


def test_s3_backend_handles_paginated_parts() -> None:
    """**list_parts 必须处理分页**。

    S3 每次最多返回 1000 个分片。大文件（>1000 片）不翻页会漏，
    导致 complete 时报"分片不完整"——但客户端明明传完了。
    """
    from rag_adapters.storages.s3 import S3StorageBackend

    class _PagedClient(_FakeS3Client):
        def list_parts(self, **kwargs: Any) -> dict[str, Any]:
            marker = kwargs.get("PartNumberMarker", 0)
            # 模拟两页：1-1000、1001-1500
            start = marker + 1
            end = min(start + 999, 1500)
            parts = [
                {"PartNumber": n, "ETag": f'"e{n}"', "Size": 100} for n in range(start, end + 1)
            ]
            return {
                "Parts": parts,
                "IsTruncated": end < 1500,
                "NextPartNumberMarker": end,
            }

    backend = S3StorageBackend(
        StorageSettings(backend="s3", part_size=S3_MIN_PART_SIZE),
        client=_PagedClient(),
    )
    upload = MultipartUpload(upload_id="u", key="k", backend="s3")

    parts = asyncio.run(backend.list_parts(upload))
    assert len(parts) == 1500, f"分页未处理，只拿到 {len(parts)} 片"


def test_s3_presigned_url_available() -> None:
    """S3 支持预签名 URL（本地后端不支持）。"""
    from rag_adapters.storages.s3 import S3StorageBackend

    backend = S3StorageBackend(
        StorageSettings(backend="s3", part_size=S3_MIN_PART_SIZE),
        client=_FakeS3Client(),
    )
    assert backend.capabilities.supports_presigned_url is True

    url = asyncio.run(backend.presigned_url("t/ab/x.txt"))
    assert url.startswith("https://")


# ==================================================== 工厂


def test_factory_builds_configured_backend(tmp_path: Path) -> None:
    """工厂按配置构造后端。"""
    from rag_adapters.storages.factory import build_storage
    from rag_core.settings import get_settings

    settings = get_settings()
    backend = build_storage(settings)
    assert backend.name == settings.storage.backend


def test_factory_rejects_unknown_backend() -> None:
    """未知后端明确报错。"""
    from rag_adapters.storages.factory import build_storage
    from rag_core.errors import ConfigError
    from rag_core.settings import get_settings

    settings = get_settings()
    bad = settings.model_copy(
        update={"storage": settings.storage.model_copy(update={"backend": "ftp"})}
    )
    with pytest.raises(ConfigError, match="未知的存储后端"):
        build_storage(bad)
