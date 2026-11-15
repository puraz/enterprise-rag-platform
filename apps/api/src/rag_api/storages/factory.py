"""存储后端工厂。

## 为什么后端实例是进程级单例

S3 的 boto3 client 内部维护连接池，每请求新建会导致：
- 每次 TLS 握手（几十毫秒）；
- 连接数暴涨打满 fd；
- 凭证链被反复解析。

所以后端实例挂在 `app.state` 上（与 Database/Redis 同样的模式）。

## 为什么要一个"多后端注册表"而不是单个实例

`blobs.storage_backend` 记录了每个对象存在哪个后端（A3-2 设计）。
数据迁移期间（本地 → S3）会同时存在两个后端：
- 老的 blob 在 local，新的在 s3；
- 读取时必须按 blob 记录的 backend 选择实现。

所以工厂返回 `dict[str, StorageBackend]`，而不是单个实例。
"""

from __future__ import annotations

from collections.abc import Callable

from rag_api.storages.base import StorageBackend
from rag_api.storages.local import LocalStorageBackend
from rag_api.storages.s3 import S3StorageBackend
from rag_core.errors import ConfigError
from rag_core.settings import Settings

_BUILDERS: dict[str, Callable[[Settings], StorageBackend]] = {
    "local": lambda settings: LocalStorageBackend(settings.storage),
    "s3": lambda settings: S3StorageBackend(settings.storage),
}


def build_storage(settings: Settings) -> StorageBackend:
    """按配置构造**主**存储后端（当前写入目标）。"""
    name = settings.storage.backend
    builder = _BUILDERS.get(name)
    if builder is None:
        known = ", ".join(sorted(_BUILDERS))
        raise ConfigError(f"未知的存储后端 {name!r}（可选：{known}）")
    return builder(settings)


def build_all_storages(settings: Settings) -> dict[str, StorageBackend]:
    """构造所有可用后端（用于按 blob 记录跨后端读取）。

    **只构造配置里指定的那个 + 本地后端**：
    - 配置的主后端总是可用（当前写入目标）；
    - local 总是可用（迁移期读老数据、测试、降级）。

    不构造所有后端的原因：S3 client 创建会读凭证链，没有凭证时
    会阻塞在 IMDS 元数据查询上（容器里超时 1-2 秒）。
    """
    storages: dict[str, StorageBackend] = {}
    primary = build_storage(settings)
    storages[primary.name] = primary

    if primary.name != "local" and settings.storage.local_root:
        storages["local"] = LocalStorageBackend(settings.storage)

    return storages


def get_storage(storages: dict[str, StorageBackend], name: str) -> StorageBackend:
    """按后端名取实例。

    blob 记录里存了 storage_backend，读取时必须用它——
    否则迁移期间老数据（在 local）会读不到。
    """
    backend = storages.get(name)
    if backend is None:
        raise ConfigError(f"存储后端 {name!r} 未初始化（可能是配置变更导致的）")
    return backend


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))
