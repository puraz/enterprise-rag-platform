"""对象存储层：抽象接口 + 本地/S3 实现 + 工厂。"""

from rag_api.storages.base import (
    MultipartUpload,
    ObjectMetadata,
    ObjectNotFoundError,
    PartInfo,
    StorageBackend,
    StorageCapabilities,
    StorageError,
    build_storage_key,
    sanitize_key,
)
from rag_api.storages.factory import (
    available_backends,
    build_all_storages,
    build_storage,
    get_storage,
)
from rag_api.storages.local import LocalStorageBackend
from rag_api.storages.s3 import S3StorageBackend

__all__ = [
    "LocalStorageBackend",
    "MultipartUpload",
    "ObjectMetadata",
    "ObjectNotFoundError",
    "PartInfo",
    "S3StorageBackend",
    "StorageBackend",
    "StorageCapabilities",
    "StorageError",
    "available_backends",
    "build_all_storages",
    "build_storage",
    "build_storage_key",
    "get_storage",
    "sanitize_key",
]
