"""通用请求/响应实体（分页等）。

分页契约的唯一真相在 `rag_contracts.api.pagination`；这里集中 re-export，
让路由从一个目录拿到所有请求/响应实体。
"""

from __future__ import annotations

from rag_contracts.api.pagination import MAX_OFFSET, Page, PageMeta, PageRequest

__all__ = ["MAX_OFFSET", "Page", "PageMeta", "PageRequest"]
