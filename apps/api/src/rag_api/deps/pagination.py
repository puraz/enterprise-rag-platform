"""分页参数依赖。

## 为什么要有这个依赖

如果每个列表接口自己声明 `limit`/`offset` 的校验，会出现：
- 上限不一致（A 接口 50、B 接口 1000）；
- 有的接口允许 offset=1000000，一次请求扫全表；
- 前端要为每个接口写不同的参数处理。

统一成一个依赖后，规则只有一处。
"""

from __future__ import annotations

from fastapi import Depends, Query

from rag_contracts.api.pagination import MAX_OFFSET, PageRequest
from rag_core.errors import BadRequestError


def page_query(
    limit: int = Query(default=20, ge=1, le=100, description="每页条数，上限 100"),
    offset: int = Query(default=0, ge=0, description="跳过的条数"),
    include_total: bool = Query(
        default=False,
        description="是否返回总条数（会触发额外 COUNT 查询，大表上较慢）",
    ),
) -> PageRequest:
    """解析分页参数。

    `offset` 超过安全上限时返回 400，而不是让它慢慢扫：
    百万级表上 offset=100000 要扫描并丢弃十万行，通常已经秒级。
    引导前端改用过滤条件收窄范围（例如按状态、时间范围筛）。

    上限是 10000（见 `MAX_OFFSET`），对管理端的翻页体验足够，
    又不会让单次请求退化成全表扫描。
    """
    if offset > MAX_OFFSET:
        raise BadRequestError(
            f"offset 不能超过 {MAX_OFFSET}，请改用过滤条件缩小范围",
            details={"max_offset": MAX_OFFSET, "requested": offset},
        )
    return PageRequest(limit=limit, offset=offset, include_total=include_total)


PageQuery = Depends(page_query)
