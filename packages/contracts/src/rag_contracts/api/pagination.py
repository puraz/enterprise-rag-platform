"""分页与响应包装契约。

## 为什么成功响应不做 `{code, message, data}` 外层包裹

两种主流做法：

| 做法 | 问题 |
|---|---|
| 全部包 `{code:0, data:...}` | 前端每层都要 `.data`；HTTP 状态码与业务 `code` 双轨，
  容易出现"200 但 code=500"这种歧义 |
| **成功返回裸数据、失败返回 ErrorResponse**（本项目） | 需要把错误响应
  **声明进 OpenAPI**，否则前端生成类型时看不到 |

第二种是 GitHub / Stripe 的做法。代价是必须显式声明错误响应类型
（见 `rag_api.openapi.responses`），否则前端就无法分支处理错误。

## 分页形态（用户已确认）

扁平结构 + limit/offset：

    {
      "items": [...],
      "meta": {"limit": 20, "offset": 0, "total": 123, "has_more": true}
    }

`total` 是 `int | null`：
- 前端做无限滚动时只用 `has_more`，**不传 `include_total` 就不会有 COUNT 查询**；
- 管理端需要"共 N 条"时才请求 total。

这是对 `AGENTS.md` 原"禁用 OFFSET"约束的**有意识放宽**：
offset 深翻页会随页数线性变慢（PG 要扫描并丢弃前 N 行）。
缓解手段：
1. 列表接口默认 `limit <= 100`，限制单页规模；
2. `offset` 超过阈值（默认 10000）时返回 400，引导前端改用过滤条件收窄范围；
3. 需要深翻页的大场景（审计日志）单独提供游标接口。

## 为什么 meta 里的字段用 `limit/offset` 而不是 `page/size`

`limit/offset` 是数据库与 API 的通用语言，前端换算页码只需
`offset = (page - 1) * limit`。用 `page/size` 会让服务端多一层换算，
且与 SQL 的 LIMIT/OFFSET 概念脱节。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PageMeta(BaseModel):
    """分页元信息。"""

    model_config = ConfigDict(frozen=True)

    limit: int = Field(description="每页条数（服务端上限 100）", ge=1, le=100)
    offset: int = Field(description="跳过的条数", ge=0)
    total: int | None = Field(
        default=None,
        description=(
            "总条数。仅在请求带 `include_total=true` 时返回，"
            "否则为 null —— 大表上 COUNT(*) 很慢，不应默认执行"
        ),
    )
    has_more: bool = Field(description="是否还有下一页。无限滚动只需看这个字段")


class Page[T](BaseModel):
    """分页响应。

    用法（路由层）：

        @router.get("/documents", response_model=Page[DocumentSummary])
        async def list_documents(...) -> Page[DocumentSummary]:
            return Page(items=rows, meta=PageMeta(...))

    **泛型模型在 TS 生成时会出现 `Page_DocumentSummary_` 这样的名字**：
    Pydantic 对参数化泛型的命名是 `Page[DocumentSummary]`，
    生成器会转换成下划线形式。可读性一般但类型正确，
    前端用 `Page_DocumentSummary_` 或自行 `type DocumentPage = ...` 别名即可。
    """

    model_config = ConfigDict(frozen=True)

    items: list[T] = Field(description="当前页数据")
    meta: PageMeta = Field(description="分页元信息")


class PageRequest(BaseModel):
    """分页请求参数（可作为 query 依赖使用）。

    路由层用法：

        async def list_documents(page: PageRequest = Depends(page_query)): ...

    校验规则放在这里而不是每个路由各写一遍——否则不同接口的
    上限会不一致（A 接口 50、B 接口 1000），前端无法统一处理。
    """

    model_config = ConfigDict(frozen=True)

    limit: int = Field(default=20, ge=1, le=100, description="每页条数，上限 100")
    offset: int = Field(default=0, ge=0, description="跳过的条数")
    include_total: bool = Field(
        default=False,
        description="是否返回总条数（会触发额外 COUNT 查询）",
    )


# offset 的安全上限：超过后拒绝，引导前端用过滤条件收窄。
# 10000 行的偏移在百万级表上约需扫描 10000 行并丢弃 —— 已在百毫秒量级。
MAX_OFFSET = 10_000
