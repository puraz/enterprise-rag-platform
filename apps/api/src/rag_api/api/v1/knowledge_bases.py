"""知识库路由。

## 端点设计

| 方法 | 路径 | 状态码 | 权限 |
|---|---|---|---|
| GET | `/knowledge-bases` | 200 | 登录 + 可见性过滤 |
| POST | `/knowledge-bases` | **201** + `Location` | `require_kb_write()` |
| GET | `/knowledge-bases/{id}` | 200 | 不可见 404 |
| PATCH | `/knowledge-bases/{id}` | 200 | `require_kb_write()` |
| DELETE | `/knowledge-bases/{id}` | **204** | `require_kb_write()` |

## 为什么创建返回 201 而不是 200

`201 + Location` 是"创建了一个新资源"的标准表达。前端据此知道
"应该跳到详情页"而不是"刷新当前列表"。返回 200 会让"创建"与
"更新"在客户端无法区分。

## 为什么删除返回 204 而不是 200 + `{ok: true}`

`204` 表示"成功且没有内容可返回"。DELETE 之后前端不缺任何信息
（它本来就要刷新列表）。返回 `{ok:true}` 会让前端多解析一次 JSON，
而且引入了第四种响应形态（成功响应只应该有单对象 / 分页集合 / 无内容三种）。

## 为什么读接口不要求 `require_kb_write`

只读用户（viewer）必须能看到自己有权访问的库与文档——否则
"只读"变成"什么都看不到"。写权限在**每一个写端点**单独要求。

## 分页参数的依赖写法

统一用 `page: PageQuery = Depends(page_query)`（与其他路由一致）。
`PageQuery` 本身已是 `Depends(...)`，这里再套一层 `Depends` 是
FastAPI 的既有约定（`PageQuery` 作为类型别名），不影响行为——
但**不要**两处都不写 `Depends`，那会让 FastAPI 把它当请求体。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.deps import Principal, get_current_principal, get_db
from rag_api.deps.pagination import PageQuery, page_query
from rag_api.openapi import DELETE_ENDPOINT, TENANT_ENDPOINT, TENANT_WRITE_ENDPOINT
from rag_api.permissions import PermissionContext, get_permission_context
from rag_api.services.knowledge import KnowledgeBaseService, to_item
from rag_contracts.api.knowledge import (
    CreateKnowledgeBaseRequest,
    KnowledgeBaseItem,
    UpdateKnowledgeBaseRequest,
)
from rag_contracts.api.pagination import Page, PageMeta
from rag_core.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge"])


def _service(session: AsyncSession) -> KnowledgeBaseService:
    return KnowledgeBaseService(session)


@router.get(
    "",
    response_model=Page[KnowledgeBaseItem],
    summary="知识库列表",
    description=(
        "返回当前用户**可见**的知识库（`private` 库只有创建者能看到，"
        "`tenant` 库全租户可见）。\n\n"
        "`total` 默认不返回：`COUNT(*)` 在库多时很慢，"
        "无限滚动只需 `has_more`。"
    ),
    responses=TENANT_ENDPOINT,
)
async def list_knowledge_bases(
    principal: Principal = Depends(get_current_principal),
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
    page: PageQuery = Depends(page_query),
) -> Page[KnowledgeBaseItem]:
    result = await _service(session).list(
        ctx=ctx,
        limit=page.limit,
        offset=page.offset,
        include_total=page.include_total,
    )
    items = [to_item(kb) for kb in result.items]
    return Page(
        items=items,
        meta=PageMeta(
            limit=page.limit,
            offset=page.offset,
            total=result.total,
            # `has_more` 用"本页是否取满"判断：不满说明后面没有了。
            # 这比再查一次 COUNT 便宜，也是 offset 分页的标准做法
            # （代价是最后一页恰好取满时会多一次空请求）。
            has_more=len(items) == page.limit,
        ),
    )


@router.post(
    "",
    response_model=KnowledgeBaseItem,
    status_code=201,
    summary="创建知识库",
    description=(
        "创建后立即可用（无需异步处理）。\n\n"
        "**不支持 `visibility=department`**：组织架构表尚未建模，"
        "建出来的库将无人可见。\n\n"
        "**chunking 参数创建后不可改**：改了会让已入库的 chunk 与新配置"
        "不一致。要换配置请重传新版本。"
    ),
    responses=TENANT_WRITE_ENDPOINT,
)
async def create_knowledge_base(
    payload: CreateKnowledgeBaseRequest,
    response: Response,
    principal: Principal = Depends(get_current_principal),
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> KnowledgeBaseItem:
    # 写权限单独在这里要求：只读用户能看列表，但不能建库
    ctx.require_kb_write()

    kb = await _service(session).create(ctx=ctx, payload=payload)

    # Location 头给出新资源地址（201 的配套要求）。
    # 用相对路径而不是拼 base_url：反向代理/多域名部署时
    # 服务端不知道对外域名，用相对路径让客户端自己解析。
    response.headers["Location"] = f"/api/v1/knowledge-bases/{kb.id}"
    return to_item(kb)


@router.get(
    "/{kb_id}",
    response_model=KnowledgeBaseItem,
    summary="知识库详情",
    description="不可见或不存在都返回 404——不泄漏资源存在性。",
    responses=TENANT_ENDPOINT,
)
async def get_knowledge_base(
    kb_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> KnowledgeBaseItem:
    kb = await _service(session).get(ctx=ctx, kb_id=kb_id)
    return to_item(kb)


@router.patch(
    "/{kb_id}",
    response_model=KnowledgeBaseItem,
    summary="更新知识库",
    description=(
        "只能改 `name` / `description` / `confidentiality`。\n\n"
        "- 改 `visibility` 是权限变更，应走专门的授权接口并留审计；\n"
        "- 改 chunking 参数会让已入库 chunk 与新配置不一致。"
    ),
    responses=TENANT_WRITE_ENDPOINT,
)
async def update_knowledge_base(
    kb_id: uuid.UUID,
    payload: UpdateKnowledgeBaseRequest,
    principal: Principal = Depends(get_current_principal),
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> KnowledgeBaseItem:
    ctx.require_kb_write()
    kb = await _service(session).update(ctx=ctx, kb_id=kb_id, payload=payload)
    return to_item(kb)


@router.delete(
    "/{kb_id}",
    status_code=204,
    summary="删除知识库",
    description=(
        "软删除（状态置 archived），数据保留供审计与回滚。\n\n"
        "**库内还有文档时返回 409**：级联删除会牵扯 chunk、向量、blob 引用，"
        "任一处漏掉就是孤儿数据。请先删除文档。"
    ),
    responses=DELETE_ENDPOINT,
)
async def delete_knowledge_base(
    kb_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    ctx: PermissionContext = Depends(get_permission_context),
    session: AsyncSession = Depends(get_db),
) -> None:
    ctx.require_kb_write()
    await _service(session).archive(ctx=ctx, kb_id=kb_id)
