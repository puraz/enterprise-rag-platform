"""OpenAPI 响应声明：把错误响应与状态码真正写进契约。

## 这一步解决什么问题

FastAPI 默认只为**成功响应**生成 schema。`/api/v1/me` 在 OpenAPI 里
只有 `200`，前端从它生成 TS 类型后，**根本不知道失败时返回什么形状**
——这就是"前端不好对接"的根因。

修复方式：给每个路由显式声明 `responses={401: {...}, 403: {...}}`。
有两种写法：

| 写法 | 问题 |
|---|---|
| 每个路由手写 `responses=` | 几百个接口重复，且容易漏 |
| **响应组常量 + 批量装配**（选这个） | 一处定义，按接口类型引用 |

## 响应组的设计

错误响应不是"每个接口都一样"，而是按接口类型分几组：

- `AUTH_ERRORS`：需要登录（401）+ 校验失败（422）
- `TENANT_ERRORS`：AUTH + 无权限（403）+ 限流（429）
- `ADMIN_ERRORS`：TENANT + 平台角色不足（403 的另一种语义）
- `WRITE_ERRORS`：TENANT + 冲突（409）
- `INTERNAL_ERRORS`：500 兜底（所有接口都有）

每组都包含 `ErrorResponse` 作为 schema —— 前端生成的类型里
就能看到 `ErrorResponse`，从而写统一的错误处理。
"""

from __future__ import annotations

from typing import Any

from rag_contracts.errors import ErrorResponse

# OpenAPI 的 responses 结构：{status_code: {"model": ..., "description": ...}}
ResponseSpec = dict[int | str, dict[str, Any]]


def _error(code: int, description: str) -> dict[str, Any]:
    """构造单个错误响应声明。

    `model` 让 FastAPI 把 `ErrorResponse` 的 schema 写进
    `components/schemas`，前端生成类型时就能引用它。
    """
    return {"model": ErrorResponse, "description": description}


# ---- 响应组 ----
# 命名体现"什么样接口会遇到哪些错误"，而不是按状态码分组。

AUTH_ERRORS: ResponseSpec = {
    401: _error(401, "未认证或 token 无效（AUTH.*）"),
    422: _error(422, "请求参数校验失败（COMMON.VALIDATION_FAILED）"),
}

TENANT_ERRORS: ResponseSpec = {
    **AUTH_ERRORS,
    403: _error(403, "无权限（AUTHZ.*）或租户不匹配"),
    404: _error(404, "资源不存在（越权访问也返回此码，不泄漏资源存在性）"),
    429: _error(429, "请求过于频繁（COMMON.RATE_LIMITED），响应头含 Retry-After"),
}

WRITE_ERRORS: ResponseSpec = {
    **TENANT_ERRORS,
    409: _error(409, "状态冲突（COMMON.CONFLICT / *_STATUS_CONFLICT），需重新读取"),
}

ADMIN_ERRORS: ResponseSpec = {
    **TENANT_ERRORS,
    403: _error(403, "需要平台管理角色（AUTHZ.INSUFFICIENT_ROLE）"),
}

INTERNAL_ERRORS: ResponseSpec = {
    500: _error(500, "服务内部错误（INTERNAL.UNEXPECTED），仅返回 request_id"),
}

UPLOAD_ERRORS: ResponseSpec = {
    **WRITE_ERRORS,
    413: _error(413, "请求体过大（COMMON.PAYLOAD_TOO_LARGE / UPLOAD.FILE_TOO_LARGE）"),
    415: _error(415, "不支持的内容类型（COMMON.UNSUPPORTED_MEDIA_TYPE）"),
}


def with_internal(errors: ResponseSpec) -> ResponseSpec:
    """给任意响应组补上 500 兜底。

    所有接口都可能 500（未捕获异常），所以每一组最终都要合并它。
    单独做函数而不是在每个组里手写 500：避免某个组漏掉。
    """
    return {**errors, **INTERNAL_ERRORS}


# 常用组合（路由层直接引用）
PUBLIC_ENDPOINT: ResponseSpec = with_internal({422: _error(422, "参数校验失败")})
"""公开接口（健康检查、登录）：无认证要求，但仍可能 422/500。"""

AUTHENTICATED_ENDPOINT: ResponseSpec = with_internal(AUTH_ERRORS)
"""只要求登录、不涉及租户资源的接口（如 /me）。"""

TENANT_ENDPOINT: ResponseSpec = with_internal(TENANT_ERRORS)
"""租户内读接口（列表、详情）。"""

TENANT_WRITE_ENDPOINT: ResponseSpec = with_internal(WRITE_ERRORS)
"""租户内写接口（创建、更新、删除）。"""

ADMIN_ENDPOINT: ResponseSpec = with_internal(ADMIN_ERRORS)
"""管理面接口。"""

UPLOAD_ENDPOINT: ResponseSpec = with_internal(UPLOAD_ERRORS)
"""上传相关接口。"""
