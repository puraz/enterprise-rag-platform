"""OpenAPI 契约守卫：确保前端能真正对接。

## 这些测试防的是什么

没有这些守卫，最常见的退化是：

1. **新路由漏写 `responses=`** → OpenAPI 里只有 200，
   前端生成的 TS 类型里没有错误分支，线上 401 时前端崩溃。
2. **成功响应没有 response_model** → OpenAPI 里该接口的 schema 是空对象，
   前端拿到 `unknown`，等于所有字段都要手写。
3. **分页结构不一致** → 有的接口返回 `{items}`、有的 `{data}`、有的裸数组，
   前端要为每个列表写一套解析。
4. **状态码语义错乱** → 创建返回 200（应为 201）、越权返回 403
   而资源不存在时返回 404（可被枚举）。

每条测试都对应一个具体的前端痛点。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from rag_api.main import create_app
from rag_contracts.errors import ErrorResponse
from rag_core.settings import Settings


def _test_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "env": "local",
        "dev_auth_enabled": True,
        "startup_dependency_check": False,
        "database": {"password": "x" * 32},
        "redis": {"password": "x" * 32},
        "identity": {"jwt_secret": "y" * 40},
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(scope="module")
def openapi_spec() -> dict[str, Any]:
    """取 app 生成的 OpenAPI 文档。"""
    app = create_app(_test_settings())
    with TestClient(app) as client:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        return response.json()


# ==================================================== 错误响应声明


def test_all_routes_declare_error_responses(openapi_spec: dict[str, Any]) -> None:
    """**每个业务路由都必须声明非 2xx 响应。**

    FastAPI 默认只生成 200。少了 401/403/422/500 的声明，
    前端从 OpenAPI 生成 TS 后就不知道失败时拿到什么形状——
    这正是"前端不好对接"的核心原因。
    """
    offenders: list[str] = []

    for path, operations in openapi_spec["paths"].items():
        # 健康检查允许只有 2xx + 503（不涉及认证）
        for method, operation in operations.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            codes = set(operation.get("responses", {}))
            has_client_error = any(str(c).startswith("4") or str(c) == "500" for c in codes)
            if not has_client_error:
                offenders.append(f"{method.upper()} {path}")

    assert not offenders, (
        "以下路由没有声明任何错误响应，前端无法生成错误类型：\n  "
        + "\n  ".join(offenders)
        + "\n请加 responses=PUBLIC_ENDPOINT / AUTHENTICATED_ENDPOINT / TENANT_ENDPOINT 等"
    )


# 不属于业务 API 契约的端点的**例外**，每个都要写明理由。
#
# 健康检查：运维探针端点，不是前端消费的业务接口。
# 它的 503 需要携带 `{checks: {postgres: "error", ...}}` 让运维一眼看出
# 哪个依赖挂了 —— 用统一的 ErrorResponse 会丢失这些诊断信息，
# 反而让"为什么没就绪"变成猜谜。
_NON_BUSINESS_PREFIXES = ("/health/",)


def _is_business_endpoint(path: str) -> bool:
    return not any(path.startswith(prefix) for prefix in _NON_BUSINESS_PREFIXES)


def test_error_responses_use_error_response_schema(openapi_spec: dict[str, Any]) -> None:
    """所有**业务接口**的 4xx/5xx 声明的 schema 都必须是 ErrorResponse。

    如果某个路由手写了别的形状（比如 `{"detail": str}`），
    前端就要为它单独写一套错误处理——契约就分裂了。

    健康检查是显式例外（见 `_NON_BUSINESS_PREFIXES` 的说明）。
    """
    offenders: list[str] = []

    for path, operations in openapi_spec["paths"].items():
        if not _is_business_endpoint(path):
            continue
        for method, operation in operations.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            for code, response in operation.get("responses", {}).items():
                if not str(code).startswith(("4", "5")):
                    continue
                schema = response.get("content", {}).get("application/json", {}).get("schema", {})
                ref = schema.get("$ref", "")
                if not ref.endswith("/ErrorResponse"):
                    offenders.append(f"{method.upper()} {path} {code} → {ref or schema}")

    assert not offenders, "以下错误响应的 schema 不是 ErrorResponse：\n  " + "\n  ".join(offenders)


def test_error_response_is_in_components(openapi_spec: dict[str, Any]) -> None:
    """`ErrorResponse` 必须在 components.schemas 里。

    前端生成 TS 时从这里拿类型。缺了它，所有错误声明的 ref 都会悬空。
    """
    schemas = openapi_spec.get("components", {}).get("schemas", {})
    assert "ErrorResponse" in schemas
    # ErrorCode 是 IntEnum，OpenAPI 里会内联到 ErrorResponse 的 code 字段
    # （不作为独立 schema），但契约包里必须导出它供 TS 使用
    assert "code_name" in schemas["ErrorResponse"]["properties"], (
        "ErrorResponse 必须同时提供 code（数字）与 code_name（语义名）"
    )


# ==================================================== 成功响应模型


def test_all_routes_declare_response_model() -> None:
    """**每个路由都必须有 response_model 或明确的 204。**

    没有 response_model 时，OpenAPI 里该接口的 200 schema 是空对象，
    前端生成出 `unknown` —— 所有字段都要手写，等于没有类型。
    """
    app = create_app(_test_settings())
    offenders: list[str] = []

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        # 健康检查等少数端点允许自定义 Response
        if route.response_class is not None:
            continue
        has_model = route.response_model is not None
        # 204 表示无内容，允许无 response_model
        allows_204 = 204 in (route.status_code,) or 204 in getattr(route, "responses", {})
        if not has_model and not allows_204:
            offenders.append(f"{sorted(route.methods or [])} {route.path}")

    assert not offenders, "以下路由缺少 response_model：\n  " + "\n  ".join(offenders)


# ==================================================== 分页契约


def test_pagination_contract_shape() -> None:
    """分页响应结构固定为 `{items, meta}`。

    不用裸数组（无法扩展）、不叫 `data`/`list`（前端要记不同名字）。
    """
    from rag_contracts.api.pagination import Page, PageMeta

    meta = PageMeta(limit=20, offset=0, total=None, has_more=True)
    page = Page[int](items=[1, 2, 3], meta=meta)

    dumped = page.model_dump()
    assert set(dumped) == {"items", "meta"}
    assert set(dumped["meta"]) == {"limit", "offset", "total", "has_more"}


def test_pagination_total_optional_by_default() -> None:
    """默认不返回 total —— 避免每次列表都触发 COUNT(*)。

    这条守的是性能：`COUNT(*)` 在百万级表上是全表扫描或索引扫描，
    加上它会让所有列表接口慢一个量级。
    """
    from rag_contracts.api.pagination import PageMeta

    meta = PageMeta(limit=20, offset=0, has_more=False)
    assert meta.total is None


def test_page_query_rejects_excessive_offset() -> None:
    """offset 超过上限必须 400，引导前端用过滤条件。

    不拦的话，`offset=1000000` 会让 PG 扫描并丢弃百万行——
    单次请求就可能几十秒，且会占满连接。
    """
    from rag_api.deps.pagination import page_query
    from rag_contracts.api.pagination import MAX_OFFSET
    from rag_core.errors import BadRequestError

    # 合法值
    request = page_query(limit=20, offset=0, include_total=False)
    assert request.limit == 20

    with pytest.raises(BadRequestError, match="offset 不能超过"):
        page_query(limit=20, offset=MAX_OFFSET + 1, include_total=False)


def test_page_query_enforces_limit_cap() -> None:
    """limit 上限 100（由 Query 的 le=100 校验，这里验证契约常量一致）。"""
    from rag_contracts.api.pagination import PageRequest

    # Pydantic 层同样有约束，防止绕过 FastAPI 直接构造
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        PageRequest(limit=1000, offset=0, include_total=False)


# ==================================================== 状态码语义


def test_error_response_carries_request_id(openapi_spec: dict[str, Any]) -> None:
    """ErrorResponse 的 schema 必须包含 request_id 字段。

    request_id 是用户报障时唯一能关联到日志的线索，
    不能是可选的。
    """
    schema = openapi_spec["components"]["schemas"]["ErrorResponse"]
    assert "request_id" in schema["properties"]
    assert "request_id" in schema.get("required", [])
    # code 必须是整数（前端做 switch 与号段判断）
    assert schema["properties"]["code"]["type"] == "integer"
    # code_name 必须是必填字符串（日志与审计依赖它）
    assert "code_name" in schema.get("required", [])


def test_response_groups_cover_common_cases() -> None:
    """响应组必须覆盖 401/403/404/409/422/429/500 这些常见状态码。

    缺哪个，对应场景的错误在 OpenAPI 里就不可见。
    """
    from rag_api.openapi import (
        ADMIN_ENDPOINT,
        AUTHENTICATED_ENDPOINT,
        PUBLIC_ENDPOINT,
        TENANT_ENDPOINT,
        TENANT_WRITE_ENDPOINT,
        UPLOAD_ENDPOINT,
    )

    assert 422 in PUBLIC_ENDPOINT
    assert {401, 422, 500} <= set(AUTHENTICATED_ENDPOINT)
    assert {401, 403, 404, 429, 500} <= set(TENANT_ENDPOINT)
    assert 409 in TENANT_WRITE_ENDPOINT
    assert 403 in ADMIN_ENDPOINT
    assert 413 in UPLOAD_ENDPOINT


def test_error_response_model_matches_handler_output() -> None:
    """处理器实际输出的字段必须与契约模型一致。

    如果处理器多返回字段（比如错误地带上堆栈），契约就失准了。
    """
    from rag_core.errors import NotFoundError

    response = NotFoundError("文档不存在").to_response("req-1")
    assert isinstance(response, ErrorResponse)
    assert set(response.model_dump()) == {
        "code",
        "code_name",
        "message",
        "request_id",
        "fields",
        "details",
    }
