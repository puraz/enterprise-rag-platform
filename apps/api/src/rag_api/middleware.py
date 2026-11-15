"""HTTP 中间件：request_id、访问日志、异常到响应的转换。

## 顺序很重要

中间件的执行顺序是**栈式**的：先注册的最外层。
本项目需要：

    RequestContext（最外层：生成 request_id，之后所有日志都带上）
      └─ ExceptionHandler（把异常变成统一契约）
           └─ AccessLog（记录耗时与状态码）
                └─ 路由

为什么 RequestContext 必须在最外层：
如果异常处理器在它外面，那么**异常处理器自己打日志时还没有 request_id**，
用户报障给的 ID 就查不到那条错误日志。

为什么不用 FastAPI 的 `@app.middleware("http")`：
那会把所有逻辑挤在一个函数里，且异常处理位于 Starlette 的
`ExceptionMiddleware` 之外，捕获不到路由里抛的 `AppError`。
这里用纯 ASGI 中间件 + 显式异常处理器，边界更清晰。
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_core.logging import new_request_id, request_context

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware:
    """为每个请求建立日志上下文，并把 request_id 回写到响应头。

    为什么回写响应头：用户报障时能直接把这个值贴给客服，
    而不是让客服去猜时间段再翻日志。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 允许上游（网关/前端）传入 request_id，便于跨服务串联。
        # 但要校验长度，避免被塞入超长字符串污染日志。
        incoming = _header(scope, REQUEST_ID_HEADER)
        request_id = incoming if incoming and len(incoming) <= 64 else new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        # 用 request_context 而不是直接 bind：它保证退出时精确还原，
        # 避免上下文泄漏到同一事件循环里的其他请求
        with request_context(request_id=request_id):
            await self.app(scope, receive, send_with_header)


class AccessLogMiddleware:
    """结构化访问日志。

    只记录**必要**字段：方法、路径、状态码、耗时。
    **不记录** query string 与请求体：它们可能含 PII、文档内容、token。

    路径里的 ID 段（UUID）会被归一化（`/documents/<uuid>` → `/documents/{id}`），
    否则日志基数爆炸（每个文档 ID 一条不同的路径），
    指标聚合与告警完全失效。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        log = structlog.stdlib.get_logger("http.access")
        started = time.perf_counter()
        status_code = 500

        async def capture_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            log.info(
                "http_request",
                method=scope.get("method"),
                path=_normalize_path(scope.get("path", "")),
                status=status_code,
                elapsed_ms=round(elapsed_ms, 2),
            )


def _header(scope: Scope, name: str) -> str | None:
    """从 ASGI scope 读取请求头（小写匹配）。"""
    target = name.lower().encode()
    for key, value in scope.get("headers", []):
        if key == target:
            return str(value.decode("latin-1"))
    return None


_UUID_SEGMENT = re.compile(
    r"/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?=/|$)"
)
_NUMERIC_SEGMENT = re.compile(r"/\d+(?=/|$)")


def _normalize_path(path: str) -> str:
    """把路径里的 ID 段替换成占位符，避免日志/指标基数爆炸。

    `/api/v1/documents/3f2a.../versions/2` → `/api/v1/documents/{id}/versions/{n}`
    """
    path = _UUID_SEGMENT.sub("/{id}", path)
    return _NUMERIC_SEGMENT.sub("/{n}", path)


# 类型别名，便于装配函数签名可读
ASGIMiddleware = Callable[[ASGIApp], Awaitable[None] | ASGIApp]


class MetricsMiddleware:
    """把每个 HTTP 请求记成 Prometheus 指标。

    ## 必须在 AccessLog 里层（先算好状态码）

    Prometheus 的 Counter 需要 status，而 status 在响应开始时才知道。
    与 AccessLog 一样用 `capture_status` 包装 send。

    ## route label 必须用**路由模板**

    FastAPI 在 `scope["route"]` 里给出模板（`/api/v1/documents/{doc_id}`），
    用它而不是实际路径：实际路径的 UUID 段会让时间序列数量随数据量爆炸
    （十万文档 = 十万条 `http_requests_total` 序列，Prometheus 内存被打爆）。

    如果 `route` 不可用（404 之类没匹配到路由），退化为归一化路径。
    归一化只处理 UUID 与纯数字段，仍有基数风险——所以 404 的
    route 标注成 `__unmatched__` 而不是实际路径，彻底避免扫描器
    （`/.env`、`/wp-admin` 这类探测）把基数炸开。
    """

    def __init__(self, app: ASGIApp, metrics: object) -> None:
        self.app = app
        self._metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from rag_core.metrics import Metrics

        metrics: Metrics = self._metrics  # type: ignore[assignment]
        started = time.perf_counter()
        status_code = 500

        async def capture_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            method = str(scope.get("method", "UNKNOWN"))
            route = _route_label(scope)
            elapsed = time.perf_counter() - started

            metrics.http_requests.labels(method=method, route=route, status=str(status_code)).inc()
            metrics.http_request_duration.labels(method=method, route=route).observe(elapsed)


def _route_label(scope: Scope) -> str:
    """取路由模板作为 label。

    Starlette 在请求处理完成后会把命中的 `Route` 放进 `scope["route"]`。
    未命中（404）时没有 route，**统一标成 `__unmatched__`**——
    不用实际路径，因为扫描器会用随机路径探测，每个都成一个新序列。
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return str(path)
    return "__unmatched__"
