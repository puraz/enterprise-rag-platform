"""全局异常处理器：把各类异常统一成 `ErrorResponse` 契约。

## 为什么必须有这一层

`docs/PROMPT.md` 要求「统一响应与全局异常错误码」。没有它会出现：

- FastAPI 默认的 `{"detail": ...}` 与我们的契约不一致 → 前端要写两套解析；
- Pydantic 校验错误默认回显 `input` 字段 → **用户密码泄漏**（A2-4 已证明）；
- 未捕获异常把堆栈暴露给客户端 → 泄漏内部路径、SQL、依赖版本。

## 处理的三类异常

| 异常 | 状态码 | 处理 |
|---|---|---|
| `AppError` | 由分类推导 | 用它的 code/message/details |
| `RequestValidationError` | 422 | 转成字段级明细，**丢弃 input** |
| 其它 `Exception` | 500 | 只回 `INTERNAL.UNEXPECTED` + request_id，堆栈只进日志 |

## 为什么 500 不回详情

攻击者拿到堆栈就能知道：框架版本（据此找已知漏洞）、
表结构（SQL 报错）、文件路径（推断部署方式）。
这些信息对排障的帮助远小于对攻击者的帮助——
留 request_id 让运维去日志里查就够了。

## 日志级别按 severity 决定

在 `except` 块里按 `exc.severity` 选日志级别，而不是一律 `error`：
- `INFO`（404、422）每天成千上万次，记成 error 会让告警淹没在噪音里；
- `CRITICAL`（越权、密钥失效）必须能触发立即告警。

这是 `AppError.severity` 存在的意义（A2-4 设计）。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_contracts.errors import ErrorCode, ErrorResponse, Severity
from rag_core.errors import AppError, from_pydantic
from rag_core.logging import get_logger

log = get_logger(__name__)

_SEVERITY_TO_LOG_LEVEL = {
    Severity.INFO: "info",
    Severity.WARNING: "warning",
    Severity.ERROR: "error",
    Severity.CRITICAL: "critical",
}


def _request_id(request: Request) -> str:
    """从 scope 取中间件注入的 request_id。"""
    state = request.scope.get("state") or {}
    value = state.get("request_id")
    return str(value) if value else "unknown"


def _json(response: ErrorResponse, status_code: int) -> JSONResponse:
    """序列化统一错误响应。

    用 `model_dump(mode="json")` 而不是 `model_dump()`：
    后者会把 `ErrorCode` 枚举原样留下，`json.dumps` 才能处理
    （FastAPI 的 JSONResponse 用标准 json 库）。
    """
    return JSONResponse(status_code=status_code, content=response.model_dump(mode="json"))


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """处理所有业务异常。"""
    request_id = _request_id(request)
    level = _SEVERITY_TO_LOG_LEVEL.get(exc.severity, "error")

    # 不记录 exc 的 message 之外的任何内容：details 里可能有用户输入。
    # 真正需要排障时用 request_id 去查链路日志。
    getattr(log, level)(
        "app_error",
        code=str(exc.code),
        category=exc.category.value,
        severity=exc.severity.value,
        status=exc.http_status,
        retryable=exc.is_retryable,
        # 只在服务端日志里记路径与错误详情，前端只拿到 message
        path=request.url.path,
        method=request.method,
        error=str(exc),
    )

    return _json(exc.to_response(request_id), exc.http_status)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """处理请求校验错误。

    **关键**：不能直接用 `exc.errors()` —— 它包含 `input`（用户提交的原始值）
    和 `ctx`（含异常对象）。`from_pydantic` 会把它们剥掉（A2-4 实测证明
    密码框长度不足时密码会明文回显）。
    """
    request_id = _request_id(request)
    converted = from_pydantic(exc)

    log.info(
        "request_validation_failed",
        path=request.url.path,
        method=request.method,
        field_count=len(converted.fields),
        # 只记字段位置，不记用户提交的值
        field_locations=[field.location for field in converted.fields],
    )

    return _json(converted.to_response(request_id), converted.http_status)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """处理 Starlette/FastAPI 自身抛的 HTTPException。

    典型来源：路由不存在（404）、方法不允许（405）。
    这些不是我们的业务异常，但响应体必须符合契约，
    否则前端解析 404 时会崩（期望 JSON 却拿到 `{"detail": "Not Found"}`）。
    """
    request_id = _request_id(request)
    code_map = {
        404: ErrorCode.COMMON_NOT_FOUND,
        405: ErrorCode.COMMON_METHOD_NOT_ALLOWED,
    }
    code = code_map.get(exc.status_code, ErrorCode.COMMON_BAD_REQUEST)
    message = "资源不存在" if exc.status_code == 404 else str(exc.detail)

    log.info(
        "http_exception",
        path=request.url.path,
        status=exc.status_code,
        detail=str(exc.detail),
    )

    return _json(
        ErrorResponse.build(code=code, message=message, request_id=request_id),
        exc.status_code,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底：未预期异常。

    对外只回 `INTERNAL.UNEXPECTED` + request_id，
    **绝不回堆栈或异常消息**（可能含 SQL、路径、内部标识）。
    堆栈只写服务端日志，用 request_id 关联。
    """
    request_id = _request_id(request)

    log.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error_type=type(exc).__name__,
        error=str(exc),
        exc_info=True,
    )

    return _json(
        ErrorResponse.build(
            code=ErrorCode.INTERNAL_UNEXPECTED,
            message="服务内部错误，请提供 request_id 联系管理员",
            request_id=request_id,
        ),
        500,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """把处理器挂到应用上。

    顺序无关（FastAPI 按异常类型精确匹配），
    但**兜底的 `Exception` 必须注册**，否则未捕获异常会走
    Starlette 的默认行为（返回纯文本堆栈）。
    """
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
