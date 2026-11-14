"""统一异常体系：与 Web 框架无关的业务异常定义。

为什么 core 里的异常**不继承** HTTPException：
- worker 也要抛同样的业务错误（解析失败、配额超限），它没有 HTTP 概念；
- 一旦继承 HTTPException，worker 就得依赖 starlette，端边界被污染。

所以这里的异常只描述"发生了什么、谁的责任、要不要告警"，
**到 API 层才翻译成 HTTP 响应**（A4 的全局异常处理器）。

关键设计：`http_status` 由 `category` 推导出默认值，个别异常类可覆盖。
这样"分类→状态码"只有一张表，不用在每个 raise 处重复声明。
"""

from __future__ import annotations

from typing import Any, ClassVar

from rag_contracts.errors import ErrorCategory, ErrorCode, ErrorResponse, FieldError, Severity

# 分类 → 默认 HTTP 状态码。只有这张表能决定状态码。
_CATEGORY_STATUS: dict[ErrorCategory, int] = {
    ErrorCategory.VALIDATION: 422,
    ErrorCategory.AUTHENTICATION: 401,
    ErrorCategory.AUTHORIZATION: 403,
    ErrorCategory.NOT_FOUND: 404,
    ErrorCategory.CONFLICT: 409,
    ErrorCategory.RATE_LIMIT: 429,
    ErrorCategory.QUOTA: 402,
    ErrorCategory.DEPENDENCY: 503,
    ErrorCategory.INTERNAL: 500,
}


class AppError(Exception):
    """所有业务异常的基类。

    子类只需声明 `code` / `category` / `severity` / `default_message`，
    剩下的（状态码、日志级别、能否重试）都由基类推导。

    `details` 的设计边界：只放**可公开**的结构化信息。
    内部异常类型、SQL 片段、文件路径、用户输入原值一律不放 —— 这些
    会通过错误响应回到不可信的客户端。
    """

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_UNEXPECTED
    category: ClassVar[ErrorCategory] = ErrorCategory.INTERNAL
    severity: ClassVar[Severity] = Severity.ERROR
    default_message: ClassVar[str] = "服务内部错误"
    # 覆盖分类默认状态码，仅在同分类需要不同状态时使用（如 402 用于配额）
    status_override: ClassVar[int | None] = None

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, str | int | float | bool] | None = None,
        fields: list[FieldError] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details: dict[str, str | int | float | bool] = dict(details or {})
        self.fields: list[FieldError] = list(fields or [])
        super().__init__(self.message)

    @property
    def http_status(self) -> int:
        """对外 HTTP 状态码。"""
        if self.status_override is not None:
            return self.status_override
        return _CATEGORY_STATUS[self.category]

    @property
    def is_retryable(self) -> bool:
        """调用方是否值得重试。

        只有"依赖故障"和"限流"可重试；参数错误、权限不足、配额耗尽
        重试一万次也是同样结果，让调用方重试只会放大故障。
        """
        return self.category in (ErrorCategory.DEPENDENCY, ErrorCategory.RATE_LIMIT)

    def to_response(self, request_id: str) -> ErrorResponse:
        """转成对外契约。请求 ID 由调用方（中间件）注入。"""
        return ErrorResponse(
            code=self.code,
            message=self.message,
            request_id=request_id,
            fields=self.fields,
            details=self.details,
        )

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


# ============================================================ 通用


class ValidationFailedError(AppError):
    """入参校验失败。字段级明细放在 fields。"""

    code: ClassVar[ErrorCode] = ErrorCode.COMMON_VALIDATION_FAILED
    category: ClassVar[ErrorCategory] = ErrorCategory.VALIDATION
    severity: ClassVar[Severity] = Severity.INFO
    default_message: ClassVar[str] = "请求参数校验失败"


class BadRequestError(AppError):
    code: ClassVar[ErrorCode] = ErrorCode.COMMON_BAD_REQUEST
    category: ClassVar[ErrorCategory] = ErrorCategory.VALIDATION
    severity: ClassVar[Severity] = Severity.INFO
    default_message: ClassVar[str] = "请求不合法"


class NotFoundError(AppError):
    """资源不存在。

    越权访问**也**抛这个（而不是 403），避免攻击者通过状态码差异
    枚举出"哪些资源 ID 存在"。审计日志里记真实原因。
    """

    code: ClassVar[ErrorCode] = ErrorCode.COMMON_NOT_FOUND
    category: ClassVar[ErrorCategory] = ErrorCategory.NOT_FOUND
    severity: ClassVar[Severity] = Severity.INFO
    default_message: ClassVar[str] = "资源不存在"


class ConflictError(AppError):
    """状态冲突：乐观锁失败、重复创建、状态机不允许的流转。"""

    code: ClassVar[ErrorCode] = ErrorCode.COMMON_CONFLICT
    category: ClassVar[ErrorCategory] = ErrorCategory.CONFLICT
    severity: ClassVar[Severity] = Severity.WARNING
    default_message: ClassVar[str] = "资源状态已变化，请刷新后重试"


class IdempotencyConflictError(AppError):
    code: ClassVar[ErrorCode] = ErrorCode.COMMON_IDEMPOTENCY_CONFLICT
    category: ClassVar[ErrorCategory] = ErrorCategory.CONFLICT
    severity: ClassVar[Severity] = Severity.WARNING
    default_message: ClassVar[str] = "同一幂等键的请求内容不一致"


class RateLimitedError(AppError):
    code: ClassVar[ErrorCode] = ErrorCode.COMMON_RATE_LIMITED
    category: ClassVar[ErrorCategory] = ErrorCategory.RATE_LIMIT
    severity: ClassVar[Severity] = Severity.WARNING
    default_message: ClassVar[str] = "请求过于频繁，请稍后重试"


class PayloadTooLargeError(AppError):
    code: ClassVar[ErrorCode] = ErrorCode.COMMON_PAYLOAD_TOO_LARGE
    category: ClassVar[ErrorCategory] = ErrorCategory.VALIDATION
    severity: ClassVar[Severity] = Severity.INFO
    default_message: ClassVar[str] = "请求内容过大"
    status_override: ClassVar[int | None] = 413


class UnsupportedMediaTypeError(AppError):
    code: ClassVar[ErrorCode] = ErrorCode.COMMON_UNSUPPORTED_MEDIA_TYPE
    category: ClassVar[ErrorCategory] = ErrorCategory.VALIDATION
    severity: ClassVar[Severity] = Severity.INFO
    default_message: ClassVar[str] = "不支持的内容类型"
    status_override: ClassVar[int | None] = 415


# ======================================================== 认证 / 授权


class AuthenticationError(AppError):
    code: ClassVar[ErrorCode] = ErrorCode.AUTH_UNAUTHENTICATED
    category: ClassVar[ErrorCategory] = ErrorCategory.AUTHENTICATION
    severity: ClassVar[Severity] = Severity.INFO
    default_message: ClassVar[str] = "未认证"


class TokenExpiredError(AuthenticationError):
    """token 过期。

    与"无效"分开：前端收到这个应该先尝试 refresh，而不是直接踢到登录页。
    这个区分能让用户在长时间操作后无感续期。
    """

    code: ClassVar[ErrorCode] = ErrorCode.AUTH_TOKEN_EXPIRED
    default_message: ClassVar[str] = "登录已过期"


class TokenInvalidError(AuthenticationError):
    code: ClassVar[ErrorCode] = ErrorCode.AUTH_TOKEN_INVALID
    default_message: ClassVar[str] = "凭证无效"


class TokenRevokedError(AuthenticationError):
    code: ClassVar[ErrorCode] = ErrorCode.AUTH_TOKEN_REVOKED
    default_message: ClassVar[str] = "登录状态已失效，请重新登录"


class RefreshTokenReusedError(AuthenticationError):
    """refresh token 复用：判定凭据泄漏。

    严重级别提到 CRITICAL，因为这意味着 token 已经流出到攻击者手里。
    处理方式必须是作废整条会话链，而不是只拒绝这一次。
    """

    code: ClassVar[ErrorCode] = ErrorCode.AUTH_REFRESH_REUSED
    severity: ClassVar[Severity] = Severity.CRITICAL
    default_message: ClassVar[str] = "检测到凭证异常，请重新登录"


class AccountDisabledError(AuthenticationError):
    code: ClassVar[ErrorCode] = ErrorCode.AUTH_ACCOUNT_DISABLED
    default_message: ClassVar[str] = "账号已被禁用"


class IdentityProviderUnavailableError(AppError):
    code: ClassVar[ErrorCode] = ErrorCode.AUTH_IDP_UNAVAILABLE
    category: ClassVar[ErrorCategory] = ErrorCategory.DEPENDENCY
    severity: ClassVar[Severity] = Severity.ERROR
    default_message: ClassVar[str] = "统一认证服务暂不可用，请稍后重试"


class PermissionDeniedError(AppError):
    code: ClassVar[ErrorCode] = ErrorCode.AUTHZ_FORBIDDEN
    category: ClassVar[ErrorCategory] = ErrorCategory.AUTHORIZATION
    severity: ClassVar[Severity] = Severity.WARNING
    default_message: ClassVar[str] = "无权执行此操作"


class TenantMismatchError(PermissionDeniedError):
    """跨租户访问。

    severity=CRITICAL：正常业务流程里不该出现这种情况，
    出现了就是越权尝试或 RLS 被绕过。必须进安全告警。
    """

    code: ClassVar[ErrorCode] = ErrorCode.AUTHZ_TENANT_MISMATCH
    severity: ClassVar[Severity] = Severity.CRITICAL
    default_message: ClassVar[str] = "无权访问该资源"


class ClassificationDeniedError(PermissionDeniedError):
    """密级不足。"""

    code: ClassVar[ErrorCode] = ErrorCode.AUTHZ_CLASSIFICATION_DENIED
    default_message: ClassVar[str] = "无权访问该密级的内容"


# ============================================================ 配额


class QuotaExceededError(AppError):
    """配额用尽。

    402 而不是 429：429 的含义是"稍后再试"，而配额耗尽需要人工提额，
    重试毫无意义。前端据此显示"联系管理员"而不是"自动重试"。
    """

    code: ClassVar[ErrorCode] = ErrorCode.QUOTA_EXCEEDED
    category: ClassVar[ErrorCategory] = ErrorCategory.QUOTA
    severity: ClassVar[Severity] = Severity.WARNING
    default_message: ClassVar[str] = "配额已用尽"


class BudgetExhaustedError(AppError):
    code: ClassVar[ErrorCode] = ErrorCode.QUOTA_BUDGET_EXHAUSTED
    category: ClassVar[ErrorCategory] = ErrorCategory.QUOTA
    severity: ClassVar[Severity] = Severity.ERROR
    default_message: ClassVar[str] = "本周期成本预算已用尽"


# ====================================================== 外部供应商


class ProviderError(AppError):
    """外部供应商错误基类。

    注意：这类错误的 detail 里**绝不能放**供应商原始响应体 ——
    它可能回显请求里的 prompt（即用户文档内容），也会把内部
    endpoint / key 前缀泄给客户端。
    """

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_UNAVAILABLE
    category: ClassVar[ErrorCategory] = ErrorCategory.DEPENDENCY
    severity: ClassVar[Severity] = Severity.ERROR
    default_message: ClassVar[str] = "模型服务暂不可用"


class ProviderTimeoutError(ProviderError):
    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_TIMEOUT
    default_message: ClassVar[str] = "模型服务响应超时"


class ProviderRateLimitedError(ProviderError):
    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_RATE_LIMITED
    severity: ClassVar[Severity] = Severity.WARNING
    default_message: ClassVar[str] = "模型服务繁忙，请稍后重试"


class ProviderAuthError(ProviderError):
    """我方配置的供应商密钥失效。

    severity=CRITICAL：这是运维事故，全部租户受影响，必须立刻告警。
    对用户仍显示"服务不可用"，不暴露"我们的 key 配错了"。
    """

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_AUTH_FAILED
    severity: ClassVar[Severity] = Severity.CRITICAL
    default_message: ClassVar[str] = "模型服务暂不可用"


class ProviderInvalidResponseError(ProviderError):
    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_INVALID_RESPONSE
    default_message: ClassVar[str] = "模型服务返回异常"


# ============================================================ 内部


class DependencyUnavailableError(AppError):
    """自身依赖（PG / Redis / 存储）不可用。"""

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_DEPENDENCY_UNAVAILABLE
    category: ClassVar[ErrorCategory] = ErrorCategory.DEPENDENCY
    severity: ClassVar[Severity] = Severity.ERROR
    default_message: ClassVar[str] = "服务依赖不可用，请稍后重试"


class ConfigError(AppError):
    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_CONFIG_ERROR
    category: ClassVar[ErrorCategory] = ErrorCategory.INTERNAL
    severity: ClassVar[Severity] = Severity.CRITICAL
    default_message: ClassVar[str] = "服务配置错误"


# ===================================================== 框架异常转换


def _field_errors_from_pydantic(exc: Any) -> list[FieldError]:
    """把 Pydantic 校验错误转成可安全外发的字段明细。

    关键：**丢掉 `input` 和 `ctx`**。
    Pydantic 的 errors() 默认包含 input（用户提交的原值）和 ctx（含异常对象）。
    直接透出去会把用户刚输入的密码 / token 原样回显到响应体，
    而且会出现在前端日志和浏览器 devtools 里。

    loc 的第一段通常是 "body"/"query"/"path"，对前端有意义，保留。
    """
    fields: list[FieldError] = []
    for raw in exc.errors():
        location = [str(part) if not isinstance(part, int) else part for part in raw.get("loc", ())]
        fields.append(
            FieldError(
                location=location,
                message=str(raw.get("msg", "校验失败")),
                type=str(raw.get("type", "unknown")),
            )
        )
    return fields


def from_pydantic(exc: Any) -> ValidationFailedError:
    """把 Pydantic ValidationError 转成业务异常。

    FastAPI 默认的 422 处理器会直接返回 exc.errors()，其中包含 input
    字段 —— 这就是"密码回显"泄漏路径。我们自己接管异常处理时必须
    换成这条路，不要用默认行为。
    """
    return ValidationFailedError(fields=_field_errors_from_pydantic(exc))


def is_expected_error(exc: BaseException) -> bool:
    """判断是否为预期内的业务错误（用于决定日志级别和是否告警）。

    预期内 = 调用方或业务状态导致的拒绝，不是系统故障。
    把它们记成 ERROR 会让告警淹没在噪音里 —— 这是告警疲劳的根源。
    """
    return isinstance(exc, AppError) and exc.severity is Severity.INFO
