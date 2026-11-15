"""统一异常体系：与 Web 框架无关的业务异常定义。

为什么 core 里的异常**不继承** HTTPException：
- worker 也要抛同样的业务错误（解析失败、配额超限），它没有 HTTP 概念；
- 一旦继承 HTTPException，worker 就得依赖 starlette，端边界被污染。

所以这里的异常只描述"发生了什么、谁的责任、要不要告警"，
**到 API 层才翻译成 HTTP 响应**（全局异常处理器）。

## 状态码由错误码与分类共同决定

优先级：
1. `AppError.status_override`（个别异常需要特殊状态码时显式覆盖）
2. `code -> 状态码` 的映射表（细粒度，按业务码精确指定）
3. `category -> 状态码`（兜底）

三层的原因：绝大多数异常用分类兜底就够了；个别码需要特殊状态
（比如 `AUTH_IDP_UNAVAILABLE` 应该是 503 而非 401），
用码级映射更精确且不改分类语义。
"""

from __future__ import annotations

from typing import ClassVar

from rag_contracts.errors import (
    ErrorCategory,
    ErrorCode,
    ErrorResponse,
    FieldError,
    Severity,
    code_name,
)

# 分类 → 默认 HTTP 状态码。这是最后的兜底层。
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

# 业务码 → 状态码的**精确覆盖**。只有与分类默认值不同时才需要登记。
_CODE_STATUS_OVERRIDES: dict[ErrorCode, int] = {
    # 供应商/IdP 不可用是依赖故障，503 让网关和前端知道"稍后重试"
    ErrorCode.AUTH_IDP_UNAVAILABLE: 503,
    # dev 登录未启用：404 更准确（该端点不存在，而不是身份有问题）
    ErrorCode.AUTH_DEV_DISABLED: 404,
    # 上下文过长是调用方问题，不是供应商故障
    ErrorCode.PROVIDER_CONTEXT_TOO_LONG: 400,
    # 存储/队列积压属于依赖故障
    ErrorCode.STORAGE_UNAVAILABLE: 503,
    ErrorCode.OUTBOX_BACKLOG: 503,
}


def default_status_for(code: ErrorCode) -> int:
    """按业务码推导 HTTP 状态码。

    **不能只按域号段推导**：0xxxx 通用域里既有 `VALIDATION_FAILED`（422）
    又有 `NOT_FOUND`（404），域号段无法区分。

    所以状态码的真相是「异常类的 `category`」：
    `AppError.http_status` 优先用 `status_override`，否则用 category 推导，
    只有需要特殊状态的少数码才在映射表里精确覆盖。

    这个函数保留给「只拿到码、没有异常实例」的场景（例如文档生成、
    前端码表对照），它用码级映射 + 域兜底做**尽力而为**的推导。
    """
    override = _CODE_STATUS_OVERRIDES.get(code)
    if override is not None:
        return override
    # 按域段兜底：只用于"没有异常实例"的近似推导，
    # 实际响应状态由 AppError.http_status 决定（category 更准确）。
    domain = code.domain
    return {
        0: 400,
        1: 401,
        2: 403,
        3: 404,
        4: 404,
        5: 400,
        6: 404,
        7: 402,
        8: 503,
        9: 500,
    }.get(domain, 500)


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
    # 覆盖推导出的状态码，仅在特别需要时使用
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
    def code_name(self) -> str:
        """语义名，供日志与审计使用。"""
        return code_name(int(self.code))

    @property
    def http_status(self) -> int:
        """对外 HTTP 状态码。

        优先级：`status_override` → 码级精确覆盖 → **category 兜底**。

        用 category 而非"域号段"兜底的原因：0xxxx 通用域里既有
        校验失败（422）又有资源不存在（404），域号段区分不了。
        category 是异常类显式声明的语义，准确得多。

        （实测踩过：`COMMON_NOT_FOUND` 的码是 3，按域号段推导成了 400，
         而不是正确的 404。）
        """
        if self.status_override is not None:
            return self.status_override
        code_status = _CODE_STATUS_OVERRIDES.get(self.code)
        if code_status is not None:
            return code_status
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
        return ErrorResponse.build(
            code=self.code,
            message=self.message,
            request_id=request_id,
            fields=self.fields,
            details=self.details,
        )

    def __str__(self) -> str:
        return f"[{self.code_name}({int(self.code)})] {self.message}"


# ============================================================ 通用


class ValidationFailedError(AppError):
    """入参校验失败。字段级明细放在 fields。"""

    code = ErrorCode.COMMON_VALIDATION_FAILED
    category = ErrorCategory.VALIDATION
    severity = Severity.INFO
    default_message = "请求参数校验失败"


class BadRequestError(AppError):
    """请求语义不合法（不是字段校验失败）。

    与 `ValidationFailedError` 的区别：
    - `ValidationFailedError`（422）：字段类型/长度/格式不对，有字段级明细；
    - `BadRequestError`（**400**）：请求本身矛盾或越界，比如"分片编号 0"
      "offset 超上限""分片大小与声明不符"。

    状态码用 400 而不是 422：码名是 BAD_REQUEST，返回 422 会让
    "码名与状态码对不上"，前端按状态码做分支时无法区分这两类。
    """

    code = ErrorCode.COMMON_BAD_REQUEST
    category = ErrorCategory.VALIDATION
    severity = Severity.INFO
    default_message = "请求不合法"
    status_override = 400


class NotFoundError(AppError):
    """资源不存在。

    越权访问**也**抛这个（而不是 403），避免攻击者通过状态码差异
    枚举出"哪些资源 ID 存在"。审计日志里记真实原因。
    """

    code = ErrorCode.COMMON_NOT_FOUND
    category = ErrorCategory.NOT_FOUND
    severity = Severity.INFO
    default_message = "资源不存在"


class ConflictError(AppError):
    """状态冲突：乐观锁失败、重复创建、状态机不允许的流转。"""

    code = ErrorCode.COMMON_CONFLICT
    category = ErrorCategory.CONFLICT
    severity = Severity.WARNING
    default_message = "资源状态已变化，请刷新后重试"


class IdempotencyConflictError(AppError):
    code = ErrorCode.COMMON_IDEMPOTENCY_CONFLICT
    category = ErrorCategory.CONFLICT
    severity = Severity.WARNING
    default_message = "同一幂等键的请求内容不一致"


class RateLimitedError(AppError):
    code = ErrorCode.COMMON_RATE_LIMITED
    category = ErrorCategory.RATE_LIMIT
    severity = Severity.WARNING
    default_message = "请求过于频繁，请稍后重试"


class PayloadTooLargeError(AppError):
    code = ErrorCode.COMMON_PAYLOAD_TOO_LARGE
    category = ErrorCategory.VALIDATION
    severity = Severity.INFO
    default_message = "请求内容过大"
    status_override = 413


class UnsupportedMediaTypeError(AppError):
    code = ErrorCode.COMMON_UNSUPPORTED_MEDIA_TYPE
    category = ErrorCategory.VALIDATION
    severity = Severity.INFO
    default_message = "不支持的内容类型"
    status_override = 415


# ======================================================== 认证 / 授权


class AuthenticationError(AppError):
    code = ErrorCode.AUTH_UNAUTHENTICATED
    category = ErrorCategory.AUTHENTICATION
    severity = Severity.INFO
    default_message = "未认证"


class TokenExpiredError(AuthenticationError):
    """token 过期。

    与"无效"分开：前端收到这个应该先尝试 refresh，而不是直接踢到登录页。
    """

    code = ErrorCode.AUTH_TOKEN_EXPIRED
    default_message = "登录已过期"


class TokenInvalidError(AuthenticationError):
    code = ErrorCode.AUTH_TOKEN_INVALID
    default_message = "凭证无效"


class TokenRevokedError(AuthenticationError):
    code = ErrorCode.AUTH_TOKEN_REVOKED
    default_message = "登录状态已失效，请重新登录"


class RefreshTokenReusedError(AuthenticationError):
    """refresh token 复用：判定凭据泄漏。

    severity=CRITICAL：意味着 token 已流出到攻击者手里。
    """

    code = ErrorCode.AUTH_REFRESH_REUSED
    severity = Severity.CRITICAL
    default_message = "检测到凭证异常，请重新登录"


class AccountDisabledError(AuthenticationError):
    code = ErrorCode.AUTH_ACCOUNT_DISABLED
    default_message = "账号已被禁用"


class IdentityProviderUnavailableError(AppError):
    code = ErrorCode.AUTH_IDP_UNAVAILABLE
    category = ErrorCategory.DEPENDENCY
    severity = Severity.ERROR
    default_message = "统一认证服务暂不可用，请稍后重试"


class PermissionDeniedError(AppError):
    code = ErrorCode.AUTHZ_FORBIDDEN
    category = ErrorCategory.AUTHORIZATION
    severity = Severity.WARNING
    default_message = "无权执行此操作"


class TenantMismatchError(PermissionDeniedError):
    """跨租户访问。

    severity=CRITICAL：正常流程不该出现，出现即越权尝试或 RLS 被绕过。
    """

    code = ErrorCode.AUTHZ_TENANT_MISMATCH
    severity = Severity.CRITICAL
    default_message = "无权访问该资源"


class ClassificationDeniedError(PermissionDeniedError):
    """密级不足。"""

    code = ErrorCode.AUTHZ_CLASSIFICATION_DENIED
    default_message = "无权访问该密级的内容"


# ============================================================ 配额


class QuotaExceededError(AppError):
    """配额用尽。

    402 而不是 429：429 的含义是"稍后再试"，而配额耗尽需要人工提额，
    重试毫无意义。
    """

    code = ErrorCode.QUOTA_EXCEEDED
    category = ErrorCategory.QUOTA
    severity = Severity.WARNING
    default_message = "配额已用尽"


class BudgetExhaustedError(AppError):
    code = ErrorCode.QUOTA_BUDGET_EXHAUSTED
    category = ErrorCategory.QUOTA
    severity = Severity.ERROR
    default_message = "本周期成本预算已用尽"


# ====================================================== 外部供应商


class ProviderError(AppError):
    """外部供应商错误基类。

    注意：details 里**绝不能放**供应商原始响应体 ——
    它可能回显请求里的 prompt（即用户文档内容）。
    """

    code = ErrorCode.PROVIDER_UNAVAILABLE
    category = ErrorCategory.DEPENDENCY
    severity = Severity.ERROR
    default_message = "模型服务暂不可用"


class ProviderTimeoutError(ProviderError):
    code = ErrorCode.PROVIDER_TIMEOUT
    default_message = "模型服务响应超时"


class ProviderRateLimitedError(ProviderError):
    code = ErrorCode.PROVIDER_RATE_LIMITED
    severity = Severity.WARNING
    default_message = "模型服务繁忙，请稍后重试"


class ProviderAuthError(ProviderError):
    """我方配置的供应商密钥失效。

    severity=CRITICAL：运维事故，全部租户受影响。对用户仍显示"不可用"。
    """

    code = ErrorCode.PROVIDER_AUTH_FAILED
    severity = Severity.CRITICAL
    default_message = "模型服务暂不可用"


class ProviderInvalidResponseError(ProviderError):
    code = ErrorCode.PROVIDER_INVALID_RESPONSE
    default_message = "模型服务返回异常"


class ContextTooLongError(ProviderError):
    code = ErrorCode.PROVIDER_CONTEXT_TOO_LONG
    category = ErrorCategory.VALIDATION
    default_message = "输入内容超出模型上下文长度"


# ============================================================ 内部


class DependencyUnavailableError(AppError):
    """自身依赖（PG / Redis / 存储）不可用。"""

    code = ErrorCode.INTERNAL_DEPENDENCY_UNAVAILABLE
    category = ErrorCategory.DEPENDENCY
    severity = Severity.ERROR
    default_message = "服务依赖不可用，请稍后重试"


class ConfigError(AppError):
    code = ErrorCode.INTERNAL_CONFIG_ERROR
    category = ErrorCategory.INTERNAL
    severity = Severity.CRITICAL
    default_message = "服务配置错误"


# ===================================================== 框架异常转换


def _field_errors_from_pydantic(exc: object) -> list[FieldError]:
    """把 Pydantic 校验错误转成可安全外发的字段明细。

    关键：**丢掉 `input` 和 `ctx`**。
    Pydantic 的 errors() 默认包含 input（用户提交的原值）和 ctx（含异常对象）。
    直接透出去会把用户刚输入的密码 / token 原样回显到响应体。
    """
    fields: list[FieldError] = []
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return fields

    for raw in errors():
        location = [str(part) if not isinstance(part, int) else part for part in raw.get("loc", ())]
        fields.append(
            FieldError(
                location=location,
                message=str(raw.get("msg", "校验失败")),
                type=str(raw.get("type", "unknown")),
            )
        )
    return fields


def from_pydantic(exc: object) -> ValidationFailedError:
    """把 Pydantic ValidationError 转成业务异常。

    FastAPI 默认的 422 处理器会直接返回 exc.errors()，其中包含 input
    字段 —— 这就是"密码回显"泄漏路径。
    """
    return ValidationFailedError(fields=_field_errors_from_pydantic(exc))


def is_expected_error(exc: BaseException) -> bool:
    """判断是否为预期内的业务错误（决定日志级别和是否告警）。"""
    return isinstance(exc, AppError) and exc.severity is Severity.INFO
