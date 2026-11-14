"""错误分类：决定"要不要重试"和"算不算熔断失败"。

这是整个韧性层最容易被做错的地方，两个具体后果：

1. **把 4xx 算成熔断失败** → 攻击者（或客户端 bug）发一堆 401/422
   就能把熔断器打开，**正常用户全部被拒**。这是可被利用的故障放大。
2. **该重试的不重试 / 不该重试的狂重试** → 前者让偶发抖动变成用户可见
   错误；后者在真故障时把下游压力乘以 N 倍。

所以分类必须显式成一张表，不能靠 `except Exception: retry`。
"""

from __future__ import annotations

import asyncio
from enum import StrEnum

from rag_contracts.errors import ErrorCategory, ErrorCode
from rag_core.errors import AppError


class FailureKind(StrEnum):
    """失败的语义分类。"""

    RETRYABLE = "retryable"
    """瞬时故障：重试可能成功，且**计入熔断统计**。
    例：超时、连接被重置、供应商 5xx、我方限流。"""

    PERMANENT = "permanent"
    """调用方或业务状态导致的失败：重试无用，且**不计入熔断统计**。
    例：参数错误、未认证、无权限、配额耗尽、资源不存在。
    把它计入熔断会让客户端错误打挂服务。"""

    FATAL = "fatal"
    """我方基础设施故障：重试无用，但**必须计入熔断统计**。
    例：供应商密钥失效（我配错了）、返回结构不合法。
    这类错误重试一万次也一样，但要立刻熔断止损并告警。"""


# 按分类判定的规则表。新增错误分类时才需要改这里。
_RETRYABLE_CATEGORIES = frozenset(
    {
        ErrorCategory.DEPENDENCY,
        ErrorCategory.RATE_LIMIT,
    }
)

_PERMANENT_CATEGORIES = frozenset(
    {
        ErrorCategory.VALIDATION,
        ErrorCategory.AUTHENTICATION,
        ErrorCategory.AUTHORIZATION,
        ErrorCategory.NOT_FOUND,
        ErrorCategory.CONFLICT,
        ErrorCategory.QUOTA,
    }
)

# 分类为 DEPENDENCY 但**不该重试**的特例：配置错了，重试无意义。
# 用错误码而非异常类型判断，避免循环 import 与收敛问题。
_NON_RETRYABLE_CODES = frozenset(
    {
        ErrorCode.PROVIDER_AUTH_FAILED,
        ErrorCode.PROVIDER_INVALID_RESPONSE,
        ErrorCode.INTERNAL_CONFIG_ERROR,
    }
)

# 内置的可重试异常：网络层瞬时故障
_TRANSIENT_BUILTINS = (
    asyncio.TimeoutError,  # Python 3.11+ 与 TimeoutError 同义
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


def classify(exc: BaseException) -> FailureKind:
    """判定一次失败该如何处理。

    顺序很重要：先看显式错误码特例，再看分类规则，最后落到内置异常。
    反过来会让"供应商密钥失效"被当成依赖故障而狂重试。
    """
    if isinstance(exc, AppError):
        if exc.code in _NON_RETRYABLE_CODES:
            # 我方配置/协议问题：不重试，但计入熔断（否则会一直打下游）
            return FailureKind.FATAL
        if exc.category in _RETRYABLE_CATEGORIES:
            return FailureKind.RETRYABLE
        if exc.category in _PERMANENT_CATEGORIES:
            return FailureKind.PERMANENT
        # 剩余分类（INTERNAL）归为 FATAL
        return FailureKind.FATAL

    if isinstance(exc, _TRANSIENT_BUILTINS):
        return FailureKind.RETRYABLE

    # 未识别异常：不知道能不能重试 → 保守地不重试，但要计入熔断
    # 并在日志里留完整堆栈，因为这说明有未覆盖的错误路径。
    return FailureKind.FATAL
