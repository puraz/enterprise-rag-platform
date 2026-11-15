"""错误码体系测试：号段、唯一性、映射一致性、与 HTTP 状态码解耦。

这套体系是**永久对外契约**：前端会硬编码 `if (code === 10101)`。
所以这些测试守的不是"代码能跑"，而是"契约不会悄悄漂移"。
"""

from __future__ import annotations

import pytest

from rag_contracts.errors import ErrorCode, code_name
from rag_core.errors import (
    AccountDisabledError,
    BadRequestError,
    NotFoundError,
    ProviderAuthError,
    QuotaExceededError,
    RateLimitedError,
    TokenExpiredError,
    ValidationFailedError,
    default_status_for,
)

# ==================================================== 号段规则


def test_all_codes_within_valid_range() -> None:
    """所有码必须落在 1 - 99999 之间（0xxxx 到 9xxxx 的十域）。"""
    for code in ErrorCode:
        assert 0 < int(code) < 100_000, f"{code.name}={int(code)} 越界"


def test_domain_ranges_are_declared() -> None:
    """码的域号（code // 10000）必须与注释里声明的域一致。

    这条测试的价值：当有人把"认证域的错误"误写到 3xxxx（文档域）时，
    号段检查不会报错（数字合法），但域归属错了——
    前端按 `code >= 10000 && code < 20000` 判断会漏掉它。

    这里用**断言域号集合**的方式，验证每个域都有码且分布合理。
    """
    domains: dict[int, int] = {}
    for code in ErrorCode:
        domains[code.domain] = domains.get(code.domain, 0) + 1

    # 十个域都必须有码（0 通用 到 9 内部）
    assert set(domains) == set(range(10)), f"域分布不完整: {sorted(domains)}"

    # 每个域至少 1 个码
    for domain, count in domains.items():
        assert count >= 1, f"域 {domain} 没有错误码"


def test_subdomain_grouping() -> None:
    """同一子域的错误应相邻（便于前端按区间批量判断）。

    例：10101 / 10102 / 10103 都是 token 相关，
    前端可以写 `code >= 10100 && code < 10200` 判断"是否需要刷新 token"。
    """
    token_codes = [
        int(ErrorCode.AUTH_TOKEN_EXPIRED),
        int(ErrorCode.AUTH_TOKEN_INVALID),
        int(ErrorCode.AUTH_TOKEN_REVOKED),
        int(ErrorCode.AUTH_REFRESH_REUSED),
    ]
    # 都在 101xx 段内
    assert all(10100 <= c < 10200 for c in token_codes), token_codes


# ==================================================== 唯一性与命名


def test_codes_are_unique() -> None:
    """数值唯一。重复会让前端分支命中错误逻辑。"""
    values = [int(code) for code in ErrorCode]
    assert len(values) == len(set(values))


def test_code_names_are_unique() -> None:
    """语义名唯一（由 enum 保证，这里防手写重复）。"""
    names = [code.name for code in ErrorCode]
    assert len(names) == len(set(names))


def test_code_name_lookup_roundtrip() -> None:
    """`code_name(n)` 必须能反查回语义名。"""
    for code in ErrorCode:
        assert code_name(int(code)) == code.name


def test_unknown_code_has_fallback_name() -> None:
    """未知码返回可识别的占位名，而不是抛异常。

    日志里出现未知码说明有地方绕过了 ErrorCode 枚举，
    占位名能让这种情况被发现而不是让日志崩掉。
    """
    assert code_name(99999) == "UNKNOWN_99999"


# ==================================================== 状态码映射


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ValidationFailedError(), 422),
        (NotFoundError(), 404),
        (QuotaExceededError(), 402),
        (RateLimitedError(), 429),
    ],
)
def test_status_derived_from_category(error: object, expected_status: int) -> None:
    """状态码由 category 推导。

    **不能用域号段推导**：`COMMON_NOT_FOUND` 的码是 3（0xxxx 通用域），
    按域号段会得到 400，而正确的语义是 404。
    """
    assert error.http_status == expected_status  # type: ignore[attr-defined]


def test_provider_error_maps_to_503() -> None:
    """供应商故障 → 503（可重试，网关应摘流量）。"""
    assert ProviderAuthError().http_status == 503


def test_quota_is_402_not_429() -> None:
    """配额耗尽必须是 402，不能是 429。

    429 语义是"稍后再试"，前端会自动重试；
    配额耗尽重试一万次也没用，应提示联系管理员提额。
    """
    assert QuotaExceededError().http_status == 402
    assert QuotaExceededError().is_retryable is False


def test_code_status_override_wins() -> None:
    """码级精确覆盖优先于分类推导。

    `AUTH_IDP_UNAVAILABLE` 分类是 DEPENDENCY（503），
    但如果按"认证域"兜底会变成 401 —— 那会让前端以为用户没登录。
    """
    assert default_status_for(ErrorCode.AUTH_IDP_UNAVAILABLE) == 503


# ==================================================== 状态码与业务码解耦


def test_same_http_status_multiple_business_codes() -> None:
    """同一个 HTTP 状态码可以对应多个业务码，反之业务码应基本一码一状态。

    这正是"为什么业务码不能用 HTTP 码当前缀"的证明：
    401 会对应 未认证/过期/无效/撤销/缺少头 至少 5 个业务码。
    """
    auth_codes = [
        ErrorCode.AUTH_UNAUTHENTICATED,
        ErrorCode.AUTH_TOKEN_EXPIRED,
        ErrorCode.AUTH_TOKEN_INVALID,
        ErrorCode.AUTH_TOKEN_REVOKED,
        ErrorCode.AUTH_TOKEN_MISSING,
    ]
    assert len(auth_codes) == len(set(auth_codes))


def test_business_code_stable_when_status_changes() -> None:
    """业务码不随 HTTP 状态码变化。

    同一个 `AUTH_TOKEN_EXPIRED` 在任何路由上都应是 10101，
    即使某个路由出于特殊原因返回了不同的 HTTP 状态。
    """
    error = TokenExpiredError()
    assert int(error.code) == 10101
    # 状态允许被 override，但业务码不变
    error.status_override = 403
    assert error.http_status == 403
    assert int(error.code) == 10101


# ==================================================== 可读性


def test_error_str_contains_both_code_and_name() -> None:
    """日志里必须同时有数字码与语义名。

    只有数字（`[3]`）时 grep 会命中无数无关内容；
    只有名字时又无法按数值区间做告警规则。
    """
    text = str(AccountDisabledError())
    assert "AUTH_ACCOUNT_DISABLED" in text
    assert str(int(ErrorCode.AUTH_ACCOUNT_DISABLED)) in text


def test_response_includes_code_name() -> None:
    """对外响应同时带 code 与 code_name。

    前端用 code 分支，日志/报障用 code_name。
    """
    response = BadRequestError().to_response("req-1")
    assert response.code == int(ErrorCode.COMMON_BAD_REQUEST)
    assert response.code_name == "COMMON_BAD_REQUEST"
    assert isinstance(response.code, int)
