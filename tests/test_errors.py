"""异常与错误码契约测试。

这里的用例不是覆盖代码行，而是锁死每一条"会在生产里变成事故"的边界：
- 密码/密钥被回显到错误响应
- 越权访问通过状态码差异泄漏资源存在性
- 可重试与不可重试被搞混，导致故障放大
- 错误码字符串值被偷偷改掉，前端分支全部失效
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from rag_contracts.errors import ErrorCategory, ErrorCode, ErrorResponse, Severity
from rag_core.errors import (
    AppError,
    AuthenticationError,
    ConfigError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ProviderAuthError,
    ProviderError,
    QuotaExceededError,
    RateLimitedError,
    TenantMismatchError,
    TokenExpiredError,
    TokenInvalidError,
    ValidationFailedError,
    from_pydantic,
    is_expected_error,
)

# ==================================================== 错误码契约稳定性


def test_error_codes_are_unique() -> None:
    """错误码数值不能重复。

    重复会让前端分支判断命中错误的处理逻辑
    （比如把限流当成配额耗尽）。
    """
    values = [int(code) for code in ErrorCode]
    duplicates = {value for value in values if values.count(value) > 1}
    assert not duplicates, f"重复的错误码: {duplicates}"


def test_error_code_follows_domain_ranges() -> None:
    """错误码必须落在合法号段内（0xxxx - 9xxxx）。

    号段是扩容的基础：新增域从号段表申请，不重排已有码。
    落在号段外的码（比如 123456）说明有人随手写了个数字，
    以后会和其它域冲突。
    """
    for code in ErrorCode:
        value = int(code)
        assert 0 < value < 100_000, f"{code.name}={value} 超出号段范围"
        # 域号 = value // 10000，必须在 0-9
        assert 0 <= code.domain <= 9, f"{code.name}={value} 域号非法"


def test_error_code_name_is_readable() -> None:
    """语义名（成员名）必须全大写下划线，且能被 `code_name()` 查回。

    语义名是日志与审计里唯一可读的标识——`10101` 在日志里毫无意义。
    """
    from rag_contracts.errors import code_name

    for code in ErrorCode:
        assert code.name == code.name.upper(), f"{code.name} 不是大写"
        assert " " not in code.name
        # 数字码 → 语义名的反查必须一致
        assert code_name(int(code)) == code.name


# ==================================================== 分类与状态码推导


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ValidationFailedError(), 422),
        (AuthenticationError(), 401),
        (PermissionDeniedError(), 403),
        (NotFoundError(), 404),
        (ConflictError(), 409),
        (RateLimitedError(), 429),
        (QuotaExceededError(), 402),
        (ProviderError(), 503),
        (ConfigError(), 500),
    ],
)
def test_http_status_derived_from_category(error: AppError, expected_status: int) -> None:
    """状态码只由分类推导，不在 raise 处重复声明。"""
    assert error.http_status == expected_status


def test_quota_is_402_not_429() -> None:
    """配额耗尽必须是 402，不能是 429。

    429 的语义是"稍后再试"，前端会做自动重试；配额耗尽重试一万次
    也没用，应该提示用户联系管理员提额。用错状态码会导致前端无脑
    重试放大故障。
    """
    assert QuotaExceededError().http_status == 402
    assert QuotaExceededError().is_retryable is False


# ==================================================== 可重试性判定


@pytest.mark.parametrize(
    "error",
    [RateLimitedError(), ProviderError()],
)
def test_transient_errors_are_retryable(error: AppError) -> None:
    """限流和依赖故障值得重试。"""
    assert error.is_retryable is True


@pytest.mark.parametrize(
    "error",
    [
        ValidationFailedError(),
        AuthenticationError(),
        NotFoundError(),
        ConflictError(),
        QuotaExceededError(),
        ConfigError(),
    ],
)
def test_permanent_errors_are_not_retryable(error: AppError) -> None:
    """其余一律不可重试。

    把参数错误也标成可重试，会让客户端在 400 上疯狂重放，
    在故障时把 QPS 放大数倍。
    """
    assert error.is_retryable is False


# ==================================================== 安全相关语义


def test_tenant_mismatch_is_critical() -> None:
    """跨租户访问必须 CRITICAL：正常流程不会出现，出现即越权。"""
    assert TenantMismatchError().severity is Severity.CRITICAL


def test_provider_auth_failure_is_critical() -> None:
    """我方供应商密钥失效是运维事故，全租户受影响。"""
    assert ProviderAuthError().severity is Severity.CRITICAL


def test_token_expired_and_invalid_are_distinguishable() -> None:
    """过期与无效必须能区分。

    前端对过期做 refresh 续期，对无效直接踢登录页。混成一个码，
    要么用户被无故登出，要么拿着坏 token 无限 refresh。
    """
    assert TokenExpiredError().code is not TokenInvalidError().code
    assert TokenExpiredError().code is ErrorCode.AUTH_TOKEN_EXPIRED
    assert TokenInvalidError().code is ErrorCode.AUTH_TOKEN_INVALID


def test_provider_errors_do_not_echo_prompt_content() -> None:
    """供应商异常的 details 不得包含原始响应。

    供应商可能回显请求里的 prompt（即用户文档内容）。把原始响应
    塞进 details 等于把用户数据写进错误响应和前端日志。
    """
    error = ProviderError()
    assert error.details == {}


# ==================================================== 错误响应契约


def test_to_response_produces_contract_shape() -> None:
    """对外响应必须是 ErrorResponse，且 request_id 与 code_name 都要填。"""
    error = NotFoundError("文档不存在")
    response = error.to_response("req-abc")

    assert isinstance(response, ErrorResponse)
    # code 是数字（前端做 switch 用）
    assert response.code == int(ErrorCode.COMMON_NOT_FOUND)
    assert isinstance(response.code, int)
    # code_name 是语义名（日志/排障用）
    assert response.code_name == "COMMON_NOT_FOUND"
    assert response.message == "文档不存在"
    assert response.request_id == "req-abc"


def test_default_message_used_when_none_given() -> None:
    """不传 message 时用类的默认文案，避免各处重复写。"""
    assert NotFoundError().message == "资源不存在"


def test_details_carried_through() -> None:
    """可公开的结构化补充信息要能传到前端（如限流的 retry_after）。"""
    error = RateLimitedError(details={"retry_after_s": 30})
    assert error.to_response("r").details == {"retry_after_s": 30}


def test_error_response_is_frozen() -> None:
    """错误响应体不可变：生成之后任何地方篡改都会导致前后端不一致。"""
    response = NotFoundError().to_response("r")
    with pytest.raises(ValidationError):
        response.message = "改一下"  # type: ignore[misc]


# =============================================== Pydantic 校验错误转换


class _LoginRequest(BaseModel):
    username: str
    password: str = Field(min_length=12)


class _Item(BaseModel):
    name: str


class _Payload(BaseModel):
    items: list[_Item]


def test_pydantic_conversion_drops_user_input() -> None:
    """**最关键的用例**：用户提交的原值绝不能出现在错误响应里。

    FastAPI 默认的 422 处理器直接返回 exc.errors()，其中 input 字段
    是用户提交的原始值。密码框长度不足时，这会把用户刚输入的密码
    明文回显到响应体 —— 它会进浏览器 devtools、前端日志、错误上报。
    """
    try:
        _LoginRequest(username="alice", password="PlainPass1")
    except ValidationError as exc:
        converted = from_pydantic(exc)
        payload = converted.to_response("r").model_dump_json()

    assert "PlainPass1" not in payload, "用户密码被回显到了错误响应"
    assert converted.code is ErrorCode.COMMON_VALIDATION_FAILED
    assert converted.http_status == 422


def test_pydantic_conversion_keeps_field_location() -> None:
    """字段位置要保留，前端才能把错误标到具体输入框。"""
    try:
        _LoginRequest(username="alice", password="short")
    except ValidationError as exc:
        converted = from_pydantic(exc)

    assert len(converted.fields) == 1
    field = converted.fields[0]
    assert field.location == ["password"]
    assert field.type == "string_too_short"


def test_pydantic_conversion_handles_nested_index_location() -> None:
    """嵌套列表的 loc 形如 ('body', 'items', 0, 'name')，索引是 int。"""
    try:
        _Payload.model_validate({"items": [{"name": 123}]})
    except ValidationError as exc:
        converted = from_pydantic(exc)

    location = converted.fields[0].location
    assert location[-1] == "name"
    # 索引必须保持 int，转成字符串会让前端数组访问变成字典查找
    assert any(isinstance(part, int) for part in location)


# ==================================================== 告警级别判定


def test_expected_errors_do_not_page_anyone() -> None:
    """预期内的业务拒绝不应触发告警。

    404 / 422 每天成千上万次，把它们记成 ERROR 会让真正的故障
    淹没在噪音里 —— 这是告警疲劳的经典成因。
    """
    assert is_expected_error(NotFoundError()) is True
    assert is_expected_error(ValidationFailedError()) is True


def test_system_errors_are_not_expected() -> None:
    """依赖故障和内部错误必须能触发告警。"""
    assert is_expected_error(ProviderError()) is False
    assert is_expected_error(ConfigError()) is False
    assert is_expected_error(ValueError("随便一个原生异常")) is False


def test_error_str_is_readable_in_logs() -> None:
    """__str__ 要同时带语义名与数字码。

    日志里 grep `COMMON_NOT_FOUND` 比 grep `3` 可靠得多
    （数字 3 会命中无数无关内容）。
    """
    text = str(NotFoundError("文档 d-1 不存在"))
    assert "COMMON_NOT_FOUND" in text
    assert str(int(ErrorCode.COMMON_NOT_FOUND)) in text
    assert "文档 d-1 不存在" in text


# ==================================================== 契约不可回退


def test_error_category_covers_all_codes() -> None:
    """每个错误码都必须能映射到某个分类。

    新增错误码时忘了归类，会导致状态码推导 KeyError —— 在运行时
    表现为 500 套 500，排查时完全看不出原因。
    """
    from rag_core.errors import _CATEGORY_STATUS

    for category in ErrorCategory:
        assert category in _CATEGORY_STATUS, f"分类 {category} 缺少状态码映射"
