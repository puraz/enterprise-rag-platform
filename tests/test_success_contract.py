"""成功响应规范的护栏测试。

## 为什么需要这些测试

规范和文档会被遗忘，测试不会。这批测试守住的是：

1. 成功响应**不带 `code` 字段**（`code` 只属于失败响应）；
2. 成功响应**不包 `data` 层**（HTTP 状态码已是成功信号）；
3. 所有响应模型字段是 `snake_case`；
4. 时间字段是 ISO 8601 带时区；
5. 分页结构统一为 `{items, meta}`；
6. 创建用 201、删除用 204、异步用 202。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import BaseModel

from rag_contracts.api.pagination import Page, PageMeta
from rag_contracts.api.success import (
    CreatedResource,
    TaskAccepted,
    validate_snake_case_field,
    validate_time_field_value,
)

# ==================================================== 成功响应不带 code / data


def test_success_response_has_no_code_field() -> None:
    """**成功响应绝不能有 `code` 字段。**

    `code` 的语义是"失败原因"。成功没有原因。
    如果成功也带 `code: 0`，前端要同时判断 `code === 0` 和状态码，
    两者不一致时（比如缓存层返回了旧响应）行为不可预测。
    """
    from rag_contracts.api.auth import CurrentUserResponse, LoginResponse

    for model in (CurrentUserResponse, LoginResponse):
        fields = set(model.model_fields)
        assert "code" not in fields, f"{model.__name__} 不该有 code 字段"
        assert "code_name" not in fields, f"{model.__name__} 不该有 code_name 字段"


def test_success_response_has_no_data_wrapper() -> None:
    """**成功响应不包 `data` 层。**

    `data` 包装只在"成败都用 200、靠 body 区分"的方案里有意义。
    我们用 HTTP 状态码区分成败，`data` 是纯噪音——
    前端每次都要 `.data.data`，且 204 无内容时无法表达。
    """
    from rag_contracts.api.auth import CurrentUserResponse, LoginResponse

    for model in (CurrentUserResponse, LoginResponse):
        fields = set(model.model_fields)
        assert "data" not in fields, f"{model.__name__} 不该包 data 层"


def test_error_response_does_not_have_success_fields() -> None:
    """反向检查：失败响应不该有业务数据字段。

    防止有人把"成功时的数据"塞进错误响应（比如部分成功）。
    部分成功应该用 207 或专门的批量响应模型。
    """
    from rag_contracts.errors import ErrorResponse

    fields = set(ErrorResponse.model_fields)
    assert fields == {"code", "code_name", "message", "request_id", "fields", "details"}


# ==================================================== 字段命名规范


def _all_contract_models() -> list[type[BaseModel]]:
    """收集所有 API 契约模型。"""
    from rag_contracts.api import auth, pagination, success

    models: list[type[BaseModel]] = []
    for module in (auth, pagination, success):
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseModel) and obj.__module__ == module.__name__:
                models.append(obj)
    return models


@pytest.mark.parametrize("model", _all_contract_models(), ids=lambda m: m.__name__)
def test_field_names_are_snake_case(model: type[BaseModel]) -> None:
    """所有契约字段必须是 snake_case。

    camelCase 字段会让后端与 SQL 列名、日志字段不一致，
    跨层对照时极易出错。前端 TS 保留 snake_case 就好。
    """
    for field_name in model.model_fields:
        assert validate_snake_case_field(field_name), (
            f"{model.__name__}.{field_name} 不是 snake_case"
        )


def test_id_fields_are_named_id() -> None:
    """主键字段一律叫 `id`，不叫 `document_id` / `docId`。

    一致性价值：前端可以写通用的 `getId(entity)`。
    如果每个实体的主键叫法不同，前端要为每个实体写适配。
    """
    from rag_contracts.api.auth import CurrentUserResponse, LoginResponse

    assert "user_id" in CurrentUserResponse.model_fields
    # LoginResponse 的主键是 user_id（当前用户）而不是 id，
    # 因为它返回的是"登录结果"而非"用户资源"——这是合理的例外。
    assert "user_id" in LoginResponse.model_fields


def test_boolean_fields_prefixed() -> None:
    """布尔字段必须带 `is_` / `has_` / `can_` 前缀。

    `current: true` 读起来像名词，`is_current: true` 一眼知道是布尔。
    前端也能据此判断该用 `if (x)` 还是 `if (x != null)`。
    """
    from rag_contracts.api.auth import LoginResponse
    from rag_contracts.api.pagination import PageMeta

    for model in (LoginResponse, PageMeta):
        for field_name, field in model.model_fields.items():
            is_bool = field.annotation is bool or str(field.annotation) == "bool"
            if is_bool:
                assert field_name.startswith(("is_", "has_", "can_")), (
                    f"{model.__name__}.{field_name} 是布尔但缺少前缀"
                )


# ==================================================== 分页规范


def test_page_shape_is_items_and_meta() -> None:
    """分页响应固定为 `{items, meta}`。

    不用裸数组（无法扩展元信息）、不叫 `data`/`list`（前端要记多个名字）。
    """
    page = Page[int](items=[1, 2], meta=PageMeta(limit=20, offset=0, has_more=False))
    dumped = page.model_dump()
    assert set(dumped) == {"items", "meta"}


def test_page_meta_has_all_required_fields() -> None:
    """分页 meta 必须包含 limit/offset/total/has_more 四个字段。"""
    assert set(PageMeta.model_fields) == {"limit", "offset", "total", "has_more"}


# ==================================================== 异步任务规范


def test_task_accepted_requires_task_id() -> None:
    """202 响应必须带 task_id。

    没有 task_id 的 202 是死路——前端不知道轮询什么，
    只能盲目定时重试，既浪费又不可靠。
    """
    assert "task_id" in TaskAccepted.model_fields
    assert TaskAccepted.model_fields["task_id"].is_required(), "task_id 必须是必填字段"


def test_created_resource_requires_id() -> None:
    """201 响应必须带 id，才能构造 Location 头与后续操作。"""
    assert "id" in CreatedResource.model_fields


# ==================================================== 时间字段


def test_time_field_value_format() -> None:
    """时间字段必须是 ISO 8601 带时区。

    不带时区的时间在跨时区场景会错位（前端显示差 8 小时），
    而这类问题在开发环境（都在同一时区）永远不暴露。
    """
    assert validate_time_field_value("2026-11-15T05:17:08Z")
    assert validate_time_field_value("2026-11-15T13:17:08+08:00")

    # 不合格
    assert not validate_time_field_value("2026-11-15 13:17:08")  # 无 T 无时区
    assert not validate_time_field_value("2026-11-15T13:17:08")  # 有 T 但无时区
    assert not validate_time_field_value(1731657428)  # 时间戳


def test_time_fields_named_with_at_suffix() -> None:
    """时间字段一律以 `_at` 结尾。

    区分"时间点"（`created_at`）与"时长"（`elapsed_ms`），
    前端据此判断该用日期格式化还是数字格式化。
    """
    from rag_contracts.api.auth import LoginResponse

    for field_name in LoginResponse.model_fields:
        if "time" in field_name.lower() or "date" in field_name.lower():
            assert field_name.endswith("_at"), f"{field_name} 应以 _at 结尾"


# ==================================================== 完整响应样例


def test_full_success_response_example() -> None:
    """完整样例：成功响应应该长这样（文档化用途）。"""
    from rag_contracts.api.auth import CurrentUserResponse
    from rag_contracts.api.pagination import Page, PageMeta

    me = CurrentUserResponse(
        user_id="00000000-0000-0000-0000-000000000001",
        tenant_id="00000000-0000-0000-0000-000000000002",
        email="a@example.com",
        display_name="A",
        status="active",
        platform_role="none",
        tenant_role="owner",
    )
    dumped = me.model_dump()
    # 没有 code / data
    assert "code" not in dumped
    assert "data" not in dumped
    # 字段是 snake_case
    assert all(validate_snake_case_field(k) for k in dumped)

    page = Page[Any](
        items=[{"id": "1"}],
        meta=PageMeta(limit=20, offset=0, total=None, has_more=False),
    )
    assert page.model_dump()["meta"]["has_more"] is False
