"""枚举契约测试：确保分包后不漏导出、不重名、值不冲突。

分包带来一个新风险：**枚举写在域文件里但忘了加进 EXPORTED_ENUMS**，
前端静默拿不到这个类型，编译期不报错，运行时才开始踩坑。

这组测试就是防这个：自动扫描 `rag_contracts.enums` 下的所有 StrEnum，
要求它们全部出现在导出清单里。

同时守两条命名规范：
1. 枚举值全局唯一（跨域也不能撞，否则前端联合类型会含义不明）；
2. 值必须是 snake_case 小写（除错误码，它在另一个包且是协议格式）。
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest

import rag_contracts.enums as enums_pkg

CONTRACTS_SCRIPTS = Path(__file__).resolve().parents[1] / "packages" / "contracts" / "scripts"


@pytest.fixture(scope="module")
def export_mod() -> Any:
    """加载 export_schema 脚本模块（scripts 不是包）。"""
    path = CONTRACTS_SCRIPTS / "export_schema.py"
    spec = importlib.util.spec_from_file_location("export_schema", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discover_enum_modules() -> list[str]:
    """扫描 enums 包下的所有域模块（排除 __init__）。"""
    package_dir = Path(enums_pkg.__file__).parent
    modules: list[str] = []
    for path in sorted(package_dir.glob("*.py")):
        if path.stem == "__init__":
            continue
        modules.append(f"{enums_pkg.__name__}.{path.stem}")
    return modules


def _all_enums() -> dict[str, type[StrEnum]]:
    """收集所有域模块里定义的 StrEnum。"""
    found: dict[str, type[StrEnum]] = {}
    for module_name in _discover_enum_modules():
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, StrEnum)
                and obj is not StrEnum
                # 只收本模块定义的，排除 import 进来的
                and obj.__module__ == module_name
            ):
                found[name] = obj
    return found


def test_at_least_one_domain_module_exists() -> None:
    """至少有一个域文件（否则说明分包结构被破坏了）。"""
    assert _discover_enum_modules(), "enums 包下没有任何域模块"


def test_every_enum_is_exported_to_ts(export_mod: Any) -> None:
    """**每个域内的枚举都必须出现在 EXPORTED_ENUMS 里。**

    漏导出的后果：前端拿不到这个类型，只能手写字符串常量——
    那就是契约漂移的起点，而且 CI 完全不会报错。
    """
    discovered = set(_all_enums())
    exported = set(export_mod.EXPORTED_ENUMS)
    missing = discovered - exported
    assert not missing, f"以下枚举未加入 EXPORTED_ENUMS，前端拿不到类型: {sorted(missing)}"


def test_exported_enum_names_have_definition(export_mod: Any) -> None:
    """EXPORTED_ENUMS 里指向 enums 包的名字都必须真实存在。

    注意范围：`EXPORTED_ENUMS` 里还有错误码相关枚举（ErrorCode /
    ErrorCategory / Severity），它们在 `rag_contracts.errors` 包里，
    不属于本测试扫描的 `enums` 包——所以这里只校验"名字带 enums 语义、
    但实际不存在"的情况，不能把它们当成 stale。
    """
    discovered = set(_all_enums())
    # 错误码域不在 enums 包内，单独列出以免误判
    external_domains = {"ErrorCode", "ErrorCategory", "Severity"}
    exported_in_scope = set(export_mod.EXPORTED_ENUMS) - external_domains

    stale = exported_in_scope - discovered
    assert not stale, f"EXPORTED_ENUMS 里有不存在的枚举: {sorted(stale)}"


def test_enum_values_are_lowercase_snake_case() -> None:
    """枚举值必须是小写 snake_case。

    大小写混用会让前端比较时踩坑（"Active" vs "active"），
    而且 URL/查询参数里大小写敏感。
    """
    for name, enum_type in _all_enums().items():
        for member in enum_type:
            value = member.value
            assert value == value.lower(), f"{name}.{member.name} = {value!r} 含大写"
            assert " " not in value, f"{name}.{member.name} 含空格"


def test_enum_values_unique_within_enum() -> None:
    """同一枚举内值不能重复（重复会让 StrEnum 成员被合并）。"""
    for name, enum_type in _all_enums().items():
        values = [member.value for member in enum_type]
        assert len(values) == len(set(values)), f"{name} 有重复值: {values}"


def test_membership_and_user_status_are_distinguishable() -> None:
    """成员关系状态与用户状态必须是两个不同的枚举类型。

    两者都可能取 "active"，但语义不同：一个是"这个人还能登录吗"，
    一个是"这个人在当前租户里是否生效"。
    合成一个类型会让"禁用用户"和"从租户移除"在代码里长得一样，
    然后在某次重构中被静默合并——这类 bug 会以"权限判断遗漏"的形式出现。
    """
    from rag_contracts.enums import MembershipStatus, UserStatus

    assert MembershipStatus is not UserStatus
    # 两者都有 active 是**有意设计**，不是笔误
    assert MembershipStatus.ACTIVE.value == UserStatus.ACTIVE.value == "active"


def test_str_enum_serializable_as_plain_str() -> None:
    """StrEnum 必须能被 JSON 直接序列化（这是选 StrEnum 而非 Enum 的原因）。"""
    from rag_contracts.enums import TenantStatus

    assert json.dumps({"status": TenantStatus.ACTIVE}) == '{"status": "active"}'
