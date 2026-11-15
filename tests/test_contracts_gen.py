"""契约生成链路测试。

锁死几个"看起来能跑但实际会坑人"的边界：

- 生成产物重复声明同名 interface → 前端 TS 编译失败
- 枚举被渲染成 TS enum 而非联合类型 → 与字面量类型不兼容
- 不支持的结构静默降级成 any → 前端失去类型保护但 CI 是绿的
- 契约改了但忘记重新生成 → 前端拿到旧类型，运行时字段对不上
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

CONTRACTS = Path(__file__).resolve().parents[1] / "packages" / "contracts"
SCRIPTS = CONTRACTS / "scripts"


def _load(module_name: str) -> Any:
    """按文件路径加载脚本模块（scripts 不是包）。"""
    path = SCRIPTS / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def export_mod() -> Any:
    return _load("export_schema")


@pytest.fixture(scope="module")
def gen_ts_mod() -> Any:
    return _load("gen_ts")


@pytest.fixture(scope="module")
def bundle(export_mod: Any) -> dict[str, Any]:
    return export_mod.build_schema_bundle()


# ==================================================== Schema 导出


def test_bundle_contains_enums_and_models(bundle: dict[str, Any]) -> None:
    """导出的包必须同时包含枚举与模型。"""
    assert "ErrorCode" in bundle["enums"]
    assert "ErrorResponse" in bundle["models"]


def test_error_codes_all_present_in_schema(bundle: dict[str, Any]) -> None:
    """枚举值必须完整导出。

    漏掉几个的实现方式是"手写白名单"，那样新增错误码时前端类型不会更新，
    前端 switch 分支会漏处理新码——编译期不报错，运行时走 default。
    """
    from rag_contracts.errors import ErrorCode

    schema_values = set(bundle["enums"]["ErrorCode"]["enum"])
    enum_values = {code.value for code in ErrorCode}
    assert schema_values == enum_values


def test_schema_is_valid_json_serializable(bundle: dict[str, Any]) -> None:
    """必须能序列化成 JSON（含中文不转义）。"""
    text = json.dumps(bundle, ensure_ascii=False)
    assert "统一错误响应体" in text


# ==================================================== TS 生成


def test_no_duplicate_interface_declarations(gen_ts_mod: Any, bundle: dict[str, Any]) -> None:
    """生成的 TS 里不能出现重复的 interface 声明。

    这是实际踩到的 bug：Pydantic 会把被引用的模型内联进 $defs，
    同时它也在 EXPORTED_MODELS 里，导致 FieldError 被渲染两次 ——
    TS 编译器直接报 "Duplicate identifier"。
    """
    source = gen_ts_mod.generate_ts(bundle)
    declarations = [
        line.split()[2].rstrip(" {")
        for line in source.splitlines()
        if line.startswith("export interface ")
    ]
    assert len(declarations) == len(set(declarations)), f"重复声明: {declarations}"


def test_enums_render_as_string_literal_union(gen_ts_mod: Any, bundle: dict[str, Any]) -> None:
    """枚举必须渲染成字符串字面量联合类型，不是 TS enum。

    TS enum 会生成运行时对象，且不能与普通字符串字面量互赋。
    前端实际用法是 `if (code === "AUTH.TOKEN_EXPIRED")`，联合类型足够，
    且编译后完全消失、零运行时开销。
    """
    source = gen_ts_mod.generate_ts(bundle)
    assert "export type ErrorCode =" in source
    # ErrorCode 是 IntEnum → 数字字面量联合类型（前端可做穷尽检查）
    assert "10101" in source, "数字错误码应渲染为数字字面量"
    assert "enum ErrorCode" not in source, "不应生成 TS enum（有运行时开销）"


def test_generated_source_has_no_any(gen_ts_mod: Any, bundle: dict[str, Any]) -> None:
    """生成结果里不能出现 any。

    any 会让前端失去类型保护，而且最糟的是 CI 依然全绿 ——
    等于类型系统在这块被静默关闭。
    """
    source = gen_ts_mod.generate_ts(bundle)
    assert ": any" not in source
    assert "as any" not in source
    assert "<any>" not in source


def test_required_and_optional_fields_correct(gen_ts_mod: Any, bundle: dict[str, Any]) -> None:
    """必填字段不能带 ?，可省字段必须带 ?。

    这是"后端契约 → 前端类型"最容易错的一处：如果全标成必填，
    前端会写出对可选字段的空值判断缺失；全标成可选，
    前端会漏拿到必填字段的空值检查。
    """
    source = gen_ts_mod.generate_ts(bundle)
    # ErrorResponse.code 是必填（数字）
    assert "code: number;" in source
    # code_name 是必填字符串
    assert "code_name: string;" in source
    # fields 有 default_factory，可省
    assert "fields?: FieldError[];" in source


def test_unsupported_construct_raises(gen_ts_mod: Any) -> None:
    """遇到不支持的 schema 必须报错，不能静默降级成 any。"""
    with pytest.raises(gen_ts_mod.UnsupportedSchemaError):
        gen_ts_mod.generate_ts(
            {
                "enums": {},
                "models": {
                    "Broken": {
                        "type": "object",
                        "properties": {"x": {"allOf": [{"type": "string"}]}},
                    }
                },
            }
        )


def test_array_of_union_gets_parentheses(gen_ts_mod: Any) -> None:
    """联合类型作为数组元素时必须加括号。

    不加括号生成 `string | number[]`，语义从"元素是 string 或 number"
    变成"string 或 number 数组"——类型完全错了，但 TS 编译器不报错。
    """
    source = gen_ts_mod.generate_ts(
        {
            "enums": {},
            "models": {
                "Wrapper": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        }
                    },
                    "required": ["items"],
                }
            },
        }
    )
    assert "(string | number)[]" in source
    assert "string | number[]" not in source


def test_nullable_renders_as_union_with_null(gen_ts_mod: Any) -> None:
    """Optional 字段必须含 | null，否则前端漏判空。"""
    source = gen_ts_mod.generate_ts(
        {
            "enums": {},
            "models": {
                "Wrapper": {
                    "type": "object",
                    "properties": {"note": {"type": ["string", "null"]}},
                    "required": ["note"],
                }
            },
        }
    )
    assert "note: string | null;" in source


# ==================================================== 漂移检查


def test_committed_artifacts_match_source() -> None:
    """入库的生成产物必须与源头契约一致。

    这条测试等价于 CI 的 `make gen-contracts-check`。放在单测里，
    是为了让"改了契约忘记重新生成"在本地 `make test` 就暴露，
    而不是等 CI 才发现。
    """
    check_mod = _load("check_contracts_fresh")
    assert check_mod.main() == 0


def test_committed_artifacts_exist() -> None:
    """生成产物必须入库。

    不入库的话，前端 clone 后必须先装 Python 环境才能拿到类型，
    而 B1 阶段前端的脚手架不该依赖 Python。
    """
    assert (CONTRACTS / "schema" / "contracts.schema.json").is_file()
    assert (CONTRACTS / "generated" / "contracts.ts").is_file()
