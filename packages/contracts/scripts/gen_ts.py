"""从 JSON Schema 生成 TypeScript 类型定义。

为什么不直接用 openapi-typescript：本机与 CI 都没有 Node，而且
"契约变更必须能被 Python 侧的 CI 校验" 这条要求决定了生成器要在 Python 里跑。

支持的 JSON Schema 子集（够用即可，遇到不支持的类型**报错而不是猜**）：
- 对象 → interface
- 字符串枚举 → 字符串字面量联合类型
- string / integer / number / boolean / array / object(additionalProperties)
- $ref → 引用名
- anyOf / oneOf → 联合类型
- nullable（type 数组或 type=null）

刻意不支持（遇到会明确报错）：
- 递归 schema（当前契约没有，真需要时再实现）
- 条件 schema（if/then/else）
- 复杂组合（allOf 带冲突字段）

**报错而不是降级成 any**：静默生成 any 会让前端失去类型保护，
而 CI 是绿的——这是最坏的情况。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HEADER = """/**
 * 本文件由 `make gen-contracts` 自动生成，请勿手改。
 *
 * 源头：packages/contracts/src/rag_contracts/**
 * 生成器：packages/contracts/scripts/gen_ts.py
 *
 * 改契约的正确姿势：
 *   1. 改 Pydantic 模型 / 枚举
 *   2. make gen-contracts
 *   3. 提交生成产物（前端 clone 后无需 Python 环境）
 */
"""


class UnsupportedSchemaError(RuntimeError):
    """JSON Schema 里出现了生成器不支持的构造。

    故意抛错而不是退回 any：静默降级会让前端失去类型检查而 CI 依然绿。
    """


def _indent(level: int) -> str:
    return "  " * level


def _doc(description: str | None) -> str:
    """把描述渲染成 JSDoc 注释（多行安全）。"""
    if not description:
        return ""
    lines = [line.strip() for line in description.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return f"/** {lines[0]} */\n"
    body = "\n".join(f" * {line}" for line in lines)
    return f"/**\n{body}\n */\n"


def _ts_type(schema: dict[str, Any], level: int = 0) -> str:
    """把单个 schema 转成 TS 类型表达式。"""
    if "$ref" in schema:
        # "#/$defs/ErrorCode" -> "ErrorCode"
        return str(schema["$ref"]).rsplit("/", 1)[-1]

    if "enum" in schema:
        values = schema["enum"]
        if not values:
            raise UnsupportedSchemaError("枚举没有取值")
        return " | ".join(json.dumps(value, ensure_ascii=False) for value in values)

    for combinator in ("anyOf", "oneOf"):
        if combinator in schema:
            parts = [_ts_type(option, level) for option in schema[combinator]]
            # 去重后拼联合类型
            seen: list[str] = []
            for part in parts:
                if part not in seen:
                    seen.append(part)
            return " | ".join(seen)

    if "allOf" in schema:
        raise UnsupportedSchemaError("不支持 allOf（请改用继承或扁平字段）")

    raw_type = schema.get("type")
    # type 可能是数组（如 ["string", "null"]），Pydantic 的 Optional 就是这样
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        if len(non_null) != 1:
            raise UnsupportedSchemaError(f"不支持多类型：{raw_type}")
        base = _ts_type({**schema, "type": non_null[0]}, level)
        return f"{base} | null" if "null" in raw_type else base

    if raw_type == "string":
        return "string"
    if raw_type == "integer":
        return "number"
    if raw_type == "number":
        return "number"
    if raw_type == "boolean":
        return "boolean"
    if raw_type == "null":
        return "null"

    if raw_type == "array":
        items = schema.get("items", {})
        item_type = _ts_type(items, level)
        # 联合类型作为数组元素时必须加括号
        if "|" in item_type and not item_type.startswith("("):
            item_type = f"({item_type})"
        return f"{item_type}[]"

    if raw_type == "object":
        properties = schema.get("properties")
        if properties:
            return _object_literal(schema, level)
        additional = schema.get("additionalProperties")
        if additional is False or additional is None:
            return "Record<string, never>"
        if additional is True:
            return "Record<string, unknown>"
        return f"Record<string, {_ts_type(additional, level)}>"

    if raw_type is None:
        # 没有 type 也没有 $ref/enum：Pydantic 有时会对 Any 输出空 schema
        if schema == {}:
            return "unknown"
        raise UnsupportedSchemaError(f"无法推断类型: {schema}")

    raise UnsupportedSchemaError(f"不支持的 type: {raw_type}")


def _object_literal(schema: dict[str, Any], level: int) -> str:
    """渲染内联对象类型（用于嵌套的匿名对象）。"""
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = ["{"]
    for name, prop in properties.items():
        doc = _doc(prop.get("description"))
        optional = "" if name in required else "?"
        prop_type = _ts_type(prop, level + 1)
        for doc_line in doc.splitlines():
            if doc_line:
                lines.append(f"{_indent(level + 1)}{doc_line.strip()}")
        # 属性名包含特殊字符时加引号
        key = name if name.isidentifier() else json.dumps(name)
        lines.append(f"{_indent(level + 1)}{key}{optional}: {prop_type};")
    lines.append(f"{_indent(level)}}}")
    return "\n".join(lines)


def _render_interface(name: str, schema: dict[str, Any]) -> str:
    """渲染一个 interface（或 type alias，取决于结构）。"""
    if schema.get("type") != "object" or "properties" not in schema:
        # 不是对象（比如联合类型/别名字符串），用 type alias
        return f"export type {name} = {_ts_type(schema)};\n"

    properties = schema["properties"]
    required = set(schema.get("required", []))
    doc = _doc(schema.get("description"))

    lines = [f"export interface {name} {{"]
    for prop_name, prop in properties.items():
        prop_doc = _doc(prop.get("description"))
        for doc_line in prop_doc.splitlines():
            if doc_line:
                lines.append(f"{_indent(1)}{doc_line.strip()}")
        optional = "" if prop_name in required else "?"
        prop_type = _ts_type(prop, 1)
        key = prop_name if prop_name.isidentifier() else json.dumps(prop_name)
        lines.append(f"{_indent(1)}{key}{optional}: {prop_type};")
    lines.append("}")
    return doc + "\n".join(lines) + "\n"


def _render_enum(name: str, schema: dict[str, Any]) -> str:
    """枚举渲染成字符串字面量联合类型。

    不用 TS enum：TS enum 会生成运行时对象，且与 `"A" | "B"` 字面量类型
    不兼容。前端实际用法是 switch/if 判断字符串值，联合类型足够且零运行时。
    """
    values = schema["enum"]
    union = " | ".join(json.dumps(value, ensure_ascii=False) for value in values)
    doc = _doc(schema.get("description"))
    return doc + f"export type {name} =\n  | {union.replace(' | ', chr(10) + '  | ')};\n"


def generate_ts(bundle: dict[str, Any]) -> str:
    """把 schema 包渲染成完整的 TS 文件内容。"""
    parts: list[str] = [HEADER]

    enums: dict[str, Any] = bundle.get("enums", {})
    if enums:
        parts.append("// ============================ 枚举 ============================\n")
        for name, schema in enums.items():
            parts.append(_render_enum(name, schema))

    models: dict[str, Any] = bundle.get("models", {})
    if models:
        parts.append("// =========================== 数据模型 ===========================\n")
        # 先渲染被其它模型引用的 $defs，再渲染主体。
        # 注意：独立导出的模型可能同时出现在别人的 $defs 里（Pydantic 会内联
        # 引用），必须去重，否则 TS 里会重复声明同一个 interface 而编译报错。
        defs: dict[str, Any] = {}
        for schema in models.values():
            defs.update(schema.get("$defs", {}))

        rendered: set[str] = set()

        for name, schema in defs.items():
            if name in enums or name in models:
                # 枚举已有独立定义；models 里的稍后按主体渲染
                continue
            parts.append(_render_interface(name, schema))
            rendered.add(name)

        for name, schema in models.items():
            if name in rendered:
                continue
            # 主体模型里剔除 $defs，避免重复渲染
            body = {key: value for key, value in schema.items() if key != "$defs"}
            parts.append(_render_interface(name, body))
            rendered.add(name)

    return "\n".join(parts)


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    schema_path = here / "schema" / "contracts.schema.json"
    if not schema_path.is_file():
        raise SystemExit(f"缺少 schema 文件：{schema_path}\n请先运行 make gen-contracts")

    bundle = json.loads(schema_path.read_text(encoding="utf-8"))
    ts_source = generate_ts(bundle)

    output = here / "generated" / "contracts.ts"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ts_source, encoding="utf-8")
    print(f"已生成 TS 类型: {output}（{len(ts_source.splitlines())} 行）")


if __name__ == "__main__":
    main()
