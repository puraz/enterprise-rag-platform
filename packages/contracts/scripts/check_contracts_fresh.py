"""检查生成产物是否与源头契约漂移。

用途：CI 里跑，确保「改了 Pydantic 模型但忘记跑 make gen-contracts」会失败。

为什么这是必需的：
前端类型是生成产物，入库是为了让前端 clone 就能开发。但入库带来一个风险：
总有人改完后端契约直接提交，忘了重新生成 —— 前端拿到的是旧类型，
编译期不报错，运行时字段对不上。这个检查把"忘记"变成 CI 红。

实现：重新生成到临时目录，与入库文件逐字节比较。
不直接改文件，避免 CI 里意外写入工作区。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# scripts 目录不是包，直接按文件路径加载
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

import export_schema  # noqa: E402
import gen_ts  # noqa: E402


def main() -> int:
    here = _SCRIPTS.parent
    committed_schema = here / "schema" / "contracts.schema.json"
    committed_ts = here / "generated" / "contracts.ts"

    problems: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        actual_schema_path = export_schema.write_schema(tmp_path)
        actual_schema = actual_schema_path.read_text(encoding="utf-8")

        if not committed_schema.is_file():
            problems.append(f"缺少 {committed_schema.relative_to(here)}")
        elif committed_schema.read_text(encoding="utf-8") != actual_schema:
            problems.append(f"{committed_schema.relative_to(here)} 与源头契约不一致")

        actual_ts = gen_ts.generate_ts(export_schema.build_schema_bundle())
        if not committed_ts.is_file():
            problems.append(f"缺少 {committed_ts.relative_to(here)}")
        elif committed_ts.read_text(encoding="utf-8") != actual_ts:
            problems.append(f"{committed_ts.relative_to(here)} 与源头契约不一致")

    if problems:
        print("❌ 契约生成产物已漂移：")
        for problem in problems:
            print(f"   - {problem}")
        print("\n请运行 `make gen-contracts` 后提交生成产物。")
        return 1

    print("✅ 契约生成产物与源头一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
