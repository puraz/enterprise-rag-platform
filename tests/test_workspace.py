"""骨架冒烟测试：证明 workspace 装好了，且端边界真的被守住。

这里守的是"共享 venv 里也能 import"的漏洞：
uv workspace 把四个包装进同一个 .venv，光靠"不声明依赖"挡不住 import，
所以要在 CI 里做静态 import 检查 + 依赖声明检查。
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import rag_api
import rag_contracts
import rag_core
import rag_worker

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_SRC = REPO_ROOT / "apps" / "worker" / "src"
FORBIDDEN_MODULE = "rag_api"


def test_all_workspace_packages_are_importable() -> None:
    """四个包都能 import，且版本来自发行包元数据（单一真相）。"""
    for module in (rag_core, rag_contracts, rag_api, rag_worker):
        assert module.__version__


def test_worker_does_not_import_api() -> None:
    """静态检查层：worker 源码出现 rag_api 的 import 就失败。"""
    offenders: list[str] = []
    for path in WORKER_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == FORBIDDEN_MODULE or n.startswith(f"{FORBIDDEN_MODULE}.") for n in names):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"worker 越界 import {FORBIDDEN_MODULE}: {offenders}"


def test_worker_package_does_not_declare_api_dependency() -> None:
    """依赖声明层：worker 的 pyproject 不得出现 rag-api。"""
    pyproject_path = REPO_ROOT / "apps" / "worker" / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies: list[str] = pyproject["project"]["dependencies"]
    assert not [dep for dep in dependencies if dep.startswith("rag-api")], dependencies
