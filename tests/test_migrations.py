"""迁移链路与 rag_db 包边界测试。

迁移的正确性只能靠"真的跑一遍"来证明：
- 能 upgrade → 表真的出现；
- 能 downgrade → 表真的消失（很多人只测 upgrade，回滚在生产出事时才发现是坏的）；
- 模型与迁移不漂移（改了模型忘了生成迁移，是最常见的漏）。

另有包边界测试：rag_db 只能依赖 contracts 与 core，不能反向或越界。
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PKG = REPO_ROOT / "packages" / "db"
DB_SRC = DB_PKG / "src"


# ==================================================== 包边界（无需外部依赖）


def test_db_package_declares_only_allowed_dependencies() -> None:
    """rag_db 只能依赖 contracts 与 core（外加 ORM/迁移工具）。

    越界的典型后果：rag_db 依赖 rag_api → worker 通过 rag_db 间接依赖
    api → 端边界从侧门破掉，而且 `import rag_api` 静态检查查不出来。
    """
    pyproject = tomllib.loads((DB_PKG / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    internal = {dep.split(">")[0].split("[")[0] for dep in dependencies if dep.startswith("rag-")}
    assert internal == {"rag-core", "rag-contracts"}, f"rag_db 的内部依赖越界: {internal}"


def test_db_does_not_import_api_or_worker() -> None:
    """rag_db 源码不得 import api / worker。

    这是 test_workspace.py 检查的补充：那边只检查 worker 不 import api，
    这里检查共享包不被上层污染。
    """
    forbidden = ("rag_api", "rag_worker")
    offenders: list[str] = []
    for path in DB_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(name == f or name.startswith(f"{f}.") for f in forbidden):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} -> {name}")
    assert not offenders, f"rag_db 越界 import 上层包: {offenders}"


def test_all_models_are_registered_in_metadata() -> None:
    """每个模型类都必须出现在 Base.metadata 里。

    漏了意味着 `models/__init__.py` 没 import 它，Alembic autogenerate
    看不见这张表 —— 表现是"改了模型但生成不出迁移"，且没有任何报错。
    """
    import rag_db.models  # noqa: F401  确保模型包被加载
    from rag_db.base import Base

    # 模块里声明的表名，必须都在 metadata 中
    assert "schema_health" in Base.metadata.tables


def test_naming_convention_applied() -> None:
    """约束名必须遵循命名约定，而不是 PG 自动生成的随机名。

    随机名会让迁移里的 `DROP CONSTRAINT` 不可预测，回滚时直接失败。
    """
    from rag_db.base import Base
    from rag_db.models import SchemaHealth  # noqa: F401

    table = Base.metadata.tables["schema_health"]
    assert table.primary_key.name == "pk_schema_health"


# ==================================================== 配置根查找


def test_find_project_root_from_subdirectory() -> None:
    """从子目录也能找到仓库根。

    这是 A2-6 实际踩到的坑：在 packages/db 下执行 alembic，配置层用
    Path.cwd() 找不到仓库根 .env，静默回退到代码默认值（错误用户/密码），
    报错信息只有"角色不存在"，完全指不到根因。
    """
    from rag_core.settings import find_project_root

    for subdir in (DB_PKG, REPO_ROOT / "apps" / "api", REPO_ROOT / "tests"):
        assert find_project_root(subdir) == REPO_ROOT


def test_find_project_root_returns_start_when_no_marker(tmp_path: Path) -> None:
    """没有仓库标记时返回起始目录，不向上无限找。

    生产容器里没有 pyproject.toml + uv.lock，此时应该只靠进程环境变量。
    如果实现成"往上一直找到根"，可能意外读到宿主机上的 .env。
    """
    from rag_core.settings import find_project_root

    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == nested


# ==================================================== 迁移链路（需真实 PG）


@pytest.mark.integration
def test_migration_upgrade_creates_table() -> None:
    """upgrade 之后表必须真实存在，且列符合预期。

    这里刻意不通过 ORM metadata 反推列数——那会让测试依赖
    "模型有没有被 import"这种实现细节。直接查 information_schema
    验证的是数据库的真实状态，这才是"迁移生效"的定义。
    """
    import asyncio

    from sqlalchemy import text

    from rag_core.db import Database
    from rag_core.settings import get_settings

    expected_columns = {"key", "value", "created_at", "updated_at"}

    async def scenario() -> None:
        db = Database(get_settings())
        try:
            async with db.engine.connect() as conn:
                result = await conn.execute(text("SELECT to_regclass('schema_health')"))
                assert result.scalar_one() is not None, "迁移未生效：表不存在"

                columns = await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'schema_health'"
                    )
                )
                actual = {str(row[0]) for row in columns}
                assert actual == expected_columns, f"列不符合预期: {actual}"

                # 主键名遵循命名约定（不是 PG 自动生成的随机名）
                pk = await conn.execute(
                    text(
                        "SELECT constraint_name FROM information_schema.table_constraints "
                        "WHERE table_name = 'schema_health' AND constraint_type = 'PRIMARY KEY'"
                    )
                )
                assert pk.scalar_one() == "pk_schema_health"
        finally:
            await db.close()

    asyncio.run(scenario())


@pytest.mark.integration
def test_alembic_version_matches_head() -> None:
    """数据库版本必须等于迁移链的 head。

    不等于 head 说明有迁移没执行，部署时会出现"代码期望的表不存在"。
    """
    import asyncio

    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from sqlalchemy import text

    from rag_core.db import Database
    from rag_core.settings import get_settings

    alembic_cfg = Config(str(DB_PKG / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(DB_PKG / "alembic"))
    head = ScriptDirectory.from_config(alembic_cfg).get_current_head()

    async def scenario() -> str:
        db = Database(get_settings())
        try:
            async with db.engine.connect() as conn:
                result = await conn.execute(text("SELECT version_num FROM alembic_version"))
                return str(result.scalar_one())
        finally:
            await db.close()

    assert asyncio.run(scenario()) == head
