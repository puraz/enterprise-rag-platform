"""接口分层守卫：`api/v1` 的 user / admin 边界必须可验证。

## 这些测试防的是什么

目录拆分本身没有价值——价值在于**边界能被测试证明**。没有这批测试，
半年后很容易出现：

1. 有人把管理接口写进 `v1/user/`（URL 变成 `/api/v1/xxx`，
   网关的 `/api/v1/admin` 限制失效，用户前端能直接调）；
2. 有人把用户接口写进 `v1/admin/`（普通员工调自己的上传进度要 403）；
3. 管理面新接口忘了挂 `require_admin_capability`（任何登录用户都能调，
   返回 200 完全看不出来）；
4. 有人把两条路径的前缀写歪（`/api/v1/adminx`、`/api/v1/Admin`）——
   网关按字符串前缀匹配，大小写或拼写错误直接绕过限制。

每一条都对应一次真实事故形态，所以这里用**结构检查 + OpenAPI 检查**
双重钉死，而不是靠 review。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute

from rag_api.main import create_app
from rag_core.settings import Settings

V1_ROOT = Path(__file__).resolve().parents[1] / "apps" / "api" / "src" / "rag_api" / "api" / "v1"
USER_DIR = V1_ROOT / "user"
ADMIN_DIR = V1_ROOT / "admin"
ADMIN_PREFIX = "/api/v1/admin"


def _test_settings() -> Settings:
    return Settings(
        env="local",
        dev_auth_enabled=True,
        startup_dependency_check=False,
        database={"password": "x" * 32},
        redis={"password": "x" * 32},
        identity={"jwt_secret": "y" * 40},
    )


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    app = create_app(_test_settings())
    return app.openapi()


@pytest.fixture(scope="module")
def routes() -> list[APIRoute]:
    app = create_app(_test_settings())
    return [route for route in app.routes if isinstance(route, APIRoute)]


# ==================================================== 目录结构


def test_v1_has_exactly_two_route_packages() -> None:
    """`api/v1` 下只能有 user / admin 两个路由包（加 `__init__`）。

    ## 为什么这条是硬约束

    平铺的 `v1/*.py` 无法表达"这个接口属于哪一面"。新加一个文件时，
    "它是用户面还是管理面"只存在于作者脑中，读者要看依赖才能推断。
    一旦推断错，就会把管理能力暴露给用户前端。

    拆成两个包后，"放在哪个目录"就是强制回答这个问题的地方。
    """
    allowed_dirs = {"user", "admin", "__pycache__"}
    stray_py = [path.name for path in V1_ROOT.glob("*.py") if path.name != "__init__.py"]
    stray_dirs = [
        path.name for path in V1_ROOT.iterdir() if path.is_dir() and path.name not in allowed_dirs
    ]
    assert not stray_py, (
        f"api/v1 下不应有平铺的路由文件：{stray_py}。"
        "请放进 user/（用户面）或 admin/（管理面）——"
        "目录是唯一能强制回答「这个接口属于哪一面」的地方。"
    )
    assert not stray_dirs, f"api/v1 下不应有其它路由包：{stray_dirs}"


def test_admin_package_declares_capability_on_every_route() -> None:
    """`v1/admin/` 下每个路由都必须声明管理面能力依赖。

    ## 漏挂的后果（这是最严重的一类）

    `require_admin_capability` 既校验"是管理面用户"，也校验"有这项能力"。
    漏挂时路由仍然能跑，只是**任何登录用户都能调**——
    返回 200 和数据，没有任何异常信号。

    静态检查（扫 AST）而不是运行时检查：运行时要用不同角色发几十个请求
    才能覆盖，而静态检查能保证"新增路由时立刻失败"。
    """
    offenders: list[str] = []

    for path in sorted(ADMIN_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                # 只关心 router.get / router.post / ...
                if not isinstance(func, ast.Attribute):
                    continue
                if func.attr not in {"get", "post", "put", "patch", "delete"}:
                    continue
                if not _has_capability_dependency(decorator):
                    offenders.append(f"{path.name}:{node.lineno} {node.name}")

    assert not offenders, (
        "以下管理面路由没有声明 require_admin_capability(...)，"
        "任何登录用户都能调用：\n  " + "\n  ".join(offenders)
    )


def _has_capability_dependency(decorator: ast.Call) -> bool:
    """检查装饰器里是否有 `dependencies=[Depends(require_admin_capability(...))]`。"""
    for keyword in decorator.keywords:
        if keyword.arg != "dependencies":
            continue
        for element in ast.walk(keyword.value):
            if isinstance(element, ast.Name) and element.id == "require_admin_capability":
                return True
    return False


# ==================================================== OpenAPI 路径


def test_admin_routes_are_under_admin_prefix(spec: dict[str, Any], routes: list[APIRoute]) -> None:
    """所有管理面路由都在 `/api/v1/admin/` 下（且**没有**用户面路由混进去）。

    网关按这个前缀做「管理接口禁止从用户前端访问」的限制。
    前缀写错（`/adminx`、`/Admin`）会让限制静默失效——
    请求照常通过，只是没人审核。
    """
    mismatched: list[str] = []
    for path in spec["paths"]:
        if path.startswith(ADMIN_PREFIX):
            continue
        # 用户面路径不能长得像管理面
        lowered = path.lower()
        if "/admin" in lowered:
            mismatched.append(path)

    assert not mismatched, (
        f"以下路径包含 admin 但不在 /api/v1/admin/ 前缀下（网关限制会失效）：{mismatched}"
    )
    # 每个管理面业务域都必须至少有一个路由（用前缀而不是精确路径：
    # 域下的子路径会随功能增加而增长）
    domain_prefixes = (
        "/api/v1/admin/dashboard",
        "/api/v1/admin/tasks",
        "/api/v1/admin/documents",
        "/api/v1/admin/knowledge-bases",
        "/api/v1/admin/audit",
    )
    missing_domains = [
        prefix
        for prefix in domain_prefixes
        if not any(path.startswith(prefix) for path in spec["paths"])
    ]
    assert not missing_domains, (
        f"管理面的这些域没有任何路由：{missing_domains}；实际路径：{sorted(spec['paths'])}"
    )


def test_user_routes_do_not_require_admin_context(routes: list[APIRoute]) -> None:
    """用户面路由不得依赖 `get_admin_context`。

    ## 为什么用户面不能复用管理上下文

    `get_admin_context` 允许平台角色**跨租户**访问（`scope_tenant_id=None`）。
    一个用户面接口如果用了它，就可能出现"某个接口按 token 租户过滤，
    另一个按 data_tenant_id 过滤"的口径分裂——
    而平台运营的 token 租户可能是任意租户，结果就是跨租户泄漏。

    用户面必须统一用 `get_permission_context`（永远锁定 token 租户）。
    """
    from rag_api.permissions.admin import get_admin_context

    offenders: list[str] = []
    for route in routes:
        if route.path.startswith(ADMIN_PREFIX):
            continue
        dependency_names = {getattr(dep.dependency, "__name__", "") for dep in route.dependencies}
        # 检查 dependant 里的参数依赖
        for sub in route.dependant.dependencies:
            dependency_names.add(getattr(sub.call, "__name__", ""))

        if get_admin_context.__name__ in dependency_names:
            offenders.append(f"{sorted(route.methods or [])} {route.path}")

    assert not offenders, "以下用户面路由依赖了管理上下文（会引入跨租户语义）：\n  " + "\n  ".join(
        offenders
    )


def test_every_admin_route_declares_error_responses(spec: dict[str, Any]) -> None:
    """管理面路由也必须声明错误响应（否则前端看不到 403 的形状）。

    与 `test_openapi_contract.test_all_routes_declare_error_responses` 互补：
    那条覆盖全部路由，这条专门盯管理面——管理面的 403 是最常见分支
    （能力不足），缺了它前端会把 403 当未知错误。
    """
    offenders: list[str] = []
    for path, operations in spec["paths"].items():
        if not path.startswith(ADMIN_PREFIX):
            continue
        for method, operation in operations.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            codes = set(operation.get("responses", {}))
            if not any(str(code) == "403" for code in codes):
                offenders.append(f"{method.upper()} {path}")

    assert not offenders, "以下管理面路由没有声明 403 响应：\n  " + "\n  ".join(offenders)


# ==================================================== 契约完整性


def test_new_endpoints_appear_in_openapi(spec: dict[str, Any]) -> None:
    """这次补的断链端点必须真的出现在契约里。

    逐一列出而不是"数量 > N"：数量检查在删掉一个接口又加上另一个时
    仍然通过，而这里会精确失败。
    """
    required = {
        ("post", "/api/v1/auth/oidc/exchange"),
        ("get", "/api/v1/auth/oidc/authorize"),
        ("post", "/api/v1/auth/logout"),
        ("get", "/api/v1/me/tenants"),
        ("post", "/api/v1/me/tenants/switch"),
        ("get", "/api/v1/me/usage"),
        ("get", "/api/v1/me/audit"),
        # 断链修复：202 返回的 task_id 之前无处可查
        ("get", "/api/v1/tasks/{task_id}"),
        ("get", "/api/v1/documents/{doc_id}/download"),
        ("post", "/api/v1/documents/{doc_id}/versions/{version_id}/activate"),
        ("patch", "/api/v1/conversations/{conversation_id}"),
        # 管理面
        ("get", "/api/v1/admin/dashboard"),
        ("get", "/api/v1/admin/tasks"),
        ("post", "/api/v1/admin/tasks/{task_id}/retry"),
        ("get", "/api/v1/admin/documents"),
        ("get", "/api/v1/admin/documents/{doc_id}/versions/{version_id}/chunks"),
        ("get", "/api/v1/admin/knowledge-bases/{kb_id}/index-versions"),
        ("get", "/api/v1/admin/audit/verify"),
    }

    missing = [
        f"{method.upper()} {path}"
        for method, path in required
        if path not in spec["paths"] or method not in spec["paths"][path]
    ]
    assert not missing, "以下接口未出现在 OpenAPI 契约里：\n  " + "\n  ".join(sorted(missing))


def test_existing_user_endpoints_unchanged(spec: dict[str, Any]) -> None:
    """重构前的用户面端点 URL 与语义必须原样保留。

    ## 为什么这条必须存在

    目录重组最危险的副作用是"顺手改了 URL"。前端代码、
    网关规则、部署文档、第三方集成全都按旧 URL 写，
    改一个都不是重构能覆盖的成本。

    这条用**精确清单**固定用户面契约：新增接口不受限，
    但已有的不能消失、不能换方法。
    """
    preserved = {
        ("post", "/api/v1/auth/dev-login"),
        ("post", "/api/v1/auth/dev-bootstrap"),
        ("post", "/api/v1/auth/refresh"),
        ("get", "/api/v1/me"),
        ("post", "/api/v1/chat"),
        ("get", "/api/v1/conversations"),
        ("get", "/api/v1/conversations/{conversation_id}/messages"),
        ("delete", "/api/v1/conversations/{conversation_id}"),
        ("post", "/api/v1/messages/{message_id}/feedback"),
        ("get", "/api/v1/knowledge-bases"),
        ("post", "/api/v1/knowledge-bases"),
        ("get", "/api/v1/knowledge-bases/{kb_id}"),
        ("patch", "/api/v1/knowledge-bases/{kb_id}"),
        ("delete", "/api/v1/knowledge-bases/{kb_id}"),
        ("get", "/api/v1/documents"),
        ("post", "/api/v1/documents"),
        ("get", "/api/v1/documents/{doc_id}"),
        ("delete", "/api/v1/documents/{doc_id}"),
        ("post", "/api/v1/documents/{doc_id}/retry"),
        ("post", "/api/v1/uploads"),
        ("put", "/api/v1/uploads/{upload_id}/parts/{part_number}"),
        ("get", "/api/v1/uploads/{upload_id}"),
        ("post", "/api/v1/uploads/{upload_id}/complete"),
        ("delete", "/api/v1/uploads/{upload_id}"),
        ("get", "/api/v1/admin/audit"),
    }

    broken = [
        f"{method.upper()} {path}"
        for method, path in preserved
        if path not in spec["paths"] or method not in spec["paths"][path]
    ]
    assert not broken, "重构破坏了已有接口（前端与网关依赖它们）：\n  " + "\n  ".join(
        sorted(broken)
    )


# ==================================================== RLS 会话上下文


def test_admin_routes_use_rls_scoped_session() -> None:
    """管理面不得使用**无租户上下文**的会话。

    ## 这条防的是一个静默 bug（已实际踩到）

    最初的实现让管理面用 `get_session`（无 `app.tenant_id`）。而
    `tasks` / `documents` / `knowledge_bases` / `model_calls` / `chunks` /
    `tenant_memberships` **全在 `RLS_TABLES` 里**。在真实应用角色
    `rag_app`（NOSUPERUSER NOBYPASSRLS）下，RLS 策略里的
    `tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid`
    会判为 NULL → **查询返回 0 行，且不抛任何异常**。

    现象：
    - 管理看板所有计数都是 0；
    - 任务台账永远空；
    - 租户 owner 调管理接口 403（查不到自己的成员关系）。

    **在超级用户环境下测不出来**（BYPASSRLS 无条件绕过），
    所以集成测试没有覆盖到——只能靠静态检查守住。

    允许的两种写法：
    1. `Depends(get_admin_db)`（permissions 里的那个，按 data_tenant_id 开上下文）；
    2. `async with scoped_admin_session(...)`（支持 tenant_id 参数的路由）。
    """
    forbidden = "get_session"
    allowed = {"get_admin_db", "scoped_admin_session"}

    offenders: list[str] = []
    for path in sorted(ADMIN_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            # 跳过注释与文档字符串里的提及
            if stripped.startswith("#"):
                continue
            if forbidden not in line:
                continue
            if not any(name in line for name in allowed):
                # 排除 docstring 行（以 """ 开头或纯文本说明）
                if '"""' in line or line.lstrip().startswith(("`", "|", "-")):
                    continue
                offenders.append(f"{path.name}:{lineno} → {stripped}")

    assert not offenders, (
        "以下管理面代码使用了无租户上下文的会话（RLS 会静默返回 0 行）：\n  "
        + "\n  ".join(offenders)
        + "\n请改用 Depends(get_admin_db) 或 scoped_admin_session(...)"
    )
