"""索引重建链路与跨租户汇总的守卫测试（纯单测，零外部依赖）。

## 这批测试盯的是「静默错」

索引重建与跨租户汇总都属于"跑起来不报错、结果却是错的"那类：

| 错误 | 现象 | 为什么难发现 |
|---|---|---|
| 回填塞进一个事务 | 连接被占半小时，锁不刷新被重派 | 任务最终"成功"，只是白算两遍 |
| 不校验完成度就切换 | 检索静默丢失内容 | 用户说"查不到"，而文档在库里 |
| 跨租户汇总用了带租户上下文的会话 | 只返回一个租户 | 数字看起来合理，只是漏了别人 |
| 跨租户汇总没做角色检查 | 租户管理员看到全平台 | 返回 200 与正常数据 |

所以这里用**结构检查**（AST / 签名）+ **契约校验** 双管齐下，
而不是依赖需要真实 PG 的集成测试。
"""

from __future__ import annotations

import ast
import inspect
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from rag_core.errors import ConflictError, PermissionDeniedError
from rag_worker.handlers.rebuild import RebuildIndexHandler

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_HANDLERS = REPO_ROOT / "apps" / "worker" / "src" / "rag_worker" / "handlers"
ADMIN_ROUTES = REPO_ROOT / "apps" / "api" / "src" / "rag_api" / "api" / "v1" / "admin"


# ==================================================== TaskHandler 契约


def test_rebuild_handler_declares_owns_transaction() -> None:
    """重建 handler 必须声明 `owns_transaction = True`。

    ## 漏声明会怎样（这是本设计的核心约束）

    引擎默认把 handler 包在一个事务里。若重建走默认分支：

    - 十万 chunk 的回填全在**一个事务**里，持续几十分钟占住连接、
      阻塞 vacuum；
    - 崩溃时进度全部丢失（下次从头再来）；
    - `locked_at` 不更新 → 超过 `lock_timeout` 后被 `reclaim_stale`
      判为卡死 → **两个 worker 同时回填同一版本**（付款翻倍）。

    这三条都不会报错，只会让运维觉得"重建怎么这么慢/这么贵"。
    """
    assert RebuildIndexHandler.owns_transaction is True


def test_single_document_handlers_do_not_own_transaction() -> None:
    """单文档 handler 必须走默认分支（整段一个事务）。

    它们的数据量小（一个文档），整段回滚才是对的——失败时不该留下
    半套 chunk。如果误声明 `owns_transaction = True`，引擎会传
    `session=None`，handler 立刻 `assert` 失败。

    这条同时验证"只有长任务才接管事务"这个边界没有被扩大。
    """
    from rag_worker.handlers.chunk import ChunkDocumentHandler
    from rag_worker.handlers.embed import EmbedDocumentHandler
    from rag_worker.handlers.parse import ParseDocumentHandler

    for handler in (ParseDocumentHandler, ChunkDocumentHandler, EmbedDocumentHandler):
        assert handler.owns_transaction is False, f"{handler.__name__} 不该自管事务"


def test_rebuild_handler_accepts_session_factory() -> None:
    """重建 handler 的签名必须接受 `session_factory`。

    引擎按 `owns_transaction` 分支传不同参数。签名不接收工厂时
    会在运行期 `TypeError`——而那是在**任务已认领之后**才发生，
    表现为"任务一直失败且错误信息是 TypeError"，根因（协议不匹配）
    完全看不出来。
    """
    params = inspect.signature(RebuildIndexHandler.__call__).parameters
    assert "session_factory" in params
    assert "session" in params


def test_rebuild_handler_rejects_session_instance() -> None:
    """重建 handler 收到**已开启的会话**时必须拒绝。

    这是"误用防护"：如果哪天引擎分支写错、又给自管事务的 handler
    传了 session，handler 应该立刻失败而不是把整段回填塞进那个会话。
    静默接受会让长事务问题重新出现，且没人会注意到。
    """
    import asyncio

    handler = RebuildIndexHandler(provider=_FakeProvider())

    task = _TaskStub(
        payload={
            "knowledge_base_id": str(uuid.uuid4()),
            "index_version_id": str(uuid.uuid4()),
        }
    )

    async def _run() -> None:
        await handler(
            session=object(),  # type: ignore[arg-type]  # 故意传非 None
            task=task,
            progress=_noop_progress,
            session_factory=_never_opened_factory,
        )

    with pytest.raises(Exception, match="不应收到已开启的会话"):
        asyncio.run(_run())


def test_rebuild_handler_rejects_missing_factory() -> None:
    """缺少 `session_factory` 时必须永久失败（重试无用）。

    引擎分支写错是**配置/代码错误**，不是瞬时故障。重试三次只会
    让任务在死信里多三条一样的记录，还推迟了人工发现的时间。
    """
    import asyncio

    handler = RebuildIndexHandler(provider=_FakeProvider())

    task = _TaskStub(payload={})

    async def _run() -> None:
        await handler(
            session=None,
            task=task,
            progress=_noop_progress,
            session_factory=None,
        )

    with pytest.raises(Exception, match="缺少 session_factory") as exc_info:
        asyncio.run(_run())
    # retryable=False 才会进死信而不是被重试 3 次
    assert getattr(exc_info.value, "retryable", None) is False


# ==================================================== 管理面路由结构


def _admin_route_functions() -> dict[str, ast.AsyncFunctionDef]:
    """收集 admin 包下所有路由处理函数（按函数名索引）。"""
    found: dict[str, ast.AsyncFunctionDef] = {}
    for path in sorted(ADMIN_ROUTES.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                if isinstance(func, ast.Attribute) and func.attr in {
                    "get",
                    "post",
                    "put",
                    "patch",
                    "delete",
                }:
                    found[node.name] = node
    return found


def test_index_rebuild_requires_operate_capability() -> None:
    """发起重建需要 `OPERATE`（不是只读，也不是治理）。

    回填会消耗大量 embedding API 配额。审计员（只有 READ/AUDIT）
    与只读运营都不该能触发它。
    """
    source = (ADMIN_ROUTES / "indexing.py").read_text(encoding="utf-8")
    # create_index_rebuild 的装饰器块里必须有 OPERATE
    idx = source.index("async def create_index_rebuild")
    decorator_block = source[:idx]
    last_decorator = decorator_block[decorator_block.rindex("@router.post") :]
    assert "AdminCapability.OPERATE" in last_decorator


def test_activate_and_rollback_require_admin_capability() -> None:
    """切换与回滚需要 `ADMIN`（治理级，不是普通运维）。

    切换会立刻改变所有检索结果；回滚同样影响召回。这类"改变系统行为"
    的操作比"重跑一次任务"（OPERATE）高一档。
    """
    source = (ADMIN_ROUTES / "indexing.py").read_text(encoding="utf-8")

    for func_name in ("activate_index_version", "rollback_index_version"):
        idx = source.index(f"async def {func_name}")
        block = source[:idx]
        start = block.rindex("@router.post")
        decorator = block[start:]
        assert "AdminCapability.ADMIN" in decorator, f"{func_name} 需要 ADMIN 能力"


def test_cross_tenant_cost_is_platform_only() -> None:
    """跨租户成本汇总必须显式拒绝非平台角色。

    `AdminCapability.ADMIN` 租户管理员也有（他能管自己租户的配置），
    但"全平台成本"不属于任何租户。所以接口体内必须再有
    `scope_tenant_id is not None` 的检查——光靠能力集不够。
    """
    source = (ADMIN_ROUTES / "dashboard.py").read_text(encoding="utf-8")
    idx = source.index("async def get_cross_tenant_cost")
    body = source[idx:]

    assert "scope_tenant_id is not None" in body, "缺少平台角色检查"
    assert "PermissionDeniedError" in body, "非平台角色时应 403"


def test_cross_tenant_cost_uses_bypass_rls() -> None:
    """跨租户聚合必须用 BYPASSRLS 会话。

    `model_calls` 在 `RLS_TABLES` 里。用带租户上下文的会话做聚合时，
    RLS 会把它压成单租户——**返回的数字看起来合理，只是漏了其它租户**。
    这类错误没有任何异常信号，只能靠结构检查防住。
    """
    source = (ADMIN_ROUTES / "dashboard.py").read_text(encoding="utf-8")
    idx = source.index("async def get_cross_tenant_cost")
    body = source[idx:]

    assert "transaction_bypass_rls" in body, "跨租户查询必须绕过 RLS"


def test_cross_tenant_cost_writes_audit() -> None:
    """跨租户看数必须留痕（谁在何时看了全平台成本）。

    这是唯一跨租户的读接口，绕过了租户隔离。没有审计就无法回答
    "谁把全平台的成本数据导走了"。
    """
    source = (ADMIN_ROUTES / "dashboard.py").read_text(encoding="utf-8")
    idx = source.index("async def get_cross_tenant_cost")
    body = source[idx:]

    assert "AuditRepository" in body
    assert "tenant_id=None" in body, "全局事件应写 tenant_id=None（不进任何租户链）"


# ==================================================== 契约约束


def test_activate_requires_confirm_version_tag() -> None:
    """切换/回滚必须二次确认 version_tag。

    URL 里已有 version_id，加 tag 看似冗余——但它防的是"前端把两个
    版本的 ID 搞反"（`.map` 里拿错字段很常见）。切换会让所有检索
    立刻改用新向量，值得一次显式确认。
    """
    from pydantic import ValidationError

    from rag_contracts.api.admin import (
        ActivateIndexVersionRequest,
        RollbackIndexVersionRequest,
    )

    with pytest.raises(ValidationError):
        ActivateIndexVersionRequest()  # type: ignore[call-arg]

    req = ActivateIndexVersionRequest(confirm_version_tag="v2-bge-m3")
    assert req.confirm_version_tag == "v2-bge-m3"

    with pytest.raises(ValidationError):
        RollbackIndexVersionRequest()  # type: ignore[call-arg]


def test_rebuild_request_rejects_extra_fields() -> None:
    """重建请求不接受额外字段（防止客户端偷偷指定 version_tag/模型）。

    `version_tag` 由服务端生成（租户内唯一），模型来自进程配置。
    放行这些字段会让"运营以为换了模型，实际配置没改"变成静默的
    错版本——版本表里记下的 model 与实际向量不符。
    """
    from pydantic import ValidationError

    from rag_contracts.api.admin import CreateIndexRebuildRequest

    CreateIndexRebuildRequest(reason="换模型")  # 合法
    with pytest.raises(ValidationError):
        CreateIndexRebuildRequest(version_tag="v9")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        CreateIndexRebuildRequest(embedding_model="other")  # type: ignore[call-arg]


# ==================================================== 辅助


@dataclass
class _TaskStub:
    """任务行替身（只需要 handler 实际读的字段）。"""

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    tenant_id: uuid.UUID = field(default_factory=uuid.uuid4)
    payload: dict[str, str] = field(default_factory=dict)
    locked_by: str = "w1"


class _FakeProvider:
    """最小 embedding provider 替身（本文件只测结构，不做真实调用）。"""

    name = "fake-embed"
    capabilities = type("Caps", (), {"dimension": 8})()

    async def embed_batch(self, texts: list[str], *, task: object) -> object:
        raise AssertionError("本测试不该真的调用 embedding")


async def _noop_progress(percent: int, stage: str = "") -> None:
    return None


def _never_opened_factory(tenant_id: uuid.UUID):
    """工厂替身：本测试不该开会话（handler 应在打开前就失败）。"""
    raise AssertionError("不该开会话")


# `ConflictError` / `PermissionDeniedError` 的语义在接口里用，这里
# 显式引用一次，避免 linter 把它们当未使用导入（同时说明它们是 HTTP 409/403 的来源）。
_ = (ConflictError, PermissionDeniedError)
