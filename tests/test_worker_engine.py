"""Worker 引擎测试。

## 分两层

**纯单测**（无外部依赖，`make test` 会跑）：
- 失败分类 → 重试/死信的决策；
- 进度里程碑节流；
- 错误消息脱敏与截断；
- 无 handler 时进死信。

**并发专项**（`-m concurrency`，连真实 PG）：
- 两个 worker 同时认领，同一任务只会被一个拿到；
- 崩溃 worker 遗留的 running 任务会被回收。

## 为什么并发测试不能用 mock

`FOR UPDATE SKIP LOCKED` 的正确性**完全由数据库实现**。
用 mock 验证"我调用了带 skip_locked 的方法"只能证明我写对了 SQL 字符串，
证明不了两个事务并发时行为正确。

这类测试必须开两个真实连接、真并发执行。A3 已经用同样的方式
验证过版本号分配（当时发现了 8 个并发请求拿到同一个版本号的 bug）。
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from rag_contracts.enums import TaskStatus
from rag_core.errors import DependencyUnavailableError, ValidationFailedError
from rag_core.settings import Settings
from rag_core.settings.worker import WorkerSettings
from rag_worker.engine import ProgressReporter, WorkerEngine, _error_code_of, _safe_message

# ==================================================== 纯单测：错误码与消息


def test_error_code_uses_code_name_not_number() -> None:
    """错误码进任务表时用语义名。

    `tasks.error_code` 是给人看的（运维在死信列表里扫）。
    `31005` 对人不友好，`DOC_PARSE_FAILED` 才有意义。
    """
    # 码名格式是 `域_原因`（枚举成员名），不是 `域.原因`：
    # 那是 ErrorCode 的 **value** 格式，成员名把点换成了下划线。
    assert _error_code_of(ValidationFailedError("bad")) == "COMMON_VALIDATION_FAILED"


def test_unknown_exception_code_is_prefixed() -> None:
    """未识别异常加 `UNHANDLED_` 前缀。

    看到这个前缀就知道"有一条错误路径没被分类覆盖"，
    比混在业务错误码里更容易被发现。
    """
    assert _error_code_of(RuntimeError("boom")) == "UNHANDLED_RuntimeError"


def test_safe_message_redacts_pii() -> None:
    """错误消息必须脱敏：异常里常带用户数据。

    `raise ValueError(f"用户 {phone} 的文件损坏")` 是很自然的写法，
    但这条消息会进数据库、进日志、进管理端页面。
    """
    exc = ValueError("用户 13812345678 的文档解析失败")
    message = _safe_message(exc)
    assert "13812345678" not in message
    assert "138" in message  # 保留前 3 位便于核对


def test_safe_message_truncates() -> None:
    """截断到 200 字符：超长消息（含整个 SQL 或堆栈）会撑爆字段。"""
    message = _safe_message(ValueError("x" * 5000))
    assert len(message) <= 200


# ==================================================== 纯单测：进度节流


class _RecordingSessionFactory:
    """记录进度写入次数的假会话工厂。"""

    def __init__(self) -> None:
        self.writes: list[int] = []

    def __call__(self) -> Any:
        return self

    async def __aenter__(self) -> _RecordingSession:
        return _RecordingSession(self)

    async def __aexit__(self, *args: object) -> None:
        return None


class _RecordingSession:
    def __init__(self, factory: _RecordingSessionFactory) -> None:
        self._factory = factory

    async def commit(self) -> None:
        return None


@pytest.fixture
def recording_factory() -> Any:
    factory = _RecordingSessionFactory()

    # 替换仓储层的 update_progress：只记录调用，不碰数据库
    import rag_worker.engine as engine_module

    class _Repo:
        def __init__(self, session: Any) -> None: ...

        async def update_progress(
            self, *, task_id: uuid.UUID, worker_id: str, progress: int
        ) -> bool:
            factory.writes.append(progress)
            return True

    original = engine_module.TaskRepository
    engine_module.TaskRepository = _Repo  # type: ignore[misc]
    yield factory
    engine_module.TaskRepository = original


async def test_progress_writes_only_at_milestones(recording_factory: Any) -> None:
    """只在跨越里程碑时写库。

    这是长任务不把连接池占满的关键：300 页 PDF 如果每页写一次进度，
    就是 300 次 UPDATE；里程碑方案固定 7 次。
    """
    reporter = ProgressReporter(recording_factory, task_id=uuid.uuid4(), worker_id="w1")

    for percent in range(0, 101):
        await reporter(percent, "parsing")

    # 里程碑是"闸门"，写进去的是**真实百分比**（更准确）。
    # 95 是最后一个闸门；100 被夹到 99，而 99 没有跨过任何新闸门，
    # 所以最后一次上报不写库 —— 100% 只由 complete() 设置。
    assert recording_factory.writes == [10, 25, 40, 55, 70, 85, 95]
    assert len(recording_factory.writes) == 7, "101 次上报只应产生 7 次写库"


async def test_progress_does_not_write_twice_for_same_milestone(
    recording_factory: Any,
) -> None:
    """同一里程碑重复上报不重复写。

    解析器可能是按页回调的，`progress(70)` 会被调用很多次；
    每次都写会让节流失效。
    """
    reporter = ProgressReporter(recording_factory, task_id=uuid.uuid4(), worker_id="w1")

    await reporter(70, "persisting")
    await reporter(70, "persisting")
    await reporter(71, "persisting")

    assert recording_factory.writes == [70]


async def test_progress_clamps_out_of_range(recording_factory: Any) -> None:
    """越界百分比被夹到 0-100，不会写进负数或 150。

    防止解析器的估算 bug 变成数据库里的脏数据
    （前端进度条显示 150% 会很难解释）。
    """
    reporter = ProgressReporter(recording_factory, task_id=uuid.uuid4(), worker_id="w1")

    await reporter(-5, "init")
    await reporter(150, "done")

    # -5 被夹到 0（不写）；150 被夹到 99（跨过 95 这个闸门，写入 99）
    assert recording_factory.writes == [99]


async def test_progress_never_reports_100(recording_factory: Any) -> None:
    """**handler 不能把进度报成 100**。

    `progress=100` 的语义是"任务已成功结束"，只有 `complete()` 能设。
    handler 报 100 会让仍在运行（甚至马上要失败）的任务在前端显示"已完成"，
    而任务随后失败时，用户看到的是"100% 之后变成失败"——很像系统 bug。
    """
    reporter = ProgressReporter(recording_factory, task_id=uuid.uuid4(), worker_id="w1")

    await reporter(100, "done")

    assert 100 not in recording_factory.writes


# ==================================================== 纯单测：失败分类决策


class _FakeTask:
    """最小任务对象（handler/engine 只用到这几个属性）。"""

    def __init__(
        self,
        *,
        task_type: str = "parse_document",
        retry_count: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.id = uuid.uuid4()
        self.tenant_id = uuid.uuid4()
        self.task_type = task_type
        self.retry_count = retry_count
        self.max_retries = 3
        self.payload = payload or {}


def _build_engine(
    *,
    handlers: dict[str, Any] | None = None,
    fail_calls: list[dict[str, Any]] | None = None,
    progress_calls: list[int] | None = None,
) -> WorkerEngine:
    """构造只依赖内存状态的引擎，用于测试决策逻辑。"""
    settings = Settings()
    worker_settings = WorkerSettings(worker_id="test-worker", max_concurrency=1)
    return WorkerEngine(
        settings=settings,
        worker_settings=worker_settings,
        handlers=handlers or {},
        task_session_factory=_FakeSessionFactory(fail_calls=fail_calls),
        tenant_session_factory=lambda _t: _FakeSessionFactory(),
    )


class _FakeSessionFactory:
    def __init__(self, fail_calls: list[dict[str, Any]] | None = None) -> None:
        self._fail_calls = fail_calls

    def __call__(self) -> Any:
        return self

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def commit(self) -> None:
        return None


async def test_permanent_failure_goes_to_dead_letter_without_retry() -> None:
    """永久失败直接进死信，不重试。

    这是 A8 的关键决策：文件损坏/格式不支持重试 3 次还是同样的结果，
    白占 3 个 worker 槽位几分钟，还让用户多等三轮退避。
    """
    import rag_worker.engine as engine_module

    recorded: list[dict[str, Any]] = []

    class _Repo:
        def __init__(self, session: Any) -> None: ...

        async def fail(self, **kwargs: Any) -> str:
            recorded.append(kwargs)
            return TaskStatus.DEAD.value

    async def handler(**_: Any) -> None:
        raise ValidationFailedError("payload 非法")

    engine = _build_engine(handlers={"parse_document": handler})
    original = engine_module.TaskRepository
    engine_module.TaskRepository = _Repo  # type: ignore[misc]
    try:
        await engine._execute(_FakeTask())
    finally:
        engine_module.TaskRepository = original

    assert len(recorded) == 1
    assert recorded[0]["retry"] is False, "永久失败不该被重试"
    assert recorded[0]["backoff_s"] == 0
    assert engine.stats.dead == 1


async def test_transient_failure_is_retried_with_backoff() -> None:
    """瞬时失败重试，且退避时间随重试次数指数增长。"""
    import rag_worker.engine as engine_module

    recorded: list[dict[str, Any]] = []

    class _Repo:
        def __init__(self, session: Any) -> None: ...

        async def fail(self, **kwargs: Any) -> str:
            recorded.append(kwargs)
            return TaskStatus.FAILED.value

    async def handler(**_: Any) -> None:
        raise DependencyUnavailableError("存储超时")

    engine = _build_engine(handlers={"parse_document": handler})
    original = engine_module.TaskRepository
    engine_module.TaskRepository = _Repo  # type: ignore[misc]
    try:
        await engine._execute(_FakeTask(retry_count=0))
        await engine._execute(_FakeTask(retry_count=2))
    finally:
        engine_module.TaskRepository = original

    assert recorded[0]["retry"] is True
    assert recorded[0]["backoff_s"] == 30  # 30 * 2^0
    assert recorded[1]["backoff_s"] == 120  # 30 * 2^2


async def test_backoff_is_capped() -> None:
    """退避有上限：第 10 次重试不该等 8 小时。"""
    import rag_worker.engine as engine_module

    recorded: list[dict[str, Any]] = []

    class _Repo:
        def __init__(self, session: Any) -> None: ...

        async def fail(self, **kwargs: Any) -> str:
            recorded.append(kwargs)
            return TaskStatus.FAILED.value

    async def handler(**_: Any) -> None:
        raise DependencyUnavailableError("down")

    engine = _build_engine(handlers={"parse_document": handler})
    original = engine_module.TaskRepository
    engine_module.TaskRepository = _Repo  # type: ignore[misc]
    try:
        await engine._execute(_FakeTask(retry_count=10))
    finally:
        engine_module.TaskRepository = original

    assert recorded[0]["backoff_s"] == 600, "退避应被截到 10 分钟"


async def test_missing_handler_goes_to_dead_letter() -> None:
    """没有 handler 是配置错误，进死信而不是无限重试。"""
    import rag_worker.engine as engine_module

    recorded: list[dict[str, Any]] = []

    class _Repo:
        def __init__(self, session: Any) -> None: ...

        async def fail(self, **kwargs: Any) -> str:
            recorded.append(kwargs)
            return TaskStatus.DEAD.value

    engine = _build_engine(handlers={})  # 空注册表
    original = engine_module.TaskRepository
    engine_module.TaskRepository = _Repo  # type: ignore[misc]
    try:
        await engine._execute(_FakeTask(task_type="rebuild_index"))
    finally:
        engine_module.TaskRepository = original

    assert len(recorded) == 1
    assert recorded[0]["retry"] is False
    assert recorded[0]["error_code"] == "INTERNAL.NOT_IMPLEMENTED"
    assert "rebuild_index" in recorded[0]["error_message"]


async def test_handler_receives_tenant_session_and_progress() -> None:
    """handler 拿到的是租户会话，且 progress 可调用。

    租户会话很关键：handler 里的业务读写必须走 RLS 生效的会话，
    不能走 admin——那等于把隔离撤掉。
    """
    import rag_worker.engine as engine_module

    seen: dict[str, Any] = {}

    class _Repo:
        def __init__(self, session: Any) -> None: ...

        async def complete(self, *, task_id: uuid.UUID, worker_id: str) -> bool:
            return True

        async def update_progress(
            self, *, task_id: uuid.UUID, worker_id: str, progress: int
        ) -> bool:
            seen.setdefault("progress", []).append(progress)
            return True

    async def handler(*, session: Any, task: Any, progress: Any) -> None:
        seen["session"] = session
        await progress(50, "half")

    engine = _build_engine(handlers={"parse_document": handler})
    original = engine_module.TaskRepository
    engine_module.TaskRepository = _Repo  # type: ignore[misc]
    try:
        await engine._execute(_FakeTask())
    finally:
        engine_module.TaskRepository = original

    assert seen["session"] is not None
    # 50 跨过 40 这个闸门 → 写一次，写入值是真实的 50
    assert seen["progress"] == [50]
    assert engine.stats.succeeded == 1


# ==================================================== 并发专项（真实 PG）


@pytest.mark.integration
@pytest.mark.concurrency
async def test_concurrent_claim_never_returns_same_task_twice() -> None:
    """**A8-2 的核心验收点**：两个 worker 并发认领，同一任务只被一个拿到。

    为什么必须用真实数据库：`FOR UPDATE SKIP LOCKED` 的语义由 PG 实现。
    mock 只能证明"我写了 skip_locked 这个参数"，证明不了并发行为。

    测试方法：建 N 个任务，两个独立引擎各自认领，断言：
    1. 两边拿到的任务 ID 集合不重叠；
    2. 任务总数不超过 N（不会凭空多出任务）。
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from rag_core.settings import get_settings
    from rag_db.repositories import TaskRepository

    settings = get_settings()
    db = settings.database
    user = db.migration_user or db.user
    password = (
        db.migration_password.get_secret_value()
        if db.migration_password is not None
        else db.password.get_secret_value()
    )
    dsn = f"postgresql+asyncpg://{user}:{password}@{db.host}:{db.port}/{db.name}"

    # 独立引擎（不用事务回滚的 fixture）：要测真并发的连接隔离。
    # 用 migration 角色绕开 RLS —— 任务认领在生产里走 admin 连接，
    # 这里用同一个原理（都是 BYPASSRLS 语义）。
    engine = create_async_engine(dsn, pool_size=4)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    tenant_id = uuid.uuid4()
    await _create_tenant(factory, tenant_id=tenant_id)
    try:
        task_ids = await _create_tasks(factory, tenant_id=tenant_id, count=6)

        # 两个"worker"并发认领。
        #
        # **必须限定 task_type**：`claim(task_types=None)` 会认领**所有租户**
        # 的所有 pending 任务（worker 的真实行为——它要处理全平台的队列）。
        # 测试里不限定就会把别的测试遗留的 pending 任务一并抢走，
        # 表现为"隔离跑通过、全量跑失败"的假 flake。
        #
        # 用本测试独有的 task_type 做隔离，比"假设库里没有别人的任务"可靠。
        isolation_type = f"parse_document_test_{uuid.uuid4().hex[:8]}"

        async def claim(worker_id: str) -> list[uuid.UUID]:
            async with factory() as session:
                tasks = await TaskRepository(session).claim(
                    worker_id=worker_id,
                    limit=3,
                    task_types=[isolation_type],
                )
                await session.commit()
                return [t.id for t in tasks]

        # 把本测试的任务改成隔离类型
        from sqlalchemy import text as sql_text

        async with factory() as session:
            await session.execute(
                sql_text("UPDATE tasks SET task_type = :tt WHERE tenant_id = :tid"),
                {"tt": isolation_type, "tid": tenant_id},
            )
            await session.commit()

        first, second = await asyncio.gather(claim("worker-a"), claim("worker-b"))

        assert len(first) == 3
        assert len(second) == 3
        assert set(first).isdisjoint(second), "同一个任务被两个 worker 同时认领了"
        assert set(first) | set(second) == set(task_ids), "有任务既没被 A 也没被 B 认领"
    finally:
        await _cleanup(factory, tenant_id=tenant_id)
        await engine.dispose()


@pytest.mark.integration
@pytest.mark.concurrency
async def test_stale_running_task_is_reclaimed() -> None:
    """崩溃 worker 留下的 running 任务会被回收。

    不回收的话任务永远停在 running：文档一直显示"处理中"，
    队列却是空的——没有报错，最难排查的一类故障。
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from rag_core.settings import get_settings
    from rag_db.repositories import TaskRepository

    settings = get_settings()
    db = settings.database
    user = db.migration_user or db.user
    password = (
        db.migration_password.get_secret_value()
        if db.migration_password is not None
        else db.password.get_secret_value()
    )
    dsn = f"postgresql+asyncpg://{user}:{password}@{db.host}:{db.port}/{db.name}"

    engine = create_async_engine(dsn, pool_size=4)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    tenant_id = uuid.uuid4()
    await _create_tenant(factory, tenant_id=tenant_id)
    try:
        await _create_tasks(factory, tenant_id=tenant_id, count=1)

        # worker 认领后"崩溃"（不提交完成，连接断开）
        async with factory() as session:
            claimed = await TaskRepository(session).claim(worker_id="crashed", limit=1)
            await session.commit()
        task_id = claimed[0].id

        # 把 locked_at 推到很久以前，模拟锁超时
        async with factory() as session:
            await session.execute(
                text("UPDATE tasks SET locked_at = now() - interval '2 hours' WHERE id = :id"),
                {"id": task_id},
            )
            await session.commit()

        # 回收：lock_timeout_s=60 → 2 小时前的锁必然过期
        async with factory() as session:
            reclaimed = await TaskRepository(session).reclaim_stale(lock_timeout_s=60)
            await session.commit()

        assert reclaimed == 1

        # 回收后应回到 pending 且 retry_count +1（避免无限重派毒任务）
        async with factory() as session:
            row = await session.execute(
                text("SELECT status, retry_count FROM tasks WHERE id = :id"),
                {"id": task_id},
            )
            status, retry_count = row.one()
        assert status == TaskStatus.PENDING.value
        assert retry_count == 1, "回收必须递增重试次数，否则毒任务会被无限重派"
    finally:
        await _cleanup(factory, tenant_id=tenant_id)
        await engine.dispose()


@pytest.mark.integration
async def test_progress_write_requires_ownership() -> None:
    """进度只能由认领者写。

    任务被回收重派给新 worker 后，旧 worker 如果还能写进度，
    前端会看到"进度来回跳"——两个 worker 在同一任务上互相覆盖。
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from rag_core.settings import get_settings
    from rag_db.repositories import TaskRepository

    settings = get_settings()
    db = settings.database
    user = db.migration_user or db.user
    password = (
        db.migration_password.get_secret_value()
        if db.migration_password is not None
        else db.password.get_secret_value()
    )
    dsn = f"postgresql+asyncpg://{user}:{password}@{db.host}:{db.port}/{db.name}"

    engine = create_async_engine(dsn, pool_size=4)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    tenant_id = uuid.uuid4()
    await _create_tenant(factory, tenant_id=tenant_id)
    try:
        await _create_tasks(factory, tenant_id=tenant_id, count=1)

        async with factory() as session:
            claimed = await TaskRepository(session).claim(worker_id="owner", limit=1)
            await session.commit()
        task_id = claimed[0].id

        async with factory() as session:
            repo = TaskRepository(session)
            # 正确的 owner 能写
            assert await repo.update_progress(task_id=task_id, worker_id="owner", progress=40)
            # 冒充者不能写
            assert not await repo.update_progress(
                task_id=task_id, worker_id="impostor", progress=90
            )
            await session.commit()
    finally:
        await _cleanup(factory, tenant_id=tenant_id)
        await engine.dispose()


# ---------------------------------------------------------------- 测试数据辅助


async def _create_tenant(factory: Any, *, tenant_id: uuid.UUID) -> None:
    """建测试租户。

    **不需要建 user**：`tasks` 表没有 created_by 列（任务属于系统操作，
    不记人），租户是它唯一的外部依赖（`TenantScopedMixin` 的 FK）。

    用户表是**全局的**（无 tenant_id）——"谁在这个租户里"由
    `tenant_memberships` 表达。这是 A3 的设计，不是这里取巧。
    """
    from sqlalchemy import text

    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:id, :slug, :name, 'active', now(), now())"
            ),
            {"id": tenant_id, "slug": f"t-{tenant_id.hex[:8]}", "name": "worker 测试租户"},
        )
        await session.commit()


async def _create_tasks(factory: Any, *, tenant_id: uuid.UUID, count: int) -> list[uuid.UUID]:
    """建 N 个 pending 任务，返回 ID 列表。"""
    from sqlalchemy import text

    ids: list[uuid.UUID] = []
    async with factory() as session:
        for _index in range(count):
            task_id = uuid.uuid4()
            ids.append(task_id)
            await session.execute(
                text(
                    "INSERT INTO tasks (id, tenant_id, task_type, status, "
                    "idempotency_key, payload, progress, retry_count, max_retries, "
                    "available_at, created_at, updated_at, version) "
                    "VALUES (:id, :tid, 'parse_document', 'pending', :key, "
                    "'{}'::jsonb, 0, 0, 3, now(), now(), now(), 1)"
                ),
                {
                    "id": task_id,
                    "tid": tenant_id,
                    "key": f"test:{task_id.hex}",
                },
            )
        await session.commit()
    return ids


async def _cleanup(factory: Any, *, tenant_id: uuid.UUID) -> None:
    """清理测试数据（按外键顺序）。

    不用事务回滚是因为并发测试需要真提交。所以必须显式清干净，
    否则残留的租户会让下次跑测试撞唯一约束。
    """
    from sqlalchemy import text

    async with factory() as session:
        await session.execute(text("DELETE FROM tasks WHERE tenant_id = :tid"), {"tid": tenant_id})
        await session.execute(text("DELETE FROM tenants WHERE id = :tid"), {"tid": tenant_id})
        await session.commit()
