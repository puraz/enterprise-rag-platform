"""集成测试共享 fixture。

## 为什么需要两个 fixture

启用 RLS 后，租户内表的每一次读写都要求会话里设置了 `app.tenant_id`。
测试因此分成两类：

- **`admin_session`**（超级用户）：准备测试数据、查断言。
  超级用户绕过 RLS，适合建租户/用户这类跨租户的准备工作。
- **`tenant_session`**：以 `rag_app` 角色 + 指定 tenant 上下文运行，
  用来**验证应用真实路径下的行为**（含 RLS 生效）。

之前所有测试都用"提交 + finally 清理"，踩到了两个坑：
1. 清理与新事务互锁，DELETE 静默失败，脏数据让后续测试莫名失败；
2. 测试之间靠"表里只有我的数据"这种隐含假设，环境一脏就挂。

现在用**外层事务 + 统一回滚**：测试怎么插都不留痕。
需要验证"真的提交了"或真并发的场景，自己管连接（见 test_task_models.py
的并发测试），不用这个 fixture。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_core.settings import get_settings


def _dsn(user: str, password: str) -> str:
    settings = get_settings()
    db = settings.database
    return f"postgresql+asyncpg://{user}:{password}@{db.host}:{db.port}/{db.name}"


@pytest.fixture(autouse=True)
def _use_migration_role_for_integration_tests(request: pytest.FixtureRequest):
    """让集成测试统一用高权限角色。

    多数集成测试直接 `Database(get_settings())` 建独立连接（为了测真并发），
    它们需要能跨租户准备数据 —— 这正是超级用户才能做的事。

    **只对 integration 标记的测试生效**：纯单测不连库，更不该被环境变量污染。
    之前无条件生效，导致 `Settings(env="prod", ...)` 这类直接构造的单测
    因为环境变量被改而失败（prod 守卫、密码断言全乱）。

    想验证 RLS 的测试必须绕开这个 fixture：用 `tenant_session`
    或 test_rls.py 里显式创建 rag_app 引擎。
    """
    if request.node.get_closest_marker("integration") is None:
        yield
        return

    settings = get_settings()
    db = settings.database
    if not db.migration_user:
        yield
        return

    import os

    saved: dict[str, str | None] = {
        "RAG_DATABASE__USER": os.environ.get("RAG_DATABASE__USER"),
        "RAG_DATABASE__PASSWORD": os.environ.get("RAG_DATABASE__PASSWORD"),
    }
    os.environ["RAG_DATABASE__USER"] = db.migration_user
    if db.migration_password is not None:
        os.environ["RAG_DATABASE__PASSWORD"] = db.migration_password.get_secret_value()
    get_settings.cache_clear()

    yield

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """**高权限**测试会话，结束时自动回滚。

    用迁移角色（`migration_user`，通常是超级用户）连接：
    超级用户绕过 RLS，可以直接读写任何表 —— 适合准备数据与断言结果。

    **注意**：用这个 fixture 的测试**验证不了 RLS**。
    要验证 RLS 必须用 `tenant_session` 或 `test_rls.py` 里的 rag_app 连接。

    用法：
        async def test_xxx(db_session: AsyncSession) -> None:
            repo = TenantRepository(db_session)
            await repo.create(slug="t1", name="测试")
            # 测试结束自动回滚
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    settings = get_settings()
    db = settings.database
    user = db.migration_user or db.user
    password = (
        db.migration_password.get_secret_value()
        if db.migration_password is not None
        else db.password.get_secret_value()
    )

    engine = create_async_engine(_dsn(user, password), poolclass=None)
    connection = await engine.connect()
    transaction = await connection.begin()
    session = async_sessionmaker(bind=connection, expire_on_commit=False, autoflush=False)()
    try:
        yield session
    finally:
        await session.close()
        await transaction.rollback()
        await connection.close()
        await engine.dispose()


@pytest.fixture
async def tenant_session() -> AsyncIterator[tuple[AsyncSession, uuid.UUID]]:
    """**应用角色 + tenant 上下文**的会话，用于验证 RLS 生效路径。

    返回 `(session, tenant_id)`。测试里应先用 `db_session` 建好数据，
    再用本 fixture 验证"应用角色能看到什么"。

    会话开始时执行 `set_config('app.tenant_id', :t, true)`（事务级），
    与生产代码 `Database.transaction(tenant_id=...)` 的行为一致。
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    settings = get_settings()
    db = settings.database
    engine = create_async_engine(_dsn(db.user, db.password.get_secret_value()), poolclass=None)
    connection = await engine.connect()
    transaction = await connection.begin()

    tenant_id = uuid.uuid4()
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant_id)}
    )

    session = async_sessionmaker(bind=connection, expire_on_commit=False, autoflush=False)()
    try:
        yield session, tenant_id
    finally:
        await session.close()
        await transaction.rollback()
        await connection.close()
        await engine.dispose()
