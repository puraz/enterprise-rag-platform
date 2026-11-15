"""Alembic 运行环境（异步）。

三个关键决策：

1. **DSN 从 rag_core.Settings 读，不写进 alembic.ini**。
   ini 里的 sqlalchemy.url 会成为"第二份连接配置"，与 .env 漂移后
   迁移可能连到与 API 不同的库——这是最难排查的一类事故（迁移显示成功，
   但业务表不在）。

2. **用 async 引擎**。我们的 DSN 是 postgresql+asyncpg://，Alembic 默认的
   同步环境会直接报错 "The asyncio extension requires an async driver"。
   官方 async 模板的做法是包一层 asyncio.run。

3. **迁移连接用 NullPool**。迁移是一次性短生命周期操作，用连接池没有收益，
   而且池里的连接在 asyncio.run 结束时可能没被干净关闭，留下警告或
   连接泄漏。
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from rag_core.settings import get_settings
from rag_db.base import Base
from rag_db.models import *  # noqa: F403  # 必须导入所有模型，autogenerate 才能看到表

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# autogenerate 的对照源：所有继承 Base 的模型都在这里
target_metadata = Base.metadata


def _database_url() -> str:
    """从 Settings 取 DSN，而不是从 ini。

    每次调用都重新取，保证 `make migrate` 与 API 用的是同一份配置。
    """
    # 用高权限 DSN：迁移需要 CREATE/ALTER/POLICY 权限，
    # 而运行时角色 rag_app 只有 DML 权限（最小权限原则）。
    return get_settings().database.migration_dsn.get_secret_value()


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 文本，不连库。

    用途：给 DBA 审阅、或在无法直连生产库时由人工执行。
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # 变更对比时忽略这些 schema，避免把扩展自带的表纳入比较
        include_schemas=False,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在已建立的连接上执行迁移。"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,  # 列类型变化也生成迁移（默认不看类型变化）
        compare_server_default=True,  # 默认值变化也生成
        # render_as_batch=False：PG 支持大部分 ALTER，不需要批处理模式
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """异步执行迁移。"""
    connectable = create_async_engine(
        _database_url(),
        poolclass=pool.NullPool,  # 一次性操作，不需要连接池
    )

    async with connectable.connect() as connection:
        # Alembic 的 context 是同步 API，用 run_sync 桥接
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """在线模式入口。"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
