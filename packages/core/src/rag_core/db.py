"""PostgreSQL 连接层：异步引擎、会话工厂、事务上下文、健康检查。

为什么用 SQLAlchemy 2.0 async 而不是裸 asyncpg：
- A3 要写迁移（Alembic 依赖 SQLAlchemy 的 metadata 才能 autogenerate）；
- 分区表、RLS 策略、复杂 JOIN 用手写 SQL 维护成本高；
- 底层仍然是 asyncpg，全异步不受影响。

但仍然遵守"SQL 只在 repository" 的约束：业务代码只拿到 `AsyncSession`，
不直接建连接、不直接写 SQL（A3 起由 repository 承担）。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_core.errors import DependencyUnavailableError
from rag_core.logging import get_logger
from rag_core.settings import Settings

log = get_logger(__name__)


def build_engine(settings: Settings) -> AsyncEngine:
    """按配置建异步引擎。

    这里的每个参数都不是"默认值就能糊过去"的，各自对应一类生产事故：

    - **pool_size / max_overflow**：决定并发上来会不会把 PG 打爆。
      连接上限 = (pool_size + max_overflow) × 副本数，必须小于 PG 的
      max_connections，否则高峰期会有一批副本拿不到连接。
    - **pool_timeout**：池满之后等多久放弃。不设就是无限等，表现为
      请求全部挂住、网关超时，而不是快速失败。快速失败才能触发降级。
    - **pool_pre_ping**：取连接前先探活。没有它，PG 重启或网络闪断后
      池里全是死连接，第一批请求全部报错。
    - **pool_recycle**：连接最长存活时间。PG 或中间防火墙可能单方面
      断连，定期回收比事后排查省事。
    - **server_settings**：asyncpg 特有的会话参数注入点。statement_timeout
      是**会话级**参数，必须在建连时设；不设的话一个忘记加 LIMIT 的
      查询能把连接占住直到 TCP 超时，池很快耗尽。
    """
    return create_async_engine(
        settings.database.dsn.get_secret_value(),
        echo=False,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout_s,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={
            "server_settings": {
                "application_name": settings.app_name,
                "statement_timeout": str(settings.database.statement_timeout_ms),
                "lock_timeout": str(settings.database.lock_timeout_ms),
            }
        },
    )


class Database:
    """数据库资源持有者：引擎 + 会话工厂 + 生命周期。

    用类而不是模块级全局变量，原因有两条：
    1. 测试可以创建多个隔离实例，互不干扰；
    2. 避免"import 时就建连接池"——那会让 pytest 在收集阶段就连库，
       没有 PG 的环境（CI 的纯单测 job）会直接崩。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: AsyncEngine = build_engine(settings)
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            # 提交后仍可读对象属性，避免提交后访问属性触发额外查询
            expire_on_commit=False,
            # 关闭自动 flush：什么时候写库必须显式，否则一次无关的
            # 查询都可能把半成品数据刷进库，事务边界变得不可预测
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        """底层引擎。只给迁移脚本和测试用，业务代码不要碰。"""
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """会话工厂。给 FastAPI 依赖注入用。"""
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """自动提交的短会话。

        适合单条读取或独立写入，不跨多次操作。需要多语句原子性时用
        `transaction()`。

        注意：这里**不**用 `session.begin()`，所以不会自动开事务——
        SQLAlchemy 在每个语句上隐式开事务并需要显式 commit。
        为了不让人踩"忘了 commit 数据没写进去"，这里退出时统一 commit；
        异常时统一 rollback。
        """
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    @asynccontextmanager
    async def transaction(
        self, *, tenant_id: uuid.UUID | None = None
    ) -> AsyncIterator[AsyncSession]:
        """显式事务边界。

        用 `async with db.transaction() as session:` 包住的代码要么全成功
        要么全回滚。异常时**先回滚再抛**，不给连接留下未结束的事务
        ——那种状态会继续持锁阻塞别人。

        隔离级别保持默认的 READ COMMITTED。本项目的并发正确性靠
        乐观锁 + 唯一索引 + 原子 UPDATE 保证，不靠提高隔离级别；
        升到 REPEATABLE READ 会带来序列化失败重试，收益远小于成本。

        ## tenant_id 参数与 RLS

        传入 `tenant_id` 时，会在事务开始后立即执行
        `set_config('app.tenant_id', :t, true)`（第三个参数 true = 事务级），
        让数据库的行级安全策略生效。

        **必须用事务级设置，不能用会话级 `SET`**：
        连接池会复用连接，会话级设置会泄漏到下一个借用的请求 ——
        如果那个请求因为异常路径没重设，就会读到上一个租户的数据。
        这是真实可发生的越权，且极难复现（只在特定时机出现）。

        `SET LOCAL` / `set_config(..., true)` 在事务提交或回滚时
        自动清除，因此连接归还池后是干净的。

        **不传 tenant_id 时不做任何设置**：RLS 策略会因
        `current_setting('app.tenant_id', true)` 返回 NULL 而拒绝所有行。
        这对"忘了传"是最安全的默认（看不到数据，而不是看到全部）。
        需要跨租户的后台任务应显式使用 `transaction_bypass_rls()`。
        """
        async with self._session_factory() as session, session.begin():
            if tenant_id is not None:
                # 用 set_config 而不是 SET LOCAL：参数可绑定，避免 SQL 拼接。
                # 第三个参数 true 表示 is_local（事务级），等价于 SET LOCAL。
                await session.execute(
                    text("SELECT set_config('app.tenant_id', :tid, true)"),
                    {"tid": str(tenant_id)},
                )
            yield session

    @asynccontextmanager
    async def transaction_bypass_rls(self) -> AsyncIterator[AsyncSession]:
        """不设置租户上下文的事务，供合法的跨租户路径使用。

        典型场景：
        - 管理端看板统计所有租户；
        - 后台对账（扫全部租户的孤儿 blob）；
        - 数据迁移与修复。

        **这个方法的调用点必须能被审计追到**：它绕过了租户隔离，
        误用就是越权。建议只在 `rag_api.services.admin_*` 和
        `rag_worker.jobs.*` 里使用，且每次调用都写审计。

        注意：只有连接角色具备 BYPASSRLS 或不设策略时才真正"绕过"。
        普通角色（rag_app）在没有 app.tenant_id 时看到的是**零行**，
        不是全部行 —— 这是有意的安全默认。
        """
        # 显式设成一个不可能匹配的哨兵值？不——留空让策略拒绝所有行，
        # 需要跨租户时必须改用 rag_admin 角色（有 BYPASSRLS）。
        async with self._session_factory() as session, session.begin():
            yield session

    async def ping(self) -> None:
        """连通性检查，供 readiness 探针和启动检查使用。

        连不通时抛 `DependencyUnavailableError` 而不是返回 False：
        让所有调用方在异常路径上统一处理，不用每处判断布尔值。
        """
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:
            log.error("database_ping_failed", error=str(exc), exc_info=True)
            raise DependencyUnavailableError("数据库不可用") from exc

    async def server_version(self) -> str:
        """取 PG 版本，用于启动日志与排障。"""
        async with self._engine.connect() as conn:
            result = await conn.execute(text("SHOW server_version"))
            return str(result.scalar_one())

    async def installed_extensions(self) -> dict[str, str]:
        """列出已安装扩展。

        A3 建 chunks 表依赖 vector 扩展。启动时检查一次，能把
        "到 A3 才发现没装" 提前成 "第一次启动就说清楚缺什么"。
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("SELECT extname, extversion FROM pg_extension ORDER BY extname")
            )
            return {str(row[0]): str(row[1]) for row in result}

    async def close(self) -> None:
        """释放连接池。进程退出、测试清理时必须调用。"""
        await self._engine.dispose()


async def wait_until_ready(
    ping: Callable[[], Awaitable[None]],
    *,
    attempts: int = 10,
    delay_s: float = 0.5,
) -> None:
    """启动期重试等待依赖就绪。

    为什么需要：容器编排里 API 和 PG 同时启动，PG 至少要几秒才可连。
    不重试的话 API 首次启动就崩，只能靠重启碰运气。

    上限必须有限（attempts）：依赖真挂了要快速失败让编排器重启/告警，
    而不是无限等待，表现为"容器卡在启动中"没人知道为什么。
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            await ping()
            log.info("dependency_ready", attempt=attempt)
            return
        except Exception as exc:
            last_error = exc
            log.warning(
                "dependency_not_ready",
                attempt=attempt,
                attempts=attempts,
                error=str(exc),
            )
            if attempt < attempts:
                await asyncio.sleep(delay_s)

    raise DependencyUnavailableError(f"依赖在 {attempts} 次重试后仍不可用") from last_error
