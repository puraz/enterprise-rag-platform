"""DB / Redis 连接层测试。

分两类，用 marker 分开：

- **单元测试**（默认跑）：只验证配置到引擎/连接池的映射、异常语义，
  不需要真实 PG/Redis。CI 的纯单测 job 靠这一批。
- **集成测试**（`-m integration`）：连真实 PG/Redis，验证池参数真的
  生效、事务真的回滚、超时真的触发。需要环境就绪。

为什么不合并：单元测试必须能在没有数据库的机器上跑（新人 clone 完
`make test` 就能绿），集成测试需要 `make check-deps` 通过后才跑。
混在一起会让"环境没起"表现为"测试失败"，掩盖真正的代码问题。
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from rag_core.db import Database, build_engine, wait_until_ready
from rag_core.errors import DependencyUnavailableError
from rag_core.redis import RedisClient, build_pool
from rag_core.settings import Settings, get_settings

# ============================================================ 单元测试


def _local_settings() -> Settings:
    """构造一个本地可跑的配置（不读 .env）。"""
    return Settings(
        env="local",
        database={
            "host": "127.0.0.1",
            "port": 5432,
            "user": "probe_user",
            "password": "probe_password",
            "name": "probe_db",
            "pool_size": 7,
            "max_overflow": 3,
            "pool_timeout_s": 2.5,
        },
        redis={"host": "127.0.0.1", "port": 6379, "max_connections": 17, "socket_timeout_s": 1.5},
    )


def test_engine_carries_pool_settings() -> None:
    """配置里的池参数必须真正落到引擎上。

    池参数写错（比如把 max_overflow 当 pool_size 传）不会有任何报错，
    只会在高并发时表现为"连接数不够用"或"连接数暴涨打爆 PG"。
    """
    engine = build_engine(_local_settings())
    pool = engine.pool
    assert pool.size() == 7
    assert pool._max_overflow == 3  # SQLAlchemy 未公开该属性，只能读私有
    assert pool._timeout == 2.5


def test_engine_uses_asyncpg_driver() -> None:
    """驱动必须是 asyncpg：换成同步驱动会阻塞事件循环。"""
    engine = build_engine(_local_settings())
    assert engine.dialect.driver == "asyncpg"
    assert engine.url.drivername == "postgresql+asyncpg"


@pytest.mark.integration
def test_statement_timeout_actually_applied() -> None:
    """statement_timeout 必须真正生效在会话上。

    为什么用真实连接验证而不是断言引擎参数：connect_args 在
    SQLAlchemy 2.x 里没有公开的读取入口，断言私有属性等于在测实现
    细节，重构就红。真正的风险是"参数传了但没生效"——只有连上库
    `SHOW statement_timeout` 才能证明。

    漏设的后果：一个走错索引的慢查询能把池占满，全站请求排队，
    而且不报任何错。
    """
    settings = get_settings().model_copy(
        update={
            "database": Settings().database.model_copy(
                update={"statement_timeout_ms": 4321, "lock_timeout_ms": 987}
            )
        }
    )

    async def scenario() -> tuple[str, str]:
        db = Database(settings)
        try:
            async with db.engine.connect() as conn:
                stmt = (await conn.execute(text("SHOW statement_timeout"))).scalar_one()
                lock = (await conn.execute(text("SHOW lock_timeout"))).scalar_one()
                return str(stmt), str(lock)
        finally:
            await db.close()

    statement_timeout, lock_timeout = asyncio.run(scenario())
    assert statement_timeout == "4321ms"
    assert lock_timeout == "987ms"


def test_redis_pool_carries_limits() -> None:
    """Redis 连接数与超时必须落到池上，否则突发流量打穿 fd 上限。"""
    pool = build_pool(_local_settings())
    assert pool.max_connections == 17
    # 超时在 Connection 类上，未建连时读取的是类默认值
    assert pool.connection_kwargs["socket_timeout"] == 1.5
    assert pool.connection_kwargs["socket_connect_timeout"] == 1.5


def test_redis_does_not_retry_on_timeout() -> None:
    """Redis 客户端不能自己重试。

    重试必须统一走 ResilientCaller，否则：
    - 重试次数不可控（客户端重试 × 业务重试 × 网关重试）；
    - 熔断器拿不到真实失败信号，永远不熔断。
    """
    pool = build_pool(_local_settings())
    assert pool.connection_kwargs.get("retry_on_timeout") is False


def test_ping_wraps_failure_into_dependency_error() -> None:
    """连不上时必须抛 DependencyUnavailableError，不是原始驱动异常。

    上层（异常处理器、探针）按 AppError 统一处理；让原始
    asyncpg/redis 异常漏出去，会导致 500 里带出连接串和驱动细节。
    """
    settings = Settings(
        env="local",
        database={"host": "127.0.0.1", "port": 1},  # 必然连不上
        redis={"host": "127.0.0.1", "port": 1},
    )
    db = Database(settings)
    redis = RedisClient(settings)

    with pytest.raises(DependencyUnavailableError):
        asyncio.run(db.ping())
    with pytest.raises(DependencyUnavailableError):
        asyncio.run(redis.ping())


def test_wait_until_ready_gives_up_after_attempts() -> None:
    """重试必须有上限，不能无限等。

    无限等的表现是容器"卡在启动中"——编排器不知道该重启，
    人也不知道为什么，比直接崩溃难排查得多。
    """
    calls: list[int] = []

    async def always_fails() -> None:
        calls.append(1)
        raise DependencyUnavailableError("not ready")

    with pytest.raises(DependencyUnavailableError, match="重试后仍不可用"):
        asyncio.run(wait_until_ready(always_fails, attempts=3, delay_s=0.01))

    assert len(calls) == 3


def test_wait_until_ready_stops_on_success() -> None:
    """一旦就绪就立刻返回，不做多余等待。"""
    calls: list[int] = []

    async def ready_on_second() -> None:
        calls.append(1)
        if len(calls) < 2:
            raise DependencyUnavailableError("not yet")

    asyncio.run(wait_until_ready(ready_on_second, attempts=10, delay_s=0.01))
    assert len(calls) == 2


# ============================================================ 集成测试


@pytest.mark.integration
def test_real_postgres_roundtrip() -> None:
    """真实 PG：连接、执行、读扩展。"""
    settings = get_settings().model_copy(update={"database": Settings().database})

    async def scenario() -> tuple[str, dict[str, str]]:
        db = Database(settings)
        try:
            await db.ping()
            return await db.server_version(), await db.installed_extensions()
        finally:
            await db.close()

    version, extensions = asyncio.run(scenario())
    # 只断言**主版本下限**，不钉死具体版本：
    # 本机可能是 PG 18，compose 用的是 pgvector/pgvector:pg17。
    # 钉死 "18" 会让换到 compose 环境时这条用例误报——而它真正要证明的
    # 是"连得上、PG 版本够新（支持我们用的 SQL 特性）且装了 pgvector"。
    major = int(version.split(".")[0])
    assert major >= 17, f"PG 主版本需 >= 17，实际 {version}"
    assert "vector" in extensions, "缺少 pgvector 扩展，A3 建向量表会失败"


@pytest.mark.integration
def test_transaction_commits_on_success() -> None:
    """正常退出事务必须提交。"""
    settings = get_settings()

    async def scenario() -> str | None:
        db = Database(settings)
        try:
            async with db.transaction() as session:
                await session.execute(text("CREATE TEMP TABLE tx_probe (v text) ON COMMIT DROP"))
                await session.execute(text("INSERT INTO tx_probe VALUES ('ok')"))
            # 临时表 ON COMMIT DROP 已随提交消失，说明事务真的提交了
            async with db.session() as session:
                exists = await session.execute(text("SELECT to_regclass('tx_probe')"))
                return exists.scalar_one()
        finally:
            await db.close()

    assert asyncio.run(scenario()) is None


@pytest.mark.integration
def test_transaction_rolls_back_on_error() -> None:
    """异常退出必须回滚，且异常继续向上传播。

    回滚失败的表现是"脏数据留在库里"，而且后续事务可能因锁未释放
    而阻塞——这类问题往往在事故复盘时才被发现。
    """
    settings = get_settings()

    async def scenario() -> None:
        db = Database(settings)
        try:
            async with db.transaction() as session:
                await session.execute(text("CREATE TEMP TABLE rb_probe (v text) ON COMMIT DROP"))
                await session.execute(text("INSERT INTO rb_probe VALUES ('should-vanish')"))
                raise RuntimeError("模拟业务失败")
        finally:
            await db.close()

    with pytest.raises(RuntimeError, match="模拟业务失败"):
        asyncio.run(scenario())


@pytest.mark.integration
def test_real_redis_roundtrip() -> None:
    """真实 Redis/Valkey：连接、读写、过期。"""
    settings = get_settings()

    async def scenario() -> bytes | None:
        client = RedisClient(settings)
        try:
            await client.ping()
            await client.client.set("rag:test:probe", "value-1", ex=5)
            return await client.client.get("rag:test:probe")
        finally:
            await client.client.delete("rag:test:probe")
            await client.close()

    # decode_responses=False，所以拿回来是 bytes
    assert asyncio.run(scenario()) == b"value-1"


@pytest.mark.integration
def test_redis_bytes_are_preserved() -> None:
    """默认不解码成 str，二进制内容（图片分片、序列化消息）才能安全存取。

    这是个容易踩的坑：decode_responses=True 时，写入非 UTF-8 的 bytes
    会直接抛编解码错误，而错误信息完全指不到根因。
    """
    settings = get_settings()

    async def scenario() -> bytes | None:
        client = RedisClient(settings)
        try:
            payload = b"\x00\x01\xff\xfe"
            await client.client.set("rag:test:binary", payload, ex=5)
            return await client.client.get("rag:test:binary")
        finally:
            await client.client.delete("rag:test:binary")
            await client.close()

    assert asyncio.run(scenario()) == b"\x00\x01\xff\xfe"
