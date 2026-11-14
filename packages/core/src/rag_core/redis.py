"""Redis 连接层：异步客户端、健康检查、生命周期。

本机跑的是 Valkey（Redis 协议兼容），所以这里的语义完全按 Redis 写。
换回官方 Redis 或切到集群模式，只改 `build_client` 一处。

为什么用 `redis.asyncio` 而不是 aioredis：
aioredis 已并入 redis-py 官方仓库（aioredis 2.x 之后不再单独维护），
`redis.asyncio` 是持续维护的路径。这个坑社区里踩得很多，架构上没差别。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.asyncio.connection import ConnectionPool

from rag_core.errors import DependencyUnavailableError
from rag_core.logging import get_logger
from rag_core.settings import Settings

log = get_logger(__name__)


def build_pool(settings: Settings) -> ConnectionPool:
    """按配置建连接池。

    关键参数：

    - **max_connections**：连接上限。Redis 单连接很便宜，但每条连接
      占一个文件描述符；不设上限时突发流量可能打穿 ulimit，
      报 "Too many open files" 这种极难定位的错误。
    - **socket_timeout**：读写超时。不设的话一个卡住的 Redis 会把
      asyncio 任务永远挂住（Redis 客户端默认没有超时）。
    - **socket_connect_timeout**：建连超时，同样必须有。
    - **health_check_interval**：定期 PING 探活，防止拿到半开连接。
      和 PG 的 pool_pre_ping 是一回事，但 Redis 是后台定时做。
    - **decode_responses=False**（默认）：保留 bytes。这是个有意的选择，
      见 `RedisClient.get_json` 的说明。
    """
    return ConnectionPool.from_url(
        settings.redis.url.get_secret_value(),
        max_connections=settings.redis.max_connections,
        socket_timeout=settings.redis.socket_timeout_s,
        socket_connect_timeout=settings.redis.socket_timeout_s,
        health_check_interval=30,
        retry_on_timeout=False,  # 重试统一交给 ResilientCaller，不在这里偷偷重试
    )


class RedisClient:
    """Redis 资源持有者。和 Database 一样用类而非全局单例。

    为什么不封装成"万能工具类"：
    业务代码需要的是原生命令（SET/GET/EVAL/HSET…），过度封装会逼出
    "给每种用法加一个方法"的反模式。这里只负责**连接与生命周期**，
    命令直接用 `client` 属性。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: ConnectionPool = build_pool(settings)
        self._client: Redis = Redis(connection_pool=self._pool)

    @property
    def client(self) -> Redis:
        """原生异步客户端。限流 Lua、原子自增、缓存都用它。"""
        return self._client

    def pipeline(self, *, transaction: bool = True) -> Pipeline:
        """批量命令。

        transaction=True 时用 MULTI/EXEC 包起来，保证这批命令原子执行。
        但**注意**：MULTI/EXEC 不能做"读-判断-写"，那种原子性必须用
        Lua 脚本（A13 的限流器就是 Lua）。pipeline 只保证"一起发、
        中间不被别的客户端插入"。
        """
        return self._client.pipeline(transaction=transaction)

    @asynccontextmanager
    async def watch(self, *keys: str) -> AsyncIterator[Redis]:
        """WATCH/MULTI 乐观锁模式。

        用法：
            async with redis.watch("counter") as client:
                current = await client.get("counter")
                pipe = client.pipeline()
                pipe.multi()
                pipe.set("counter", int(current or 0) + 1)
                await pipe.execute()

        如果 watch 的 key 在 execute 前被改过，execute 会抛 WatchError，
        调用方重试。这是 Redis 侧实现"读-改-写"的正确方式。

        但**计数类场景优先用 INCR**（单命令原子），别用 watch 绕一圈。
        """
        from redis.exceptions import WatchError

        async with self._client.pipeline(transaction=True) as pipe:
            await pipe.watch(*keys)
            try:
                yield self._client
            except WatchError:
                log.warning("redis_watch_conflict", keys=list(keys))
                raise

    async def ping(self) -> None:
        """连通性检查。连不通抛 DependencyUnavailableError。"""
        try:
            await self._client.ping()
        except Exception as exc:
            log.error("redis_ping_failed", error=str(exc), exc_info=True)
            raise DependencyUnavailableError("缓存服务不可用") from exc

    async def info_section(self, section: str = "server") -> dict[str, str]:
        """读 INFO 的某个段，用于启动日志与排障。"""
        raw = await self._client.info(section)
        return {str(key): str(value) for key, value in raw.items()}

    async def close(self) -> None:
        """关闭客户端与连接池。"""
        await self._client.aclose()
        await self._pool.disconnect()
