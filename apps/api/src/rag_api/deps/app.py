"""进程级单例依赖：不涉及请求身份，只从 app.state 取资源。

这三类资源的共同点：
- 生命周期与进程一致（lifespan 里创建、关闭）；
- 与请求无关（不依赖 token、不依赖租户）；
- 无状态（连接池自己管理状态）。

因此它们可以放在依赖链的最底层，被任何其它依赖复用。
"""

from __future__ import annotations

from fastapi import Request

from rag_api.identity.tokens import TokenService
from rag_core.db import Database
from rag_core.redis import RedisClient
from rag_core.settings import Settings


def get_settings_from_app(request: Request) -> Settings:
    """运行期配置。

    从 `app.state` 取而不是调 `get_settings()`：
    测试会创建多个不同配置的 app 实例，`get_settings()` 有 lru_cache，
    取到的可能是上一个实例的配置。
    """
    settings: Settings = request.app.state.settings
    return settings


def get_database(request: Request) -> Database:
    """进程级数据库资源（含运行时引擎与管理引擎）。"""
    database: Database = request.app.state.database
    return database


def get_redis(request: Request) -> RedisClient:
    """进程级 Redis 客户端。"""
    redis: RedisClient = request.app.state.redis
    return redis


def get_token_service(request: Request) -> TokenService:
    """token 签发/校验服务。密钥来自配置，故随进程创建。"""
    service: TokenService = request.app.state.token_service
    return service
