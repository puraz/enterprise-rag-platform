"""API 端应用装配。

## lifespan 里做什么

启动：
1. 装配 structlog（必须在第一行，否则后续日志无格式）；
2. 建 DB/Redis 连接池；
3. 等待依赖就绪（有限重试）；
4. 打印生效配置（脱敏）与角色信息；
5. 校验身份提供者配置（dev 是否被误开）。

关闭：
1. 关闭连接池（**顺序：先 DB 后 Redis**，与创建相反）。

## 为什么在启动时校验而不是请求时

配置错误（如生产环境误开 dev 登录）应该在**进程启动阶段**失败，
让编排器立刻发现问题并回滚。等到第一个请求才发现，
意味着这个错误配置已经跑了一段时间。

## 路由分层

    /api/v1/*           用户面（需要用户 token）
    /api/v1/admin/*     管理面（需要平台角色）
    /health/*           健康检查（无需认证）

用户面与管理面**在同应用内路由隔离**，但鉴权依赖完全不同。
`docs/PROMPT.md` 要求"管理接口禁止从用户前端访问"，
所以管理面还会在网关层做额外限制（C 阶段）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI

from rag_api.api.health import router as health_router
from rag_api.api.v1.auth import build_auth_router, build_me_router
from rag_api.api.v1.uploads import router as uploads_router
from rag_api.errors_handlers import register_exception_handlers
from rag_api.identity.factory import available_providers
from rag_api.identity.tokens import TokenService
from rag_api.middleware import AccessLogMiddleware, RequestContextMiddleware
from rag_api.storages.factory import build_all_storages
from rag_core.db import Database, wait_until_ready
from rag_core.errors import ConfigError
from rag_core.logging import get_logger, setup_logging
from rag_core.redis import RedisClient
from rag_core.settings import Settings, get_settings

log = get_logger(__name__)


def _build_lifespan(settings: Settings) -> Any:
    """构造 lifespan。用闭包传 settings，便于测试注入不同配置。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 1. 日志必须最先装配
        setup_logging(settings)

        # 2. 启动期配置自检：误配要在启动时炸，不要等第一个请求
        _validate_startup_config(settings)

        database = Database(settings)
        redis = RedisClient(settings)
        # 构造所有可用后端：配置的主后端 + local（迁移期读老数据）
        storages = build_all_storages(settings)
        primary_storage = storages[settings.storage.backend]

        try:
            # 3. 等依赖就绪（容器编排里 API 与 PG 同时启动，PG 需要几秒）。
            #    纯 HTTP 契约测试可以关掉这一步（startup_dependency_check=False），
            #    但生产必须开启。
            if settings.startup_dependency_check:
                await wait_until_ready(database.ping, attempts=10, delay_s=0.5)
                await wait_until_ready(redis.ping, attempts=5, delay_s=0.5)

                pg_version = await database.server_version()
                extensions = await database.installed_extensions()
                log.info(
                    "startup_dependencies_ready",
                    pg_version=pg_version,
                    extensions=sorted(extensions),
                    db_user=settings.database.user,
                )

            # 4. 打印生效配置（密钥已脱敏）
            log.info(
                "startup_config",
                env=str(settings.env),
                app_name=settings.app_name,
                log_format=settings.resolved_log_format,
                db_dsn=settings.database.redacted_dsn,
                redis_url=settings.redis.redacted_url,
                identity_provider=settings.identity.provider,
                dev_auth_enabled=settings.dev_auth_enabled,
                identity_providers_available=list(available_providers()),
                storage_backend=settings.storage.backend,
                storage_backends=list(storages),
            )

            app.state.settings = settings
            app.state.database = database
            app.state.redis = redis
            app.state.token_service = TokenService(settings.identity)
            app.state.storage = primary_storage
            app.state.storages = storages
            app.state.app_version = _app_version()

            yield
        finally:
            # 关闭顺序与创建相反：Redis 先关（它更轻量），DB 后关
            await redis.close()
            await database.close()
            log.info("shutdown_complete")

    return lifespan


def _validate_startup_config(settings: Settings) -> None:
    """启动期配置自检。

    `Settings` 的生产守卫已经拦住了大部分问题（dev 密钥、debug、HS256）。
    这里补充两项**只有应用层才知道**的检查：

    1. dev 身份提供者是否被误开（生产已由守卫拦截，这里兜非枚举环境）；
    2. provider 名是否在注册表里（拼错时要立刻炸，而不是请求时 500）。
    """
    if settings.env.is_production_like and settings.identity.provider == "dev":
        raise ConfigError("生产环境不允许使用 dev 身份提供者")

    if settings.identity.provider not in available_providers():
        raise ConfigError(
            f"未知的身份提供者 {settings.identity.provider!r}，"
            f"可选：{sorted(available_providers())}"
        )


def _app_version() -> str:
    from importlib.metadata import version

    try:
        return version("rag-api")
    except Exception:  # pragma: no cover
        return "unknown"


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建应用实例。

    工厂函数而非模块级 `app = FastAPI()`：
    测试需要多个不同配置的实例（有的开 dev 登录、有的关），
    模块级单例做不到。`main.py` 里再创建生产用的实例。
    """
    resolved = settings or get_settings()

    app = FastAPI(
        title="Enterprise RAG Platform API",
        version=_app_version(),
        description="企业级多租户 RAG 平台。用户面 /api/v1，管理面 /api/v1/admin。",
        lifespan=_build_lifespan(resolved),
        # 关闭默认的 /docs 在非本地环境：生产暴露交互式文档
        # 等于给了攻击者完整的接口清单
        docs_url="/docs" if not resolved.env.is_production_like else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not resolved.env.is_production_like else None,
    )

    # 中间件：注册顺序 = 由外到内，所以 RequestContext 先注册
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    # ---- 路由装配 ----
    api_v1 = APIRouter(prefix="/api/v1")

    # dev 登录：**只在显式开启且非生产时注册端点**（不注册 = 404，
    # 比"注册了但返回 403"更彻底）。每次创建新 router，
    # 避免模块级单例把 dev 端点泄漏给其它 app 实例。
    dev_enabled = resolved.dev_auth_enabled and not resolved.env.is_production_like
    api_v1.include_router(build_auth_router(dev_enabled=dev_enabled))
    api_v1.include_router(build_me_router())
    api_v1.include_router(uploads_router)

    app.include_router(health_router)
    app.include_router(api_v1)

    log.info(
        "app_created",
        env=str(resolved.env),
        dev_auth_enabled=resolved.dev_auth_enabled,
        routes=len(app.routes),
    )
    return app
