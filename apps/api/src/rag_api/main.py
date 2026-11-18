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

from rag_adapters.storages.factory import build_all_storages
from rag_api.api.health import router as health_router
from rag_api.api.v1.admin_audit import router as admin_audit_router
from rag_api.api.v1.auth import build_auth_router, build_me_router
from rag_api.api.v1.uploads import router as uploads_router
from rag_api.errors_handlers import register_exception_handlers
from rag_api.identity.factory import available_providers
from rag_api.identity.tokens import TokenService
from rag_api.middleware import (
    AccessLogMiddleware,
    MetricsMiddleware,
    RequestContextMiddleware,
)
from rag_api.security.pii import redact_text as redact_pii
from rag_core.db import Database, wait_until_ready
from rag_core.errors import ConfigError
from rag_core.logging import get_logger, register_text_redactor, setup_logging
from rag_core.redis import RedisClient
from rag_core.settings import Settings, get_settings
from rag_core.tracing import build_tracer

log = get_logger(__name__)


def _build_lifespan(settings: Settings) -> Any:
    """构造 lifespan。用闭包传 settings，便于测试注入不同配置。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 1. 日志必须最先装配。
        #    先注册 PII 脱敏再 setup_logging：保证从第一条日志起
        #    手机号/身份证/邮箱就已经被打码（而不是启动日志里漏出去）。
        register_text_redactor(redact_pii)
        setup_logging(settings)

        # 2. 启动期配置自检：误配要在启动时炸，不要等第一个请求
        _validate_startup_config(settings)

        database = Database(settings)
        redis = RedisClient(settings)
        # 构造所有可用后端：配置的主后端 + local（迁移期读老数据）
        storages = build_all_storages(settings)
        primary_storage = storages[settings.storage.backend]
        # 追踪器：**与其他进程级资源一起构造**，因为它的生命周期也是进程级。
        # 关闭时必须 shutdown（不是只 flush），否则最后一批 span 会丢。
        tracer = build_tracer(settings.tracing)

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

            # 5. 模型 provider（**进程级单例，按需构造**）
            #
            # ## 为什么不在启动时无条件构造（实测踩到的坑）
            #
            # 第一版在这里直接 `build_llm_provider(settings)`，结果：
            # 所有只需要鉴权/健康检查的测试都因为"未配置 RAG_LLM__API_KEY"
            # 而起不来。**问答能力缺失不该让整个 API 起不来**——
            # 上传、鉴权、知识库管理都不依赖 LLM。
            #
            # 所以改成：**首次真正需要问答时才构造**（见 `get_llm_provider`）。
            # 构造失败的异常在请求时暴露（500 + 明确错误码），
            # 而不是让进程启动失败。
            #
            # 但**配置探测要提前**：启动日志里明确写"LLM 未配置"，
            # 否则"问答 500"会变成没人知道原因的谜。
            llm_configured = bool(settings.llm.api_key)
            if not llm_configured:
                log.warning(
                    "llm_not_configured",
                    impact="问答接口（/api/v1/chat）不可用，其它接口正常",
                    hint="设置 RAG_LLM__API_KEY",
                )

            app.state.settings = settings
            app.state.database = database
            app.state.redis = redis
            app.state.token_service = TokenService(settings.identity)
            app.state.storage = primary_storage
            app.state.storages = storages
            app.state.tracer = tracer
            app.state.app_version = _app_version()
            # 惰性持有：第一次访问时才构造（线程/协程安全由 GIL 与
            # 单次赋值保证——两个请求同时构造会各建一个，但都正确，
            # 只是多一个 httpx client。可接受，不值得为它加锁）
            app.state._llm_provider = None
            app.state._embedding_provider = None
            app.state._reranker_provider = None

            log.info(
                "model_providers_lazy",
                llm_configured=llm_configured,
                llm_model=settings.llm.chat_model,
                embedding_provider=settings.embedding.provider,
                embedding_model=settings.embedding.model_name,
                embedding_dim=settings.embedding.dimension,
                embedding_api_key_configured=settings.embedding.configured,
                reranker_provider=settings.reranker.provider,
                retrieval_hybrid=settings.retrieval.use_keyword,
                retrieval_rerank=settings.retrieval.use_rerank,
                retrieval_self_check=settings.retrieval.use_self_check,
                note="provider 在首次使用时构造；远程 embedding 无需加载本地模型",
            )

            # 远程 embedding 缺 key 不阻断启动（上传/鉴权/管理都不依赖它），
            # 但必须在启动日志里说清楚，否则"文档一直停在 embedding"很难排查。
            if not settings.embedding.configured:
                log.warning(
                    "embedding_not_configured",
                    provider=settings.embedding.provider,
                    impact="文档会停在向量化阶段，检索/问答不可用",
                    hint="设置 RAG_EMBEDDING__API_KEY",
                )

            # 6. 追踪：配置了就必须能用。
            #
            # SDK 的导出是后台异步的：host 写错 / 凭证错 / 服务没起，
            # 业务侧一次异常都不会看到（表现为"Langfuse 没有数据"）。
            # 所以启动时主动探测一次，把静默失效变成一条明确 warning。
            tracing_ready = False
            if settings.tracing.configured:
                tracing_ready = await tracer.averify()
            log.info(
                "tracing_ready",
                **settings.tracing.redacted_summary,
                verified=tracing_ready,
                implementation=type(tracer).__name__,
            )
            if settings.tracing.enabled and not settings.tracing.configured:
                log.warning(
                    "tracing_enabled_without_credentials",
                    impact="只记录本地内存 span，不导出",
                    hint="设置 RAG_TRACING__PUBLIC_KEY / RAG_TRACING__SECRET_KEY",
                )

            # 配置矛盾要在启动时说出来：开了重排却没有 reranker，
            # 请求层会让它退化为 RRF 顺序（不报 500），但启动日志必须能解释
            # “为什么重排看起来没生效”。
            if settings.retrieval.use_rerank and settings.reranker.provider == "none":
                log.warning(
                    "rerank_enabled_without_provider",
                    impact="重排不会执行，检索使用 RRF 顺序",
                    hint=(
                        "设置 RAG_RERANKER__PROVIDER=siliconflow 并配置 API_KEY，"
                        "或关闭 RAG_RETRIEVAL__USE_RERANK"
                    ),
                )
            elif settings.retrieval.use_rerank and not settings.reranker.configured:
                # 选了 provider 但缺 key：构造会在首次请求时失败，
                # 而重排失败会被降级成 RRF 顺序——启动日志必须提前说出来。
                log.warning(
                    "rerank_provider_not_configured",
                    provider=settings.reranker.provider,
                    impact="重排调用会失败并降级为 RRF 顺序",
                    hint="设置 RAG_RERANKER__API_KEY",
                )

            yield
        finally:
            # 关闭顺序与创建相反：先关 provider（释放模型显存与连接池），
            # 再 Redis（更轻量），最后 DB。
            import contextlib

            if getattr(app.state, "_llm_provider", None) is not None:
                with contextlib.suppress(Exception):
                    await app.state._llm_provider.aclose()
            if getattr(app.state, "_embedding_provider", None) is not None:
                with contextlib.suppress(Exception):
                    await app.state._embedding_provider.aclose()
            if getattr(app.state, "_reranker_provider", None) is not None:
                # `NoOpReranker` 没有 aclose（没有资源需要释放），
                # 这里用 getattr 判断而不是让 suppress 吞 AttributeError。
                reranker_aclose = getattr(app.state._reranker_provider, "aclose", None)
                if reranker_aclose is not None:
                    with contextlib.suppress(Exception):
                        await reranker_aclose()
            # 追踪最后关：前面任何资源关闭时产生的 span 都还能被记录。
            with contextlib.suppress(Exception):
                await tracer.aclose()
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

    # ---- 指标（Prometheus）----
    #
    # 在 lifespan 之外构建：`/metrics` 端点可能在任何请求周期被调用，
    # 不能等 lifespan 启动才创建。指标本身不依赖外部资源（纯内存计数）。
    from rag_core.metrics import build_metrics

    metrics = build_metrics()
    app.state.metrics = metrics

    # 中间件：注册顺序 = 由外到内，所以 RequestContext 先注册。
    #
    # MetricsMiddleware 放在最内层（最后注册）：它需要 `scope["route"]`
    # 已由路由器填充，否则拿到的是 None，所有请求都归到 __unmatched__。
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(MetricsMiddleware, metrics=metrics)
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
    api_v1.include_router(admin_audit_router)
    # 知识库与文档（B4）：KB 的 CRUD + blob 登记为文档并触发解析
    from rag_api.api.v1.documents import router as documents_router
    from rag_api.api.v1.knowledge_bases import router as knowledge_bases_router

    api_v1.include_router(knowledge_bases_router)
    api_v1.include_router(documents_router)
    # 问答与会话（A16）：SSE 流式 chat + 会话 CRUD + 反馈
    from rag_api.api.v1.chat import router as chat_router

    api_v1.include_router(chat_router)

    app.include_router(health_router)
    app.include_router(api_v1)

    log.info(
        "app_created",
        env=str(resolved.env),
        dev_auth_enabled=resolved.dev_auth_enabled,
        routes=len(app.routes),
    )
    return app


# ---------------------------------------------------------------- 惰性 provider


def get_llm_provider(app: Any) -> Any:
    """取（或首次构造）LLM provider。

    进程级单例：httpx 连接池复用。首次访问构造，
    失败时抛 `ConfigError`（明确告知"没配 API key"）。

    ## 为什么不在启动时构造

    问答能力缺失不该让整个 API 起不来——上传、鉴权、知识库管理
    都不依赖 LLM。启动时硬失败会让"没配 key"变成"服务不可用"。
    """
    existing = getattr(app.state, "_llm_provider", None)
    if existing is not None:
        return existing

    from rag_adapters.providers.factory import build_llm_provider

    provider = build_llm_provider(app.state.settings)
    app.state._llm_provider = provider
    return provider


def get_embedding_provider(app: Any) -> Any:
    """取（或首次构造）Embedding provider。

    进程级单例：httpx 连接池复用。远程 API 实现不含本地模型，
    构造本身很轻（只建 client）；单例仍必要，因为每次请求新建 client
    会反复 TLS 握手并泄漏 fd。
    """
    existing = getattr(app.state, "_embedding_provider", None)
    if existing is not None:
        return existing

    from rag_adapters.providers.factory import build_embedding_provider

    provider = build_embedding_provider(app.state.settings)
    app.state._embedding_provider = provider
    return provider


def get_reranker_provider(app: Any) -> Any:
    """取（或首次构造）Reranker provider。

    进程级单例。当前只有 `NoOpReranker`（直通），
    只有 `RAG_RETRIEVAL__USE_RERANK=true` 时路由层才会调用它——
    避免"没开重排却构造 provider"。
    """
    existing = getattr(app.state, "_reranker_provider", None)
    if existing is not None:
        return existing

    from rag_adapters.providers.factory import build_reranker

    provider = build_reranker(app.state.settings)
    app.state._reranker_provider = provider
    return provider
