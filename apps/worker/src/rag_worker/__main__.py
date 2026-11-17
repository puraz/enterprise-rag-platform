"""Worker 进程入口：`python -m rag_worker` / `rag-worker`。

## 进程生命周期

    main()
      ├─ load_env_files()          读 .env（不依赖 cwd）
      ├─ get_settings()            RAG_* → Settings（校验 + 生产守卫）
      ├─ setup_logging()           结构化日志 + PII 脱敏钩子
      ├─ Database / Redis          建连接池
      ├─ build_all_storages()      存储后端（S3 client 有连接池，必须单例）
      ├─ build_embedding_provider()  **惰性**：构造不加载模型
      ├─ build_handlers()          任务注册表
      ├─ 注册信号处理              SIGTERM/SIGINT → 优雅退出
      ├─ engine.run_forever()      主循环
      └─ finally: 关闭连接池、释放存储

## 为什么信号处理是"置标志"而不是"立即退出"

容器编排（k8s / docker stop）发 SIGTERM 后会等 `terminationGracePeriod`
再发 SIGKILL。如果收到 SIGTERM 就直接 `sys.exit()`：
- 正在解析的任务被强杀；
- 数据库事务被中断（连接层会回收，但业务可能留下半成品）；
- 任务停在 `running`，要等 `lock_timeout`（默认 10 分钟）才被回收。

正确处理：置标志 → 主循环在当前批次结束后退出 → 主动关闭连接池。
这样"优雅退出"的实际效果是**最多等一个批次**，而不是等 10 分钟。

## 为什么 worker 需要 admin 连接

`tasks` 表受 RLS 保护，但 worker 的是"所有租户的任务队列"。
用普通角色扫 `tasks` 会被 RLS 过滤成空——**不报错，只是永远捡不到任务**。
所以认领/回收/进度/状态回写走 admin（BYPASSRLS），
而 handler 的业务读写走租户会话（RLS 生效）。
详见 `engine.py` 模块 docstring 第 2 点。

## 退出码

- 0：正常退出（收到信号）
- 1：启动失败（配置错误、依赖不可用）
- 2：运行期致命错误（主循环抛未捕获异常）

用不同退出码是为了让编排器区分"重启就好"与"要人看"。
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from typing import Any

from rag_adapters.providers.factory import build_embedding_provider
from rag_adapters.storages import build_all_storages
from rag_core.db import Database
from rag_core.logging import get_logger, setup_logging
from rag_core.settings import Settings, get_settings, load_env_files
from rag_worker.engine import WorkerEngine
from rag_worker.handlers import build_handlers

log = get_logger(__name__)


def _install_signal_handlers(engine: WorkerEngine, loop: asyncio.AbstractEventLoop) -> None:
    """把 SIGTERM/SIGINT 变成"优雅退出"请求。

    `loop.add_signal_handler` 而不是 `signal.signal`：后者会在
    信号到达时打断当前字节码，在 async 代码里可能让协程停在
    `await` 中间（虽然 CPython 会重试，但行为依赖实现细节）。
    `add_signal_handler` 通过事件循环回调调度，语义干净。
    """

    def _request_shutdown(signal_name: str) -> None:
        log.warning("signal_received", signal=signal_name, note="将在当前批次结束后退出")
        engine.request_shutdown()

    def _make_stdlib_handler(name: str) -> Any:
        """构造 stdlib 信号回调。

        必须用工厂把 `name` 绑死：`lambda ...: _request_shutdown(name)`
        写在循环里会捕获**变量**而非当前值，两个回调最终都指向最后一个
        信号名（B023 抓的就是这个）。这里日志会明确写错信号名，
        排障时误导性极强。
        """

        def _handler(_signum: int, _frame: object) -> None:
            _request_shutdown(name)

        return _handler

    for sig, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGINT, "SIGINT")):
        try:
            loop.add_signal_handler(sig, _request_shutdown, name)
        except NotImplementedError:  # pragma: no cover - Windows 无 add_signal_handler
            # Windows 没有这套机制。worker 生产部署在 Linux 容器里，
            # 这里只保证本地开发（Ctrl+C）不至于崩溃。
            signal.signal(sig, _make_stdlib_handler(name))


async def run(settings: Settings) -> int:
    """装配资源并跑主循环。返回退出码。"""
    from rag_core.logging import register_text_redactor
    from rag_core.pii import redact_text

    # 与 API 一致：在 setup_logging 之前注册 PII 脱敏，
    # 保证第一条日志起就生效（否则启动日志可能带用户数据）。
    # 注册幂等，多进程各注册一次没有问题。
    register_text_redactor(redact_text)
    setup_logging(settings)

    log.info(
        "worker_booting",
        env=settings.env.value,
        worker_id=settings.worker.worker_id,
        log_format=settings.resolved_log_format,
    )

    database = Database(settings)
    storages = build_all_storages(settings)
    embedding_provider: Any = None
    tracer: Any = None
    # `start_http_server` 返回 (httpd, thread)，没有全局 stop 函数；
    # 必须在 finally 里显式 shutdown + server_close，否则退出时端口/线程残留。
    metrics_httpd: Any = None

    try:
        # 启动期就探一次：连不上立刻以退出码 1 结束，
        # 让编排器重启/告警，而不是"进程活着但什么也干不了"。
        await database.ping()
        log.info(
            "database_ready",
            dsn=settings.database.redacted_dsn,
            server_version=await database.server_version(),
            rls_bypass_configured=settings.database.admin_user is not None,
        )

        # Embedding provider 是**惰性构造**：`LocalBGEEmbedding.__init__`
        # 只存配置，不加载模型（见其 `_ensure_loaded`）。模型在首次
        # embed 调用时才加载。
        #
        # 为什么这样设计：worker 可能只跑解析任务，不该为"可能用不到的
        # 能力"付 2-5 秒加载时间与 2GB 内存。多副本时这个开销会乘上副本数。
        embedding_provider = build_embedding_provider(settings)
        log.info(
            "embedding_provider_configured",
            provider=embedding_provider.name,
            dimension=embedding_provider.capabilities.dimension,
            note="模型将在首次向量化时加载",
        )

        handlers = build_handlers(
            storages=storages,
            embedding_provider=embedding_provider,
        )
        engine = WorkerEngine(
            settings=settings,
            worker_settings=settings.worker,
            handlers=handlers,
            # 跨租户会话：认领/回收/进度/完成失败回写
            task_session_factory=database.transaction_bypass_rls,
            # 租户会话：handler 的业务读写（RLS 生效）。
            # transaction() 的 tenant_id 是 keyword-only，这里适配成
            # 位置参数形式（引擎不需要知道 core 的调用约定）。
            tenant_session_factory=lambda tenant_id: database.transaction(tenant_id=tenant_id),
        )

        # 指标：构建并挂到引擎上；可选启动 /metrics HTTP 端点供 Prometheus 抓取。
        #
        # Worker 没有业务 HTTP 服务，如果不主动暴露，Prometheus 只能抓到
        # API 的指标，dashboard 的 Worker 面板会永远是空的。
        # 多副本部署时每个实例各暴露一个端口，Prometheus 用服务发现逐个抓取；
        # 也可以用 PROMETHEUS_MULTIPROC_DIR + sidecar，本步骤保留单进程方案。
        from rag_core.metrics import build_metrics

        metrics = build_metrics()
        engine.attach_metrics(metrics)
        metrics_server_started = False
        if settings.worker.metrics_port:
            from prometheus_client import start_http_server

            metrics_httpd, _metrics_thread = start_http_server(
                settings.worker.metrics_port,
                addr=settings.worker.metrics_host,
                registry=metrics.registry,
            )
            metrics_server_started = True
            log.info(
                "worker_metrics_server_started",
                addr=settings.worker.metrics_host,
                port=settings.worker.metrics_port,
                path="/metrics",
            )
        log.info(
            "worker_metrics_ready",
            multiproc=settings.env.is_production_like,
            exposed=metrics_server_started,
            note="多副本场景建议 PROMETHEUS_MULTIPROC_DIR 或逐实例抓取",
        )

        # 追踪器：未配置凭证时用 LocalTracer（内存记录），
        # 不会因为缺少 Langfuse 而起不来。
        from rag_core.tracing import build_tracer

        tracer = build_tracer(enabled=False)
        log.info("worker_tracer_ready", kind=type(tracer).__name__)

        loop = asyncio.get_running_loop()
        _install_signal_handlers(engine, loop)

        await engine.run_forever()
        return 0

    except Exception:
        # exc_info=True：启动失败的堆栈必须完整，否则"worker 起不来"
        # 会变成只能靠猜的问题。
        log.error("worker_fatal_error", exc_info=True)
        return 2
    finally:
        # 一定要关：连接池不释放会让 PG 侧连接数缓慢泄漏。
        # 用 suppress 是因为关闭失败不该覆盖上面的退出码。
        if metrics_httpd is not None:
            with contextlib.suppress(Exception):
                metrics_httpd.shutdown()
            with contextlib.suppress(Exception):
                metrics_httpd.server_close()
        with contextlib.suppress(Exception):
            await database.close()
        # 释放 embedding 模型（占 GPU 显存/内存）。不释放会让进程
        # "退出中"卡住，或让同机其它进程拿不到显存。
        #
        # 判断非 None：provider 构造可能在 try 块内失败（配置错误），
        # 此时不该再调用它的方法——suppress 会吞掉异常，掩盖真正的启动失败原因。
        if embedding_provider is not None:
            with contextlib.suppress(Exception):
                await embedding_provider.aclose()
        with contextlib.suppress(Exception):
            await tracer.aclose()

        for name, backend in storages.items():
            closer = getattr(backend, "aclose", None)
            if closer is not None:
                with contextlib.suppress(Exception):
                    await closer()
            log.debug("storage_closed", backend=name)
        log.info("worker_resources_released")


def main() -> int:
    """同步入口（`python -m rag_worker`）。"""
    # 先读 .env 再 get_settings()：Settings 本身不读文件
    # （避免"import 时就读文件"的副作用）。
    load_env_files()
    try:
        settings = get_settings()
    except Exception as exc:
        # 配置错误在打印日志前就要报出来：此时 setup_logging 还没跑，
        # 直接写 stderr 最可靠（日志框架可能因配置错误而不可用）。
        print(f"配置加载失败: {exc}", file=sys.stderr)
        return 1

    return asyncio.run(run(settings))


if __name__ == "__main__":
    sys.exit(main())
