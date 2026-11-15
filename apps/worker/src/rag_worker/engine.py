"""Worker 执行引擎：认领 → 执行 → 回写状态。

## 设计要点

### 1. 数据库是任务的权威来源，队列只是通知

    API 写文档 + 写任务 + 写 outbox（同一个事务）
      → outbox 转发到队列（异步，至少一次）
      → worker 收到通知立刻处理
      → **同时 worker 每 2 秒轮询数据库**（兜底）

为什么两者都要：
- 只靠队列：Redis 丢数据（重启未持久化、内存淘汰）→ 任务永久丢失；
- 只靠轮询：延迟高（最少一个轮询周期）。

轮询是兜底，所以间隔可以宽松（默认 2 秒）。

### 2. 认领必须跨租户，执行必须在租户内

`tasks` 表在 `RLS_TABLES` 里（租户隔离是硬要求）。但 worker 面临的
是**所有租户的任务队列**——用普通会话扫 `tasks` 会被 RLS 过滤成空，
表现为"worker 起来了但永远捡不到任务"，且不报任何错。

所以两种会话分工明确：

| 用途 | 连接角色 | 原因 |
|---|---|---|
| `claim` / `reclaim_stale` / 进度 / 完成 / 失败 | **admin（BYPASSRLS）** | 必须看到全部租户的任务 |
| handler 里的业务读写 | **租户会话（set_config）** | RLS 必须生效，越权即事故 |

**不能让业务读写也走 admin**：那等于把 RLS 这道防线整个撤掉，
handler 里一个漏写的 `WHERE tenant_id` 就是跨租户数据事故。

### 3. 进度回写必须用独立事务

反直觉但很关键：进度写进"工作事务"里，**提交前谁都看不到**——
切分/向量化跑 5 分钟，期间前端轮询看到的进度永远停在 0，
最后一下子跳到 100%。等于没有进度。

所以：
- 工作事务：短、原子、只装业务改动；
- 进度写入：每条独立短事务（`ProgressReporter` 自带会话工厂）。

### 4. 工作与"标记完成"不放在一个大事务里

看似"要么全成功要么全失败"更安全，但代价是**事务持续时间 = 任务耗时**
（可能 10 分钟）。长事务会：
- 占住一个连接（并发任务数 = 连接数，跑满就卡死）；
- 阻止 vacuum 推进（长事务期间死元组无法回收）；
- 崩溃时整个事务回滚，10 分钟算力白费。

改为"InnoDB 式"的短事务 + **handler 幂等**：
- handler 提交业务改动
- 再标记任务成功
- 两步之间崩溃 → 任务被回收重派 → handler 再跑一遍

因此 `TaskHandler` 契约里"幂等"不是建议而是**必需**。
本项目的 handler 用确定性存储键 + 乐观锁/upsert 做到幂等。

### 5. 失败分类决定重试策略

| 失败类型 | 重试 | 例子 |
|---|---|---|
| RETRYABLE | ✅ 指数退避 | 存储超时、embedding 限流、乐观锁冲突 |
| PERMANENT | ❌ 直接死信 | 文件损坏、格式不支持、payload 非法 |
| FATAL | ❌ 直接死信 | 未知 bug、我方配置错误 |

**把永久失败重试 3 次是纯浪费**：文件就是损坏的，重试 3 次还是损坏，
白占 3 个 worker 槽位几分钟，还让用户多等一轮退避。

### 6. 进度回写要节流

300 页 PDF 如果每页写一次 `tasks.progress`，300 次 UPDATE 会占满连接。
本实现只在跨越**固定里程碑**（10/25/40/55/70/85/95）时写，
写入次数有上界（7 次），不随任务规模增长。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums import TaskStatus
from rag_core.errors import AppError
from rag_core.logging import get_logger
from rag_core.pii import redact_text
from rag_core.resilience.classify import FailureKind, classify
from rag_core.settings import Settings
from rag_core.settings.worker import WorkerSettings
from rag_db.repositories import TaskRepository

log = get_logger(__name__)

# 会话工厂类型别名：无参上下文管理器，yield 一个 AsyncSession。
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

# 租户会话工厂：需要 tenant_id 才能设置 RLS 上下文。
TenantSessionFactory = Callable[[uuid.UUID], AbstractAsyncContextManager[AsyncSession]]


class TaskHandler(Protocol):
    """任务处理器契约。

    实现约定：
    - **幂等**：同一任务重复执行结果一致（重派是正常路径，不是异常）；
    - 不吞异常：失败必须抛出，由引擎决定重试或死信；
    - 不记录文档内容（日志里只能有 ID 与计数）；
    - 长任务应通过 `progress` 回调报告进度。
    """

    async def __call__(
        self,
        *,
        session: AsyncSession,
        task: Any,
        progress: Callable[[int, str], Awaitable[None]],
    ) -> None: ...


@dataclass
class WorkerStats:
    """一个 worker 实例的运行统计。"""

    claimed: int = 0
    succeeded: int = 0
    failed: int = 0
    dead: int = 0
    reclaimed: int = 0


class ProgressReporter:
    """进度回写（节流 + 独立事务）。

    只在跨越里程碑时写库；每次写入用**自己的短事务**，
    否则进度会随工作事务一起被"藏"到任务结束。
    """

    # 里程碑：只在首次跨越时写。7 个点，写入次数有上界。
    _MILESTONES = (10, 25, 40, 55, 70, 85, 95)

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        task_id: uuid.UUID,
        worker_id: str,
    ) -> None:
        self._session_factory = session_factory
        self._task_id = task_id
        self._worker_id = worker_id
        self._last_milestone = 0

    async def __call__(self, percent: int, stage: str = "") -> None:
        """报告进度。

        Args:
            percent: 0-100。
            stage: 阶段描述（进日志不进数据库——`tasks` 表没有 stage 列，
                加列要迁移，而 stage 是纯观测信息）。

        **100 永远不由 handler 上报**：`progress=100` 的语义是"任务已成功
        结束"，只有 `TaskRepository.complete()` 有权设置。handler 报 100
        会让一个仍在跑（甚至马上要失败）的任务在前端显示"已完成"。
        所以这里把上界压到 99。
        """
        clamped = max(0, min(99, percent))
        if not any(self._last_milestone < m <= clamped for m in self._MILESTONES):
            return
        self._last_milestone = clamped

        try:
            async with self._session_factory() as session:
                await TaskRepository(session).update_progress(
                    task_id=self._task_id,
                    worker_id=self._worker_id,
                    progress=clamped,
                )
                await session.commit()
        except Exception as exc:
            # 进度只是观测数据，回写失败不该中断业务
            log.warning(
                "progress_update_failed",
                task_id=str(self._task_id),
                error_type=type(exc).__name__,
            )
            return

        log.info("task_progress", task_id=str(self._task_id), percent=clamped, stage=stage)


class WorkerEngine:
    """Worker 主循环：认领、执行、回写。

    一个引擎实例对应一个进程（`worker_id` 唯一）。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        worker_settings: WorkerSettings,
        handlers: dict[str, TaskHandler],
        task_session_factory: SessionFactory,
        tenant_session_factory: TenantSessionFactory,
    ) -> None:
        """
        Args:
            handlers: 任务类型 → 处理器。**没有 handler 的任务进死信**
                （配置错误不该无限重试）。
            task_session_factory: 跨租户会话（admin/BYPASSRLS），
                用于认领、回收、进度、完成/失败回写。
            tenant_session_factory: 租户会话（RLS 生效），用于 handler 的业务读写。
        """
        self._settings = settings
        self._worker = worker_settings
        self._handlers = handlers
        self._task_sessions = task_session_factory
        self._tenant_sessions = tenant_session_factory
        self._stats = WorkerStats()
        self._shutdown = asyncio.Event()

    @property
    def stats(self) -> WorkerStats:
        return self._stats

    @property
    def should_stop(self) -> bool:
        return self._shutdown.is_set()

    def request_shutdown(self) -> None:
        """请求优雅退出。

        **不立即中断在途任务**：跑到一半被强杀会留下半成品数据。
        置标志，主循环在处理完当前批次后退出。
        """
        log.info("worker_shutdown_requested", worker_id=self._worker.worker_id)
        self._shutdown.set()

    async def run_once(self, *, limit: int | None = None) -> int:
        """执行一批任务，返回处理数量。单独暴露便于测试。"""
        batch = limit if limit is not None else self._worker.max_concurrency
        await self._reclaim_stale()

        claimed = await self._claim(batch)
        if not claimed:
            return 0

        # 并发执行。return_exceptions=True：单个任务失败不影响同批其它任务
        # （_execute 内部已 catch，这里是双保险——比如日志序列化抛错）。
        results = await asyncio.gather(
            *(self._execute(task) for task in claimed), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                log.error(
                    "task_execution_crashed",
                    error_type=type(result).__name__,
                    error=str(result)[:200],
                    exc_info=result,
                )
        return len(claimed)

    # ------------------------------------------------------------ 认领

    async def _reclaim_stale(self) -> None:
        """回收 worker 崩溃遗留的 RUNNING 任务。

        **不能省**：worker 被 kill / 容器被驱逐 / OOM 时，任务会永远停在
        running，再也不会被任何人处理。表现为"文档一直显示处理中"，
        而队列是空的——这类故障最难排查，因为没有报错。
        """
        async with self._task_sessions() as session:
            reclaimed = await TaskRepository(session).reclaim_stale(
                lock_timeout_s=self._worker.lock_timeout_s
            )
            await session.commit()

        if reclaimed:
            self._stats.reclaimed += reclaimed
            log.warning(
                "stale_tasks_reclaimed",
                count=reclaimed,
                lock_timeout_s=self._worker.lock_timeout_s,
                note="有 worker 崩溃或任务超时，已放回队列",
            )

    async def _claim(self, limit: int) -> list[Any]:
        """原子认领一批任务。

        `FOR UPDATE SKIP LOCKED`（在仓储里）让 N 个 worker 各自拿到不同任务，
        而不是串行排队。**没有它，N 个 worker 的吞吐等于 1 个。**
        """
        async with self._task_sessions() as session:
            claimed = await TaskRepository(session).claim(
                worker_id=self._worker.worker_id,
                limit=limit,
                lock_timeout_s=self._worker.lock_timeout_s,
            )
            await session.commit()

        self._stats.claimed += len(claimed)
        return claimed

    # ------------------------------------------------------------ 执行

    async def _execute(self, task: Any) -> None:
        """执行单个任务。"""
        task_id: uuid.UUID = task.id
        task_type = str(task.task_type)

        handler = self._handlers.get(task_type)
        if handler is None:
            await self._dead_letter(
                task_id=task_id,
                task_type=task_type,
                error_code="INTERNAL.NOT_IMPLEMENTED",
                error_message=f"没有 {task_type} 的处理器（可用：{sorted(self._handlers)}）",
                reason="no_handler",
            )
            return

        started = time.perf_counter()
        reporter = ProgressReporter(
            self._task_sessions, task_id=task_id, worker_id=self._worker.worker_id
        )

        try:
            # 业务写在**租户会话**里：RLS 生效，越权即失败而不是静默读到别人数据
            async with self._tenant_sessions(task.tenant_id) as session:
                await handler(session=session, task=task, progress=reporter)
                await session.commit()
        except Exception as exc:
            await self._handle_failure(task=task, exc=exc, started=started)
            return

        # 业务已提交，再标记任务成功（见模块 docstring 第 4 点）
        ok = await self._mark_complete(task_id=task_id)
        if not ok:
            # 认领已失效（任务被回收重派给别的 worker）。
            # 业务改动已提交且 handler 幂等 → 新 worker 重跑不会产生重复数据。
            log.warning(
                "task_complete_rejected",
                task_id=str(task_id),
                task_type=task_type,
                note="认领已失效（任务被回收重派），业务结果保留",
            )
            return

        self._stats.succeeded += 1
        log.info(
            "task_succeeded",
            task_id=str(task_id),
            task_type=task_type,
            elapsed_s=round(time.perf_counter() - started, 2),
        )

    async def _mark_complete(self, *, task_id: uuid.UUID) -> bool:
        try:
            async with self._task_sessions() as session:
                ok = await TaskRepository(session).complete(
                    task_id=task_id, worker_id=self._worker.worker_id
                )
                await session.commit()
                return ok
        except Exception as exc:
            # 标记成功失败：任务会因 lock_timeout 被回收重派，
            # handler 幂等所以重跑安全。不静默吞掉——要能在日志里看到。
            log.error(
                "task_complete_failed",
                task_id=str(task_id),
                error_type=type(exc).__name__,
                note="任务将因锁超时被回收，handler 幂等故重跑安全",
            )
            return False

    async def _dead_letter(
        self,
        *,
        task_id: uuid.UUID,
        task_type: str,
        error_code: str,
        error_message: str,
        reason: str,
    ) -> None:
        """直接进死信（不重试）。"""
        try:
            async with self._task_sessions() as session:
                await TaskRepository(session).fail(
                    task_id=task_id,
                    worker_id=self._worker.worker_id,
                    error_code=error_code,
                    error_message=redact_text(error_message[:200]),
                    retry=False,
                )
                await session.commit()
        except Exception as exc:
            log.error(
                "task_dead_letter_failed",
                task_id=str(task_id),
                error_type=type(exc).__name__,
                note="任务将因锁超时被回收并重试",
            )
            return

        self._stats.dead += 1
        log.error(
            "task_dead_letter",
            task_id=str(task_id),
            task_type=task_type,
            error_code=error_code,
            reason=reason,
            note="已进死信，需人工介入",
        )

    async def _handle_failure(self, *, task: Any, exc: BaseException, started: float) -> None:
        """按失败类型决定重试还是死信。"""
        task_id: uuid.UUID = task.id
        task_type = str(task.task_type)

        kind = classify(exc)
        # 只有 RETRYABLE 才重试。PERMANENT（文件损坏/参数非法）与
        # FATAL（未知 bug/配置错误）直接进死信。
        should_retry = kind is FailureKind.RETRYABLE

        retry_count = int(getattr(task, "retry_count", 0) or 0)
        # 指数退避，上限 10 分钟：避免"第 5 次重试要等 4 小时"
        backoff_s = min(30 * (2**retry_count), 600) if should_retry else 0

        error_code = _error_code_of(exc)
        error_message = _safe_message(exc)

        try:
            async with self._task_sessions() as session:
                new_state = await TaskRepository(session).fail(
                    task_id=task_id,
                    worker_id=self._worker.worker_id,
                    error_code=error_code,
                    error_message=error_message,
                    retry=should_retry,
                    backoff_s=backoff_s,
                )
                await session.commit()
        except Exception as db_exc:
            # 连"标记失败"都失败 → 数据库可能挂了。任务会因 lock_timeout
            # 被回收重派，这是正确的最终兜底（不需要额外处理）。
            log.error(
                "task_failure_marking_failed",
                task_id=str(task_id),
                error_type=type(db_exc).__name__,
                note="任务将因锁超时被回收",
            )
            self._stats.failed += 1
            return

        if new_state == TaskStatus.DEAD.value:
            self._stats.dead += 1
            log.error(
                "task_dead_letter",
                task_id=str(task_id),
                task_type=task_type,
                error_code=error_code,
                error_type=type(exc).__name__,
                failure_kind=kind.value,
                note="已进死信，需人工介入",
                exc_info=exc,
            )
        elif new_state == "not_owned":
            # 认领已失效（任务被回收重派），不该再记一次失败
            log.warning(
                "task_failure_not_owned",
                task_id=str(task_id),
                task_type=task_type,
                note="认领已失效，失败状态由新 owner 负责",
            )
        else:
            self._stats.failed += 1
            log.warning(
                "task_failed_will_retry",
                task_id=str(task_id),
                task_type=task_type,
                error_code=error_code,
                failure_kind=kind.value,
                retry_count=retry_count,
                backoff_s=backoff_s,
                elapsed_s=round(time.perf_counter() - started, 2),
            )

    async def run_forever(self) -> None:
        """主循环：轮询 + 优雅退出。

        `should_stop` 在批次之间检查，不在任务执行中检查 ——
        强杀在途任务会留下半成品数据。
        """
        log.info(
            "worker_started",
            worker_id=self._worker.worker_id,
            concurrency=self._worker.max_concurrency,
            poll_interval_s=self._worker.poll_interval_s,
            handlers=sorted(self._handlers),
        )
        try:
            while not self._shutdown.is_set():
                processed = await self.run_once()
                if processed == 0:
                    # 没捡到任务才睡。有任务时立刻继续下一批，吞吐更高。
                    # 用 sleep + 轮询标志而不是 wait_for(shutdown.wait())：
                    # wait_for 把等待实现成"可被取消的等待"，在某些平台
                    # 上会留下未取回的取消异常；这里只需要"睡一会儿再看标志"。
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            self._shutdown.wait(), timeout=self._worker.poll_interval_s
                        )
        finally:
            log.info(
                "worker_stopped",
                worker_id=self._worker.worker_id,
                claimed=self._stats.claimed,
                succeeded=self._stats.succeeded,
                failed=self._stats.failed,
                dead=self._stats.dead,
                reclaimed=self._stats.reclaimed,
            )


# ---------------------------------------------------------------- 错误处理辅助


def _error_code_of(exc: BaseException) -> str:
    """提取错误码（进 `tasks.error_code`）。

    用码名而非数字：任务表是内部表，排障时 `DOC_PARSE_FAILED`
    比 `31005` 有用得多。
    """
    if isinstance(exc, AppError):
        return exc.code_name
    return f"UNHANDLED_{type(exc).__name__}"


def _safe_message(exc: BaseException) -> str:
    """生成安全的错误描述（进 `tasks.error_message`）。

    **不能直接 `str(exc)` 入库**：异常消息可能含文件路径、SQL、用户数据。
    截断到 200 字符 + PII 脱敏是平衡点——完全丢弃会让排障只能靠猜。
    """
    return redact_text(str(exc)[:200])


__all__ = [
    "ProgressReporter",
    "SessionFactory",
    "TaskHandler",
    "TenantSessionFactory",
    "WorkerEngine",
    "WorkerStats",
]
