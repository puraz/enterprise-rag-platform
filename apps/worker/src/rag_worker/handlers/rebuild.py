"""索引重建任务处理器：整个知识库的向量重算（零停机）。

## 这个 handler 与其它 handler 的根本不同

其它 handler（parse / chunk / embed）都是**单文档、单事务**：

    引擎给一个会话 → handler 干活 → 引擎提交

索引重建不行：一个库可能有十万 chunk，回填几十分钟。整段一个事务会
同时带来三个问题：

| 问题 | 现象 |
|---|---|
| 连接被长时间占用 | 阻塞 vacuum；连接池很快被多个重建任务耗尽 |
| 崩溃丢全部进度 | 重跑从头再来（几十分钟白算） |
| `locked_at` 不更新 | 超 `lock_timeout` 被判卡死 → **两个 worker 同时回填** |

所以本 handler 声明 `owns_transaction = True`，自己按批开短事务。

## 锁续期是必须的（不是优化）

`reclaim_stale` 的判据是 `locked_at < now() - lock_timeout`。
引擎只在认领时写一次 `locked_at`。如果重建跑 20 分钟而
`lock_timeout` 是 600 秒（10 分钟），第 10 分钟这个任务就会被
另一个 worker 抢走并重跑。

后果不是"慢一点"，而是：

- 两个进程同时调 embedding API（花双倍的钱）；
- 两个进程同时写 `chunk_embeddings`（靠唯一约束不会脏写，但会互相阻塞）；
- 日志里出现两份"index_backfill_batch"，排障时完全看不懂。

所以每批之后都要 `touch_lock()` 刷新 `locked_at`。

## 进度为什么按"已完成 chunk / 总数"算

`ProgressReporter` 只接受 0-99（100 归 `complete()`）。回填进度是
**可精确计算**的（旁表 COUNT vs 知识库 chunk 总数），不用估算。
精确进度让运营能判断"这个重建还要多久"，这直接影响他们敢不敢在
工作时间发起重建。

## 幂等

判据是**旁表里没有该版本的行**（`_claim_batch` 的 EXISTS 子查询），
不是计数器。所以：

- 重派后继续跑：已回填的 chunk 不会被再次选中；
- 崩溃后重启：从断点继续，不必从头再来；
- 两个 worker 抢同一版本：靠 `FOR UPDATE SKIP LOCKED` 不重复处理同一批。

这是"至少一次"语义下的正确做法——计数器和状态位都会漂移，旁表事实不会。
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from rag_adapters.providers.embedding import (
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingTask,
)
from rag_core.logging import get_logger
from rag_db.models.chunk import IndexVersion
from rag_db.repositories import IndexVersionRepository
from rag_db.repositories.chunk import ChunkRepository

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

    TenantSessionFactory = Callable[[uuid.UUID], AbstractAsyncContextManager[AsyncSession]]

log = get_logger(__name__)

# 单批处理的 chunk 数。
#
# 与 embed handler 的 batch_size 分开：那个是"一个文档一次算多少"，
# 这里是"一次提交多少"。调大能减少事务次数，但崩溃时丢的进度更多，
# 且单次 API 调用过大容易触发供应商的请求体上限。
DEFAULT_BATCH_SIZE = 32


class RebuildIndexHandler:
    """`TaskType.REBUILD_INDEX` 的处理器。

    payload：
        {
            "knowledge_base_id": "...",
            "index_version_id": "...",   # 由 API 在发起重建时创建好
        }

    ## 为什么版本由 API 创建，而不是 handler 创建

    API 需要把 `index_version_id` 立刻返回给前端（202 + 可轮询的版本 ID）。
    如果等 handler 起来才创建，前端拿到 task_id 却查不到版本，
    轮询期间只能显示"准备中"——那段时间无法判断"是否已经开始回填"。

    而且创建版本是廉价的元数据写入（一次 INSERT），放 API 侧能让
    "参数校验失败"在同步阶段就返回 422，而不是变成一条失败的任务。
    """

    # 声明自管事务：引擎会传 session_factory 而不是 session
    owns_transaction = True

    def __init__(
        self,
        *,
        provider: EmbeddingProvider,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._provider = provider
        self._batch_size = batch_size

    async def __call__(
        self,
        *,
        session: AsyncSession | None,
        task: Any,
        progress: Callable[[int, str], Awaitable[None]],
        session_factory: TenantSessionFactory | None = None,
    ) -> None:
        """执行回填。

        ## 参数

        - `session`: 断言为 None —— 本 handler 自管事务，引擎不该传会话。
          保留参数是为了与 `TaskHandler` 协议兼容（其它 handler 需要它）。
        - `session_factory`: 租户会话工厂（RLS 生效）。每批开一个新事务。
        """
        if session_factory is None:
            # 引擎没按 owns_transaction 分支传工厂 → 配置错误，
            # 重试也不会好（同一个 handler 同一个引擎）。
            raise _permanent("重建处理器缺少 session_factory（引擎未按 owns_transaction 分派）")
        if session is not None:
            raise _permanent("重建处理器不应收到已开启的会话（会让整段回填落在一个事务里）")

        payload = dict(task.payload or {})
        kb_id = _require_uuid(payload, "knowledge_base_id")
        version_id = _require_uuid(payload, "index_version_id")
        tenant_id: uuid.UUID = task.tenant_id

        await progress(5, "loading_version")

        # ---- 读版本（短事务，读完即关）----
        async with session_factory(tenant_id) as session_:
            version = await session_.get(IndexVersion, version_id)
            if version is None:
                raise _permanent(f"索引版本 {version_id} 不存在")
            if version.knowledge_base_id != kb_id:
                raise _permanent(f"索引版本 {version_id} 不属于知识库 {kb_id}")
            if version.embedding_model != self._provider.name:
                raise _permanent(
                    f"版本使用模型 {version.embedding_model}，"
                    f"当前 provider 是 {self._provider.name}；不能混算"
                )
            if version.embedding_dim != self._provider.capabilities.dimension:
                raise _permanent(
                    f"版本维度 {version.embedding_dim}，"
                    f"当前模型维度 {self._provider.capabilities.dimension}；不能混算"
                )
            expected = version.expected_chunk_count
            tag = version.version_tag

        # ---- 逐批回填 ----
        #
        # 每批：开事务 → 取一批未回填的 chunk → embed → 写旁表 → 提交 → 续锁
        #
        # 顺序很关键：**先提交再续锁**。反过来的话，续锁成功但提交失败
        # 会让"锁是新的、进度是旧的"，重派后从头扫一遍（幂等，但浪费）。
        embedded = 0
        batch_no = 0
        while True:
            async with session_factory(tenant_id) as session_:
                pending = await ChunkRepository(session_).claim_batch_for_version(
                    index_version_id=version_id,
                    tenant_id=tenant_id,
                    knowledge_base_id=kb_id,
                    batch_size=self._batch_size,
                )
                if not pending:
                    break

                texts = [chunk.embedding_input or chunk.content for chunk in pending]
                # 供应商错误的 retryable 属性决定引擎是否重试（协议属性，不用 isinstance）：
                # 限流/超时该重试，参数/模型不存在不该。这里不吞，直接抛给引擎。
                result = await self._provider.embed_batch(texts, task=EmbeddingTask.DOCUMENT)

                # **顺序必须严格对应**：错位会让向量与内容不匹配且不报错。
                # strict=True 让长度不匹配立刻抛错而不是静默截断。
                pairs = list(zip(pending, result.vectors, strict=True))
                await ChunkRepository(session_).set_embeddings(
                    updates=[(chunk.id, vector, self._provider.name) for chunk, vector in pairs],
                    index_version_id=version_id,
                    tenant_id=tenant_id,
                )
                # 提交在本批结束时发生（session_factory 退出时 commit）

            embedded += len(pairs)
            batch_no += 1

            # 续锁：必须每批做一次，否则超过 lock_timeout 会被误判为卡死。
            #
            # 返回 False 说明锁已不属于本进程（任务被回收并派给了别的 worker）。
            # 此时**必须立刻停止**：继续跑会让两个进程同时回填同一版本，
            # 白花一倍的 embedding 费用，且日志里两份进度互相打架。
            if not await self._touch_lock(task=task, session_factory=session_factory):
                log.warning(
                    "index_rebuild_lock_lost",
                    task_id=str(task.id),
                    index_version_id=str(version_id),
                    batch=batch_no,
                    embedded=embedded,
                    note="锁已被回收（任务可能被重派给其它 worker），本进程停止回填",
                )
                return

            # 进度：0-99（100 归 complete()）
            if expected:
                percent = min(99, 5 + int(90 * min(embedded, expected) / expected))
            else:
                percent = min(95, 5 + batch_no)
            await progress(percent, f"backfill:{embedded}")

            log.info(
                "index_rebuild_batch",
                task_id=str(task.id),
                index_version_id=str(version_id),
                knowledge_base_id=str(kb_id),
                batch=batch_no,
                embedded=embedded,
                expected=expected,
            )

        # ---- 标记版本 ready ----
        #
        # **chunk_count 必须按旁表真实行数重算**，不能用 embedded：
        # 重派场景下 embedded 只统计本轮新算的，而旁表里可能已有上一轮的行。
        # 用 embedded 会让 chunk_count 偏小，activate 的完成度校验就失效了
        # （它比对 chunk_count 与 expected，偏小会拒绝切换——虽然安全，
        # 但表现为"明明回填完了却切不了"）。
        async with session_factory(tenant_id) as session_:
            actual = await IndexVersionRepository(session_).count_version_chunks(
                index_version_id=version_id
            )
            await IndexVersionRepository(session_).mark_ready(
                version_id=version_id, chunk_count=actual
            )

        await progress(99, "ready")
        log.info(
            "index_rebuild_completed",
            task_id=str(task.id),
            index_version_id=str(version_id),
            version_tag=tag,
            knowledge_base_id=str(kb_id),
            embedded_this_run=embedded,
            version_chunk_count=actual,
            expected=expected,
            note="版本已 ready，等待人工 activate 切换",
        )

        if expected is not None and actual < expected:
            # 回填"跑完了"但数量不够：说明有 chunk 缺少向量（源数据问题或
            # 并发删文档）。**不报错**——activate 会再校验一次并给出明确
            # 拒绝理由；这里只留日志，让运营看到"为什么切不了"。
            log.warning(
                "index_rebuild_incomplete",
                task_id=str(task.id),
                index_version_id=str(version_id),
                expected=expected,
                actual=actual,
                hint="activate 会拒绝切换；检查是否有文档在回填期间被删",
            )

    async def _touch_lock(self, *, task: Any, session_factory: TenantSessionFactory) -> bool:
        """刷新任务锁（防止长任务被误判为卡死）。

        ## 为什么用租户会话而不是 admin 连接

        `tasks` 在 `RLS_TABLES` 里，但租户会话按 `tenant_id` 过滤后
        恰好能更新**本任务自己**那一行。用 admin（BYPASSRLS）也能做，
        但那是特权路径——能不开就不开。

        ## 返回值：False 表示锁已丢失，调用方必须停止

        `touch_lock` 带 `worker_id` 与 `status=running` 条件：任务被回收
        并重派后这两个条件不再匹配，更新 0 行 → 返回 False。

        ## 为什么异常时返回 True（而不是 False）

        数据库抖动导致的续锁失败**不代表锁已丢**——那只是这一次没刷新成功，
        下一批还会再试。返回 False 会让一个本来正常的长任务因为一次
        网络抖动就停止，而停止后它还得等 `lock_timeout` 才被重派，
        等于白等十分钟。

        所以区分两种失败：
        - 更新 0 行（锁不属于我了）→ False，必须停；
        - 抛异常（这次没刷上）→ True，继续跑，下次再试。
        """
        from rag_db.repositories import TaskRepository

        try:
            async with session_factory(task.tenant_id) as session:
                return await TaskRepository(session).touch_lock(
                    task_id=task.id, worker_id=str(task.locked_by or "")
                )
        except Exception as exc:
            log.warning(
                "task_lock_touch_failed",
                task_id=str(task.id),
                error_type=type(exc).__name__,
                note="本次续锁未成功；仍继续回填，下批再试（锁未刷新才会被回收）",
            )
            return True


def _require_uuid(payload: dict[str, Any], key: str) -> uuid.UUID:
    """从 payload 取 UUID，缺失/非法抛**永久**失败（重试无用）。"""
    raw = payload.get(key)
    if not raw:
        raise _permanent(f"任务 payload 缺少 {key}")
    try:
        return uuid.UUID(str(raw))
    except ValueError as exc:
        raise _permanent(f"任务 payload 的 {key} 不是合法 UUID") from exc


def _permanent(message: str) -> Exception:
    """永久失败：需要人介入（重试一千次也是同样结果）。"""
    return EmbeddingProviderError(message, retryable=False)


__all__ = ["DEFAULT_BATCH_SIZE", "RebuildIndexHandler"]
