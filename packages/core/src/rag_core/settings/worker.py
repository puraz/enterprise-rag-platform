"""Worker 配置。

## 与 API 配置的关系

Worker 复用 `Settings`（同一个 `RAG_*` 命名空间），但有自己的子配置：

| 配置 | 用途 |
|---|---|
| `worker_id` | 认领标识（任务表 `locked_by`）。多副本时必须唯一 |
| `poll_interval_s` | 数据库轮询间隔（ARQ 无任务时的兜底扫描） |
| `lock_timeout_s` | 任务认领超时（超过视为 worker 崩溃，可被回收） |
| `max_concurrency` | 同时处理的任务数 |

## 为什么 worker_id 必须唯一

`TaskRepository.complete` 的条件是 `WHERE id = ? AND locked_by = ?`。
如果两个 worker 用同一个 `locked_by`：
- worker A 认领任务后崩溃；
- 任务被回收重派给 worker B（同一 ID）；
- A 恢复后把任务标成成功 —— **它其实没处理**。

所以 `worker_id` 默认用 `hostname:pid`，加随机后缀防容器内 PID 重复。
"""

from __future__ import annotations

import os
import socket
import uuid

from pydantic import BaseModel, ConfigDict, Field


def default_worker_id() -> str:
    """生成默认 worker 标识。

    格式：`{hostname}:{pid}:{随机4位}`。

    加随机后缀的原因：容器重启后 PID 可能重复（都是 1），
    只靠 `hostname:pid` 会让新旧 worker 撞号。
    """
    hostname = socket.gethostname()[:32]
    return f"{hostname}:{os.getpid()}:{uuid.uuid4().hex[:4]}"


class WorkerSettings(BaseModel):
    """Worker 运行配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    worker_id: str = Field(default_factory=default_worker_id)
    """认领标识。**多副本时必须唯一**（见模块 docstring）。"""

    poll_interval_s: float = Field(default=2.0, gt=0)
    """数据库轮询间隔。

    为什么需要轮询（ARQ 已经是队列）：
    - Redis 可能丢 job（重启未持久化、内存淘汰）；
    - 延迟任务到点后 ARQ 会唤醒，但"任务被回收重派"的场景需要额外扫描；
    - 轮询是**兜底**，间隔可以宽松（2 秒够用，不该是 100ms）。
    """

    lock_timeout_s: int = Field(default=600, ge=60)
    """任务认领超时（秒）。

    超过这个时间仍在 running 的任务视为 worker 崩溃，可被回收重派。

    默认 600 秒（10 分钟）：一份 300 页 PDF 的解析+切分+向量化
    可能需要几分钟，设短了会把正常任务误判为卡死。
    """

    max_concurrency: int = Field(default=2, ge=1, le=16)
    """同时处理的任务数。

    默认 2 而非更高：
    - embedding 在 GPU 上串行更高效（批处理比并发快）；
    - PDF 解析是 CPU 密集，并发高只会互相抢核。
    按队列长度扩缩容的正确方式是**加 worker 副本**，不是加并发数。
    """

    max_retries: int = Field(default=3, ge=0, le=10)
    """单个任务的最大重试次数（超过进死信）。"""

    cleanup_interval_s: int = Field(default=300, ge=60)
    """清理任务的执行间隔（回收过期上传会话、孤儿 blob）。"""
