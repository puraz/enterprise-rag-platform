"""上传会话与任务域枚举。

对应模型：`rag_db.models.upload` / `task`。
"""

from __future__ import annotations

from enum import StrEnum


class UploadSessionStatus(StrEnum):
    """分片上传会话状态机。

        INITIATED ──▶ UPLOADING ──▶ COMPLETING ──▶ COMPLETED
            │             │              │
            │             │              └──▶ FAILED（合并/校验失败）
            │             └──▶ EXPIRED（超时未完成）
            └──▶ ABORTED（用户取消）

    关键约束：
    - COMPLETED 之后不能再收分片（幂等返回成功）；
    - EXPIRED 与 ABORTED 的分片由清理任务回收；
    - COMPLETING 是"合并中"的中间态，防止两个请求同时 complete。
    """

    INITIATED = "initiated"
    """已创建，尚未上传任何分片。"""

    UPLOADING = "uploading"
    """至少收到一个分片。"""

    COMPLETING = "completing"
    """正在合并分片（此状态下禁止再上传分片）。"""

    COMPLETED = "completed"
    """合并完成，blob 已落库。"""

    FAILED = "failed"
    """合并或校验失败。"""

    EXPIRED = "expired"
    """超过有效期未完成，分片待回收。"""

    ABORTED = "aborted"
    """用户主动取消。"""


class TaskType(StrEnum):
    """异步任务类型。

    值必须与 worker 侧的任务注册表一致——不一致时报错是
    "未知任务类型"，但实际是名字拼写不同，很难定位。
    """

    PARSE_DOCUMENT = "parse_document"
    """解析：blob → 纯文本 + 位置信息。"""

    CHUNK_DOCUMENT = "chunk_document"
    """切分：文本 → chunks。"""

    EMBED_DOCUMENT = "embed_document"
    """向量化：chunks → embeddings。"""

    INDEX_DOCUMENT = "index_document"
    """写入索引。"""

    REBUILD_INDEX = "rebuild_index"
    """索引重建（A10）。"""

    CLEANUP_UPLOAD_SESSIONS = "cleanup_upload_sessions"
    """清理过期的上传会话。"""

    CLEANUP_ORPHAN_BLOBS = "cleanup_orphan_blobs"
    """回收无引用的 blob。"""

    DISPATCH_OUTBOX = "dispatch_outbox"
    """转发 outbox 消息。"""


class TaskStatus(StrEnum):
    """任务执行状态。

    与 `DocumentStatus` 分开：任务表记录"某次执行的进展"，
    文档表记录"业务上的最终状态"。一次文档处理可能产生多次任务执行
    （每次重试一条），管理端需要看到历史。
    """

    PENDING = "pending"
    """待执行，可被 worker 认领。"""

    RUNNING = "running"
    """执行中（已认领）。"""

    SUCCEEDED = "succeeded"
    """执行成功。"""

    FAILED = "failed"
    """执行失败，等待重试或进入死信。"""

    DEAD = "dead"
    """超过重试上限，进入死信队列，需人工介入。"""

    CANCELLED = "cancelled"
    """被取消（用户或系统）。"""


class OutboxStatus(StrEnum):
    """outbox 消息状态。

    至少一次送达：只要消息没有标记为 dispatched，就会一直被轮询重发。
    消费者必须幂等——这是"至少一次"语义的必然要求。
    """

    PENDING = "pending"
    """待转发。"""

    DISPATCHED = "dispatched"
    """已成功投递到队列。"""

    FAILED = "failed"
    """投递失败次数超限，需人工介入。"""
