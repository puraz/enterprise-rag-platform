"""管理面响应实体与映射。

契约形状来自 `rag_contracts.api.admin`；本模块做 re-export 与映射。

## 为什么聚合映射（看板）单独写在这里而不是塞进路由

看板的输入是"五六个仓储方法的返回值"，输出是一个嵌套模型。
这段"拼装"逻辑有明确的测试价值（计数缺一个、成本算错、状态字典
漏分组），放在路由里就只能通过 HTTP 测。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from rag_contracts.api.admin import (
    AdminChunkItem,
    AdminDocumentItem,
    AdminDocumentVersionItem,
    AdminTaskItem,
    AuditChainVerification,
    DashboardCounters,
    DashboardResponse,
    IndexVersionItem,
)
from rag_contracts.api.usage import UsageBucket
from rag_contracts.enums import IndexVersionStatus, ModelTaskType, TaskStatus, TaskType
from rag_db.models.chunk import Chunk, IndexVersion
from rag_db.models.document import Document, DocumentVersion
from rag_db.models.task import Task

# chunk 预览的正文截断上限。
# 一个 chunk 默认 800 字符，超长块（受保护类型如表格）可能是几 KB。
# 预览接口一次返回 20 条时，不截断可能到 200KB —— 浏览器渲染会卡。
_CHUNK_PREVIEW_LIMIT = 4000


def to_dashboard_response(
    *,
    tenant_id: uuid.UUID,
    since: datetime,
    counters: dict[str, Any],
    breakdown: Sequence[dict[str, Any]],
) -> DashboardResponse:
    """把各仓储的聚合结果拼成看板响应。

    `breakdown` 的元素来自 `ModelCallRepository.breakdown()`：
    字段：`task_type` / `provider` / `model` / `call_count` / `input_tokens` /
    `output_tokens` / `cached_tokens` / `cost_usd`。
    """
    buckets: list[UsageBucket] = []
    for row in breakdown:
        buckets.append(
            UsageBucket(
                task_type=ModelTaskType(str(row["task_type"])),
                provider=str(row["provider"]),
                model=str(row["model"]),
                call_count=int(row["call_count"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                cached_tokens=int(row["cached_tokens"]),
                cost_usd=float(row["cost_usd"]),
            )
        )

    return DashboardResponse(
        since=since,
        counters=DashboardCounters(
            knowledge_base_count=int(counters["knowledge_base_count"]),
            document_count=int(counters["document_count"]),
            documents_by_status={
                str(k): int(v) for k, v in counters["documents_by_status"].items()
            },
            tasks_by_status={str(k): int(v) for k, v in counters["tasks_by_status"].items()},
            dead_letter_count=int(counters["dead_letter_count"]),
            denied_audit_count=int(counters["denied_audit_count"]),
            total_cost_usd=float(counters["total_cost_usd"]),
        ),
        cost_by_task_type=buckets,
        generated_at=datetime.now(UTC),
    )


def to_admin_task(task: Task) -> AdminTaskItem:
    """ORM → 管理端任务项。"""
    return AdminTaskItem(
        id=task.id,
        tenant_id=task.tenant_id,
        task_type=TaskType(str(task.task_type)),
        status=TaskStatus(str(task.status)),
        progress=task.progress,
        retry_count=task.retry_count,
        max_retries=task.max_retries,
        idempotency_key=task.idempotency_key,
        document_id=task.document_id,
        knowledge_base_id=task.knowledge_base_id,
        locked_by=task.locked_by,
        locked_at=task.locked_at,
        error_code=task.error_code,
        error_message=task.error_message,
        created_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )


def to_admin_document(doc: Document) -> AdminDocumentItem:
    """ORM → 文档台账项。**不含正文**。"""
    return AdminDocumentItem(
        id=doc.id,
        knowledge_base_id=doc.knowledge_base_id,
        title=doc.title,
        original_filename=doc.original_filename,
        format=doc.format,
        status=doc.status,
        confidentiality=doc.confidentiality,
        chunk_count=doc.chunk_count,
        page_count=doc.page_count,
        size_bytes=doc.size_bytes,
        retry_count=doc.retry_count,
        error_code=doc.error_code,
        error_message=doc.error_message,
        created_by=doc.created_by,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def to_admin_version(version: DocumentVersion) -> AdminDocumentVersionItem:
    """ORM → 管理端版本项。

    `chunking_config` 是 JSONB（`{"chunk_size":..., "chunk_overlap":...}`），
    拆成两个标量字段返回：前端要在表格里按大小排序，嵌套对象做不到。
    """
    config = version.chunking_config or {}
    return AdminDocumentVersionItem(
        id=version.id,
        version_no=version.version_no,
        status=version.status,
        is_current=version.is_current,
        chunk_count=version.chunk_count,
        parser_name=version.parser_name,
        chunking_strategy=version.chunking_strategy,
        chunk_size=int(config.get("chunk_size", 0)) or 1,
        chunk_overlap=int(config.get("chunk_overlap", 0)),
        embedding_model=version.embedding_model,
        created_at=version.created_at,
    )


def to_admin_chunk(chunk: Chunk) -> AdminChunkItem:
    """ORM → chunk 预览项（正文按上限截断）。"""
    content = chunk.content or ""
    truncated = len(content) > _CHUNK_PREVIEW_LIMIT
    return AdminChunkItem(
        id=chunk.id,
        chunk_index=chunk.chunk_index,
        chunk_type=str(chunk.chunk_type),
        chunk_level=chunk.chunk_level,
        content=content[:_CHUNK_PREVIEW_LIMIT],
        is_truncated=truncated,
        char_count=len(content),
        token_count=chunk.token_count,
        page_no=chunk.page_no,
        section_path=list(chunk.section_path or []),
        has_bbox=chunk.bbox is not None,
        embedding_model=chunk.embedding_model,
    )


def to_index_version(version: IndexVersion) -> IndexVersionItem:
    """ORM → 索引版本项。"""
    return IndexVersionItem(
        id=version.id,
        knowledge_base_id=version.knowledge_base_id,
        version_tag=version.version_tag,
        status=IndexVersionStatus(str(version.status)),
        embedding_model=version.embedding_model,
        embedding_dim=version.embedding_dim,
        index_type=version.index_type,
        distance_metric=version.distance_metric,
        chunk_count=version.chunk_count,
        expected_chunk_count=version.expected_chunk_count,
        error_message=version.error_message,
        activated_at=version.activated_at,
        created_at=version.created_at,
    )


def to_chain_verification(
    *, verified: bool, checked: int, total: int, first_broken_seq: int | None
) -> AuditChainVerification:
    """链验证结果 → 响应实体。

    `is_complete` 用"检查条数 == 租户总数"判断。**必须显式表达**：
    只验前 1000 条（默认 limit）时返回 `verified=true` 会被读成
    "全表已验证"，而那可能掩盖第 1001 条之后的篡改。
    """
    return AuditChainVerification(
        verified=verified,
        checked_records=checked,
        is_complete=checked >= total,
        first_broken_seq=first_broken_seq,
        reason=None
        if verified
        else f"链在第 {first_broken_seq} 条处断开（prev_hash 或 record_hash 不匹配）",
    )


__all__ = [
    "to_admin_chunk",
    "to_admin_document",
    "to_admin_task",
    "to_admin_version",
    "to_chain_verification",
    "to_dashboard_response",
    "to_index_version",
]
