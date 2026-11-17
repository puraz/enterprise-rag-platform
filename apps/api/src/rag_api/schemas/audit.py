"""审计请求/响应实体与映射。

契约形状来自 `rag_contracts.api.audit`；本模块只做 re-export 与
ORM → 响应实体的映射。
"""

from __future__ import annotations

from rag_contracts.api.audit import AuditLogItem, AuditQueryResponse
from rag_db.models.audit import AuditLog


def to_item(record: AuditLog) -> AuditLogItem:
    """ORM → 审计列表项。"""
    return AuditLogItem(
        id=record.id,
        seq=record.seq,
        action=str(record.action),
        outcome=str(record.outcome),
        actor_user_id=record.actor_user_id,
        actor_email=record.actor_email,
        target_type=str(record.target_type) if record.target_type else None,
        target_id=record.target_id,
        target_name=record.target_name,
        request_id=record.request_id,
        ip_address=str(record.ip_address) if record.ip_address else None,
        error_code=record.error_code,
        details=record.details,
        created_at=record.created_at,
    )


def to_response(records: list[AuditLog], *, limit: int, offset: int) -> AuditQueryResponse:
    """一页审计记录 → 响应实体。"""
    items = [to_item(record) for record in records]
    return AuditQueryResponse(
        items=items,
        total=None,  # 默认不返回：审计表 COUNT(*) 很慢
        limit=limit,
        offset=offset,
        has_more=len(items) == limit,
    )


__all__ = ["AuditLogItem", "AuditQueryResponse", "to_item", "to_response"]
