"""审计查询的 HTTP 契约。

## 为什么审计项里保留 `seq` 与 `record_hash`

`seq` 是链内序号（A3-5 的哈希链）。前端展示时序用它，
而不是 `created_at`——同一毫秒的多条记录时间戳可能相同。

**不返回 `record_hash` 与 `prev_hash`**：哈希链用于服务端验证，
暴露给前端没有价值（前端无法独立验证，还增加响应体积）。
管理端需要"验证链完整性"时用单独的端点（A15）。

## 为什么 `details` 是 `dict[str, str | int | float | bool]`

审计 details 由 `_safe_details` 过滤过：只保留标量 + 短列表。
用强类型而不是 `Any`：让前端知道能拿到的形状。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuditLogItem(BaseModel):
    """一条审计记录。"""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    seq: int = Field(description="链内序号（同一租户内严格递增）")
    action: str = Field(description="动作类型，见 AuditAction")
    outcome: str = Field(description="结果：success/denied/failed")
    actor_user_id: uuid.UUID | None = Field(
        default=None, description="操作者 ID（用户删除后为 null，但记录保留）"
    )
    actor_email: str | None = Field(
        default=None, description="操作者邮箱快照（用户改名后仍可追溯）"
    )
    target_type: str | None = Field(default=None, description="目标对象类型")
    target_id: uuid.UUID | None = Field(default=None, description="目标对象 ID")
    target_name: str | None = Field(default=None, description="目标名称快照")
    request_id: str | None = Field(default=None, description="关联的请求 ID，可回查完整链路日志")
    ip_address: str | None = Field(default=None, description="客户端 IP")
    error_code: str | None = Field(default=None, description="失败/拒绝时的错误码")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="结构化详情（服务端已脱敏 PII，不含文档正文）",
    )
    created_at: datetime = Field(description="事件时间（UTC）")


class AuditQueryResponse(BaseModel):
    """审计查询响应。

    扁平结构（与 A4 的分页契约一致）：`{items, total, limit, offset, has_more}`。
    """

    model_config = ConfigDict(frozen=True)

    items: list[AuditLogItem]
    total: int | None = Field(
        default=None,
        description=(
            "总条数。**默认 null**：审计表很大，COUNT(*) 需要全表扫描。"
            "管理端的页码器改为「加载更多」即可"
        ),
    )
    limit: int
    offset: int
    has_more: bool = Field(description="是否还有下一页")
