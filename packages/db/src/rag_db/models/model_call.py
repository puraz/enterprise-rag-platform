"""模型调用明细。

## 为什么单独一张表而不是塞进 tasks

一次问答可能触发 5-8 次模型调用（改写、多查询×3、HyDE、rerank、生成），
每次调用的供应商/模型/单价/耗时都不同。塞进任务表会让表宽到不可维护，
且无法做"按模型维度聚合成本"。

## 成本账要平的三个前提

1. **每次付费调用都落一行**，包括失败的（有些供应商失败也计费，
   尤其 content_filtered）。漏记就是账不平。
2. **单价快照**：存 `unit_price_*`，不只存 `cost_usd`。供应商调价后
   仍能重算验证"当时按什么价算的"。
3. **缓存命中单独记录**：`cached_tokens` 通常按更低的单价计费。
   不区分会让成本估算偏高 30%+（prompt caching 命中率高的场景）。

## 精度问题

`cost_usd` 用 `Numeric(16, 8)` 而不是 float：浮点数累加 10 万条记录会有
可见误差（0.01 量级的钱对不上账）。Numeric 是精确十进制。

`Numeric` 在 Python 侧映射为 `Decimal`，不要和 float 混算。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from rag_contracts.enums.audit import ModelCallStatus, ModelTaskType
from rag_db.base import (
    Base,
    TenantScopedMixin,
    UUIDPrimaryKeyMixin,
)


class ModelCall(Base, UUIDPrimaryKeyMixin, TenantScopedMixin):
    """一次模型调用的完整明细。

    只追加，不修改。失败的调用也要记（见模块 docstring）。
    """

    __tablename__ = "model_calls"

    # ---- 谁在什么场景调用 ----
    task_type: Mapped[ModelTaskType] = mapped_column(
        String(32),
        nullable=False,
        comment="任务类型：chat/query_rewrite/embedding/rerank/evaluation",
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        comment="触发者（无 FK：用户删除后成本记录仍需保留）",
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        comment="关联会话（无 FK：会话可删除，成本记录保留）",
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        comment="关联消息",
    )
    request_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="请求 ID，可串联同一次请求内的多次调用",
    )

    # ---- 用了什么 ----
    provider: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="供应商标识（deepseek/openai/local-bge/…）",
    )
    model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="模型名（供应商侧的实际模型标识）",
    )
    # 是否降级产生：主模型不可用时走了备用模型，成本统计要能区分
    is_fallback: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="false",
        comment="是否为降级调用（主模型失败后走备用）",
    )
    cache_hit: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="false",
        comment="是否命中我方缓存（命中则不计费，但仍记录以统计命中率）",
    )

    # ---- 用量 ----
    input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="输入 token 数",
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="输出 token 数（embedding/rerank 为 0）",
    )
    cached_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="其中命中供应商 prompt cache 的 token 数（按更低单价计费）",
    )

    # ---- 单价快照（见模块 docstring）----
    # 单位统一为"每百万 token 的美元价"，与主流供应商报价口径一致
    unit_price_input: Mapped[Decimal | None] = mapped_column(
        Numeric(16, 8),
        nullable=True,
        comment="输入单价快照（USD / 1M tokens）",
    )
    unit_price_output: Mapped[Decimal | None] = mapped_column(
        Numeric(16, 8),
        nullable=True,
        comment="输出单价快照（USD / 1M tokens）",
    )
    unit_price_cached: Mapped[Decimal | None] = mapped_column(
        Numeric(16, 8),
        nullable=True,
        comment="缓存命中 token 的单价快照（USD / 1M tokens）",
    )
    cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(16, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
        comment="本次调用成本（USD，精确十进制，避免浮点累加误差）",
    )

    # ---- 性能 ----
    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="端到端延迟（毫秒），含重试时记录最后一次成功的耗时",
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="此调用前的重试次数（来自 ResilientCaller 统计）",
    )

    # ---- 结果 ----
    status: Mapped[ModelCallStatus] = mapped_column(
        String(32),
        nullable=False,
        default=ModelCallStatus.SUCCESS,
        server_default=ModelCallStatus.SUCCESS.value,
        comment="调用结果",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="失败错误码",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="失败详情（**禁止含 prompt 内容**：供应商错误可能回显请求体）",
    )

    # 观测关联：Langfuse trace ID（A14）
    trace_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="外部追踪系统的 trace ID（Langfuse），便于联查",
    )
    # 可公开的额外元数据（如 temperature、max_tokens）
    call_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="调用参数快照（temperature 等），禁止含 prompt 正文",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        nullable=False,
        comment="调用发生时间",
    )

    __table_args__ = (
        # 成本看板主查询：按租户 + 时间范围聚合
        Index("ix_model_calls_tenant_created", "tenant_id", "created_at"),
        # 按任务类型拆解成本（"检索改写花了多少"）
        Index("ix_model_calls_tenant_task_type", "tenant_id", "task_type", "created_at"),
        # 按模型拆解成本（"deepseek-chat 本月花了多少"）
        Index("ix_model_calls_tenant_model", "tenant_id", "provider", "model", "created_at"),
        # 日聚合任务扫描：WHERE created_at >= 昨天 且未聚合
        # （聚合状态记在单独的日汇总表里，这里只提供时间索引）
        Index("ix_model_calls_unaggregated", "created_at"),
        # 排障：找失败调用
        Index(
            "ix_model_calls_failures",
            "tenant_id",
            "created_at",
            postgresql_where=text("status != 'success'"),
        ),
    )
