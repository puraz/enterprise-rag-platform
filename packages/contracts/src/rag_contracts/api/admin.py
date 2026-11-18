"""管理面契约：看板、任务台账、文档台账、索引版本、审计验链。

## 为什么管理面契约单独一个文件

管理面的读者与用户面不同：用户面为"上传/提问"服务，管理面为
"运营/审计/排障"服务。把两者混在一个文件里会让前端生成 TS 时
把所有管理模型都塞进用户端的类型包（用户端用不到，但会跟着漂移）。

## 设计边界：管理面**不返回文档正文**

文档台账只给元数据（标题、状态、计数、失败原因）。正文与 chunk
内容属于"数据面"，只在需要排障时通过专门的 chunk 预览接口取，
且该接口单独审计。理由是管理端用户可能没有该文档的读权限——
管理权限管的是"系统怎么运行"，不是"我能看什么内容"。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from rag_contracts.api.usage import UsageBucket
from rag_contracts.enums.documents import (
    ConfidentialityLevel,
    DocumentFormat,
    DocumentStatus,
    DocumentVersionStatus,
)
from rag_contracts.enums.retrieval import IndexVersionStatus
from rag_contracts.enums.uploads import TaskStatus, TaskType


class DashboardCounters(BaseModel):
    """看板的几个核心计数。

    刻意**不包含 QPS / P95 / 错误率**：那些来自 Prometheus（进程内指标
    在重启后归零，且多副本时单实例的数字没有意义）。
    这里只放"只能从数据库回答"的业务计数。
    """

    model_config = ConfigDict(frozen=True)

    knowledge_base_count: int = Field(ge=0, description="active 知识库数")
    document_count: int = Field(ge=0, description="未删除文档总数")
    documents_by_status: dict[str, int] = Field(
        default_factory=dict, description="按处理状态分组的文档数"
    )
    tasks_by_status: dict[str, int] = Field(
        default_factory=dict, description="按状态分组的任务数（看积压与死信）"
    )
    dead_letter_count: int = Field(ge=0, description="死信任务数（需人工介入）")
    denied_audit_count: int = Field(ge=0, description="统计窗口内被拒绝的操作数")
    total_cost_usd: float = Field(ge=0, description="统计窗口内的模型花费（USD）")


class DashboardResponse(BaseModel):
    """管理看板响应。"""

    model_config = ConfigDict(frozen=True)

    since: datetime = Field(description="统计窗口起点")
    counters: DashboardCounters
    cost_by_task_type: list[UsageBucket] = Field(
        default_factory=list, description="按功能拆解的成本（钱花在改写还是生成）"
    )
    generated_at: datetime = Field(description="本次统计的生成时间")


class AdminTaskItem(BaseModel):
    """任务台账项（管理端）。

    与用户面的 `TaskItem` 的差别：
    - 带 `tenant_id` 与 `locked_by`（跨租户排障必需）；
    - 带 `dead_at` 语义（status=dead 时可重试）。
    """

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    task_type: TaskType
    status: TaskStatus
    progress: int = Field(ge=0, le=100)
    retry_count: int = Field(ge=0)
    max_retries: int = Field(ge=0)
    idempotency_key: str = Field(description="幂等键（排障时用来判断是否重复入队）")
    document_id: uuid.UUID | None = None
    knowledge_base_id: uuid.UUID | None = None
    locked_by: str | None = Field(default=None, description="认领该任务的 worker 标识")
    locked_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class AdminDocumentItem(BaseModel):
    """文档台账项。

    **不含正文**（见模块 docstring）。`error_message` 已由 worker
    脱敏（`_safe_message`），可以展示给管理员排障。
    """

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    title: str
    original_filename: str
    format: DocumentFormat
    status: DocumentStatus
    confidentiality: ConfidentialityLevel
    chunk_count: int = Field(ge=0)
    page_count: int | None = None
    size_bytes: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    error_code: str | None = None
    error_message: str | None = None
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


class AdminDocumentVersionItem(BaseModel):
    """文档版本项（管理端，含快照配置）。

    比用户面的 `DocumentVersionItem` 多三样管理端排障必需的字段：
    切分策略快照、chunk 数、解析器实现名。
    """

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    version_no: int = Field(ge=1)
    status: DocumentVersionStatus
    is_current: bool
    chunk_count: int = Field(ge=0)
    parser_name: str | None = None
    chunking_strategy: str
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)
    embedding_model: str | None = None
    created_at: datetime


class AdminChunkItem(BaseModel):
    """chunk 预览项（切分质量检查）。

    **这是管理面唯一返回正文的接口**，且必须单独审计。
    `content` 会做长度截断（服务端配置上限），避免一次预览拉回整份文档。
    """

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    chunk_index: int = Field(ge=0)
    chunk_type: str
    chunk_level: int = Field(ge=0)
    content: str = Field(description="chunk 正文（可能被截断，见 is_truncated）")
    is_truncated: bool = Field(description="正文是否被截断（预览上限）")
    char_count: int = Field(ge=0, description="原始字符数（截断前）")
    token_count: int | None = None
    page_no: int | None = None
    section_path: list[str] = Field(default_factory=list)
    has_bbox: bool = Field(description="是否带版面坐标（前端能否框选）")
    # 是否已归属某个索引版本（旁表有向量）——判断"为什么检索不到它"
    embedding_model: str | None = None


class IndexVersionItem(BaseModel):
    """索引版本（零停机重建的状态机）。"""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    version_tag: str
    status: IndexVersionStatus
    embedding_model: str
    embedding_dim: int = Field(ge=1)
    index_type: str
    distance_metric: str
    chunk_count: int = Field(ge=0, description="已写入该版本的向量数")
    expected_chunk_count: int | None = Field(
        default=None, description="预期数量；chunk_count 达标才允许切换"
    )
    error_message: str | None = None
    activated_at: datetime | None = None
    created_at: datetime


class AuditChainVerification(BaseModel):
    """审计链完整性验证结果。

    ## 为什么这个接口值得单独存在

    审计表的哈希链只有在**被验证过**时才有意义。只写不验的链
    等于"我们相信没人改过数据库"——而威胁模型里恰恰包含内部人员。
    定期（或监管前）跑一次验证，是这条链存在的理由。

    统计口径：`checked` 条数受 `limit` 限制，`is_complete=False`
    时要明确告诉调用方"只验了前 N 条"，不能让 `verified=true` 被误读为
    "全表已验证"。
    """

    model_config = ConfigDict(frozen=True)

    verified: bool = Field(description="已检查的区间内链是否完整")
    checked_records: int = Field(ge=0, description="本次实际验证的记录数")
    is_complete: bool = Field(
        description="是否覆盖了该租户的全部审计记录（false 表示只验了前 limit 条）"
    )
    first_broken_seq: int | None = Field(
        default=None, description="第一个断点的链内序号（verified=false 时有值）"
    )
    reason: str | None = Field(default=None, description="断链原因的可读说明（不含记录内容）")


__all__ = [
    "AdminChunkItem",
    "AdminDocumentItem",
    "AdminDocumentVersionItem",
    "AdminTaskItem",
    "AuditChainVerification",
    "DashboardCounters",
    "DashboardResponse",
    "IndexVersionItem",
]
