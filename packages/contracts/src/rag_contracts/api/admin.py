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


class CreateIndexRebuildRequest(BaseModel):
    """发起索引重建。

    ## 为什么版本号由服务端生成，客户端不能指定

    `version_tag` 在同一知识库内唯一（`uq_index_versions_kb_tag`）。
    让客户端指定会让"两个运营同时发起"撞唯一约束，报错信息是
    "duplicate key"，而根因是并发——很难从错误里看出来。

    服务端用 `v{序号}-{时间戳}` 生成，天然不重复且可读。

    ## 为什么不需要传 embedding 模型

    模型来自当前进程配置（`settings.embedding`）。传参会让
    "运营以为换了模型，实际配置没改"这种情况变成静默的错版本——
    版本表里会记下一个与实际向量不符的 `embedding_model`。
    """

    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(
        default=None,
        max_length=500,
        description="重建原因（进审计）。例如：换 embedding 模型 / 调 chunk 参数",
    )


class IndexRebuildResponse(BaseModel):
    """重建已受理（202）。

    ## 为什么同时返回 task_id 与 index_version_id

    两者跟踪的是不同东西：

    - `task_id`：这次回填的执行进度（0-100，失败原因）；
    - `index_version_id`：这个版本的状态（building → ready → active）。

    只给 task_id，前端在任务成功后就不知道"该切哪个版本"了；
    只给 version_id，前端无法显示进度百分比。所以都给。
    """

    model_config = ConfigDict(frozen=True)

    task_id: uuid.UUID
    index_version_id: uuid.UUID
    version_tag: str
    status: IndexVersionStatus = Field(description="新版本初始状态，正常为 building")
    poll_url: str = Field(description="任务轮询地址（GET /api/v1/tasks/{task_id}）")


class ActivateIndexVersionRequest(BaseModel):
    """切换 ACTIVE 索引版本（alias 切换）。

    ## 为什么需要 `confirm_version_tag` 二次确认

    切换会让**所有检索立刻改用新向量**。如果新版本因为数据问题
    （比如某个文档在回填期间被删）导致召回变差，用户会立刻感受到，
    而定位问题需要时间。

    URL 里已经有 version_id，再加一个 tag 的确认看似冗余——但它防的是
    "前端把两个版本的 ID 搞反"这类失误（`.map` 里拿错字段很常见）。
    切换是不可逆感受的操作（虽然能回滚），值得一次输入确认。

    **必须与目标版本的 `version_tag` 完全一致**，否则 400。
    """

    model_config = ConfigDict(extra="forbid")

    confirm_version_tag: str = Field(
        min_length=1,
        max_length=64,
        description="再次输入目标版本的 version_tag 以确认切换",
    )
    reason: str | None = Field(default=None, max_length=500, description="切换原因（进审计）")


class RollbackIndexVersionRequest(BaseModel):
    """回滚到指定索引版本。

    回滚是**秒级**的：旧版本的向量行在切换时没有删除，只是把 ACTIVE
    指回去。这也是为什么"回收旧版本"必须延迟观察——立刻回收会让回滚
    失去可能。

    ## 约束

    - 目标版本必须属于该知识库；
    - 目标版本必须是 `ready` 或 `active`（`building` 未完成、
      `failed` 没数据，都不能作为回滚目标）；
    - 同样需要 `confirm_version_tag`（同上：防止 ID 拿错）。
    """

    model_config = ConfigDict(extra="forbid")

    confirm_version_tag: str = Field(
        min_length=1,
        max_length=64,
        description="再次输入目标版本的 version_tag 以确认回滚",
    )
    reason: str | None = Field(default=None, max_length=500, description="回滚原因（进审计）")


class TenantCostSummary(BaseModel):
    """单个租户的成本聚合（跨租户汇总用）。"""

    model_config = ConfigDict(frozen=True)

    tenant_id: uuid.UUID
    tenant_slug: str
    tenant_name: str
    call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)


class CrossTenantCostResponse(BaseModel):
    """全平台成本汇总（仅平台角色）。

    ## 为什么这是一个独立接口而不是看板的 `all_tenants` 开关

    `docs/adr/0011` 明确记过：**不做成开关**。理由：

    看板的默认行为是"看我当前租户"，开关一旦存在，被误开（前端默认值
    写错、URL 复制）就会让租户管理员看到全平台数据——而且返回 200，
    没有任何异常信号。

    独立接口 + 独立能力（`ADMIN`，仅平台超管 / 租户管理员）+
    强制审计，让"跨租户看数"这件事在代码里显式可见。

    ## 为什么只给租户维度合计，不给明细分页

    明细（每次调用的记录）已经在 `model_calls` 表里且有 `tenant_id`，
    真正的对账走 `scripts/audit_integrity.py`（直连 DB，不受 API 限制）。
    这里只给"哪个租户花了多少"——那是运营做容量规划需要的形状。
    """

    model_config = ConfigDict(frozen=True)

    since: datetime = Field(description="统计窗口起点")
    until: datetime | None = Field(default=None, description="统计窗口终点，null 表示到现在")
    total_cost_usd: float = Field(ge=0, description="全平台合计")
    total_calls: int = Field(ge=0)
    tenant_count: int = Field(ge=0, description="有调用记录的租户数")
    items: list[TenantCostSummary] = Field(default_factory=list, description="按花费倒序的租户明细")


__all__ = [
    "ActivateIndexVersionRequest",
    "AdminChunkItem",
    "AdminDocumentItem",
    "AdminDocumentVersionItem",
    "AdminTaskItem",
    "AuditChainVerification",
    "CreateIndexRebuildRequest",
    "CrossTenantCostResponse",
    "DashboardCounters",
    "DashboardResponse",
    "IndexRebuildResponse",
    "IndexVersionItem",
    "RollbackIndexVersionRequest",
    "TenantCostSummary",
]
