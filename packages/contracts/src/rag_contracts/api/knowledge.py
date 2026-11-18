"""知识库接口契约。

## 为什么知识库需要独立的契约文件

知识库是"文档的容器"，也是权限的第一层（库级可见性 + 库级密级）。
它被三个地方消费：

| 消费方 | 需要什么 |
|---|---|
| 用户前端 | 列表、创建、改名、看到 `document_count` |
| 上传流程 | 校验目标库存在且可写、取默认切分参数 |
| 对话流程 | `knowledge_base_ids` 限定检索范围 |

三者字段需求不同，但形状必须一致——否则"前端传 `kb_id`、后端叫
`knowledgeBaseId`"这类漂移只会在联调时暴露。

## 为什么创建时**不接受** `visibility=department`

`KnowledgeBaseVisibility.DEPARTMENT` 是枚举里的既成值，但
`departments` 表**尚未建模**，`PermissionContext.department_ids` 恒为空。

如果创建接口放行 `department`，用户会建出一个**自己都看不到的库**
（`knowledge_base_visibility_filter` 里该分支显式不匹配）。
这是最糟的失败模式：数据建成功了，但谁都访问不到，且没有任何报错。

所以在契约层直接拒绝，错误信息说明"待组织架构建模后开放"。
枚举值保留（已发布不可删），只是在创建入口关掉。

## 为什么更新只允许改三个字段

`visibility` 与 chunking 参数创建后不可改：

- 改 `visibility` 会让"原本能看的人突然看不到"或反之，
  属于权限变更，该走专门的授权接口并留审计，而不是顺手 PATCH；
- 改 chunking 参数会让**已入库的 chunk 与新配置不一致**——
  同一份文档里既有 800 字块又有 400 字块，覆盖率统计与
  重建行为都变得无法解释。要换配置就重传新版本（版本快照会记下新参数）。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_contracts.enums.documents import (
    ConfidentialityLevel,
    DocumentFormat,
    DocumentStatus,
    DocumentVersionStatus,
    KnowledgeBaseStatus,
    KnowledgeBaseVisibility,
)

# 允许在创建时指定的可见性。`department` 不在其中：见模块 docstring。
_CREATABLE_VISIBILITIES = frozenset(
    {KnowledgeBaseVisibility.PRIVATE, KnowledgeBaseVisibility.TENANT}
)


class CreateKnowledgeBaseRequest(BaseModel):
    """创建知识库。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=200,
        description="库名。租户内唯一（同名返回 409）",
    )
    description: str | None = Field(
        default=None,
        max_length=2000,
        description="用途说明，展示在列表里",
    )
    visibility: KnowledgeBaseVisibility = Field(
        default=KnowledgeBaseVisibility.TENANT,
        description="可见范围：private（仅创建者）/ tenant（租户内公开）",
    )
    confidentiality: ConfidentialityLevel = Field(
        default=ConfidentialityLevel.INTERNAL,
        description=(
            "库级密级。文档密级不得低于它——否则机密文档能混进公开库，整个库变成一个随便问的入口"
        ),
    )

    # 切分参数：默认值与 `KnowledgeBase` 模型一致（recursive / 800 / 100）。
    # 放在创建时指定而不是事后调：切分参数是版本快照的一部分，
    # 同一文档的所有版本用同一套参数才可比。
    chunking_strategy: str = Field(
        default="recursive",
        max_length=64,
        description=(
            "切分策略：fixed / recursive / structured / sentence_window / parent_child / semantic"
        ),
    )
    chunk_size: int = Field(default=800, ge=100, le=4000, description="目标块大小（字符）")
    chunk_overlap: int = Field(default=100, ge=0, le=1000, description="相邻块重叠（字符）")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        """去掉首尾空白；全空白视为非法。

        `min_length=1` 拦不住全空格——它在 Python 里长度是 3，
        但入库后会变成一个"看不见名字"的知识库。
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("知识库名称不能为空")
        return stripped

    @field_validator("visibility")
    @classmethod
    def _reject_department(cls, value: KnowledgeBaseVisibility) -> KnowledgeBaseVisibility:
        """拒绝 department（见模块 docstring）。"""
        if value not in _CREATABLE_VISIBILITIES:
            raise ValueError(
                "暂不支持 department 可见性：组织架构（departments 表）尚未建模，"
                "建出来的库将无人可见。请使用 private 或 tenant"
            )
        return value

    @model_validator(mode="after")
    def _check_overlap(self) -> CreateKnowledgeBaseRequest:
        """重叠必须小于块大小。

        相等或更大意味着每个新块都包含上一块的全部内容——切分退化成
        "每滑动一点就产出一个块"，块数与成本爆炸，而检索精度毫无提升。
        DB 层也有同样的 CheckConstraint；这里提前拦是为了给出可读的错误信息
        （数据库约束报错只有约束名，前端无法据此提示）。
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


class UpdateKnowledgeBaseRequest(BaseModel):
    """更新知识库（只允许改元数据）。

    为什么不能改 visibility / chunking：见模块 docstring 末段。
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    confidentiality: ConfidentialityLevel | None = Field(
        default=None,
        description="调高密级会让原本可读的文档变不可读（安全方向）；调低需审计",
    )

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("知识库名称不能为空")
        return stripped

    @model_validator(mode="after")
    def _require_at_least_one(self) -> UpdateKnowledgeBaseRequest:
        """空 PATCH 是无意义的请求，直接拒绝。

        放行会让前端"什么都没改也发一次请求"变成习惯，白白产生审计噪音。
        """
        if self.name is None and self.description is None and self.confidentiality is None:
            raise ValueError("至少需要修改一个字段")
        return self


class KnowledgeBaseItem(BaseModel):
    """知识库列表项 / 详情。

    单对象就直接返回它（不包一层 `{knowledge_base: {...}}`）——
    成功响应只有三种形态：单对象 / 分页集合 / 无内容。

    ## 字段名与 DB 列名不同，映射在路由层显式做

    DB 叫 `default_chunk_size`（强调"新文档继承的默认值"），
    对外叫 `chunk_size`（对使用者来说就是这个库的切分大小）。
    不用 `validation_alias` 自动映射：那样 `KnowledgeBaseItem(...)`
    显式构造时报"字段缺失"（alias 只影响**解析**，不影响构造），
    错误信息还指着 `default_chunk_size` 这个前端从没见过的名字。
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None = None
    status: KnowledgeBaseStatus
    visibility: KnowledgeBaseVisibility
    confidentiality: ConfidentialityLevel
    document_count: int = Field(ge=0, description="库内文档数（不含已删除）")
    chunking_strategy: str = Field(description="默认切分策略（新文档继承它）")
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


# ------------------------------------------------------------------ 文档契约


class CreateDocumentRequest(BaseModel):
    """把已上传的 blob 登记为文档。

    ## 为什么上传与建文档分成两个接口

    两条路径在 complete 之后是**同一条**：

        POST /uploads → ... → POST /uploads/{id}/complete → {blob_id}
                                                            ↓
                                                   POST /documents

    秒传命中也只拿到 `blob_id`（没有分片、没有 upload_id），走的完全是
    最后这一步。如果建文档逻辑塞进 `complete`，秒传就得另写一份——
    两份代码迟早不一致。

    分开还有一个好处：`complete` 只做存储层的事（合并、校验、落 blob），
    `POST /documents` 只做业务层的事（建文档、建版本、入队）。
    职责单一，各自可独立测试。
    """

    model_config = ConfigDict(extra="forbid")

    blob_id: uuid.UUID = Field(description="已完成的物理文件 ID（来自上传 complete 或秒传）")
    knowledge_base_id: uuid.UUID = Field(description="目标知识库")
    title: str | None = Field(
        default=None,
        max_length=500,
        description="文档标题。省略时用 original_filename 去掉扩展名",
    )
    original_filename: str = Field(
        min_length=1,
        max_length=500,
        description="原始文件名。决定文档格式（扩展名）与审计记录",
    )
    confidentiality: ConfidentialityLevel | None = Field(
        default=None,
        description=(
            "文档密级。省略时继承知识库密级。只能更严不能更松：比库级更松会绕过库的密级约束"
        ),
    )

    @field_validator("original_filename")
    @classmethod
    def _strip_filename(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("文件名不能为空")
        return stripped


class DocumentVersionItem(BaseModel):
    """文档版本项。"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version_no: int = Field(ge=1)
    status: DocumentVersionStatus
    is_current: bool
    chunk_count: int = Field(ge=0)
    page_count: int | None = None
    parser_name: str | None = None
    created_at: datetime


class DocumentItem(BaseModel):
    """文档列表项。"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    title: str
    original_filename: str
    format: DocumentFormat
    status: DocumentStatus
    confidentiality: ConfidentialityLevel
    chunk_count: int = Field(ge=0)
    page_count: int | None = None
    size_bytes: int = Field(ge=0, description="原始文件大小")
    current_version_no: int = Field(ge=0, description="当前版本序号，0 表示尚无版本")
    error_code: str | None = Field(default=None, description="失败错误码（仅 failed 状态有值）")
    error_message: str | None = Field(
        default=None, description="失败原因（仅内部可见，不含文档正文）"
    )
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentItem):
    """文档详情：列表项 + 版本历史。

    ## 为什么版本历史只在详情里

    列表页一次拉 20 个文档，每个再带 5 个版本就是 100 个额外对象。
    版本历史只在"我要回滚 / 对比"时才需要，按需加载更合理。
    """

    versions: list[DocumentVersionItem] = Field(
        default_factory=list, description="版本列表（新版本在前）"
    )


class CreateDocumentResponse(BaseModel):
    """建文档的响应（202 Accepted 语义）。

    ## 为什么不是 201

    201 表示"资源已完整创建"。这里只创建了**文档元数据**，
    真正的内容（chunk / 向量）还在异步管线里。用 202 表达
    "已接受，处理是异步的"，前端据此决定"跳详情页并轮询"而不是
    "以为已经可检索了"。

    `task_id` 是轮询进度的抓手，也是排查"为什么文档一直 pending"的入口。
    """

    model_config = ConfigDict(frozen=True)

    document_id: uuid.UUID
    task_id: uuid.UUID
    status: DocumentStatus = Field(description="当前状态，正常为 pending")
    deduplicated_blob: bool = Field(
        default=False,
        description="该 blob 是否已被其它文档引用（秒传 / 重复上传同一文件）",
    )


class DocumentDownloadResponse(BaseModel):
    """文档下载地址。

    ## 为什么返回 URL 而不是直接流式返回文件

    两个后端（local / S3）的能力不同：

    | 后端 | 能签临时直链 | 说明 |
    |---|---|---|
    | S3 协议 | ✅ | 客户端直连存储，不经过 API 进程 |
    | local | ❌ | 只能由 API 代理读取 |

    统一契约用 `download_url` 表达："能拿到一个可 GET 的地址"。
    local 后端给出的是本服务的代理地址（同源，前端无需区分）。
    **签名 URL 一旦发出，有效期内任何人可访问** —— 所以签发前
    必须已完成权限与密级校验（在 service 里做，不在路由里）。
    """

    model_config = ConfigDict(frozen=True)

    document_id: uuid.UUID
    version_no: int = Field(ge=1, description="下载的是哪个版本")
    filename: str = Field(description="下载时的文件名（原始文件名）")
    content_type: str = Field(description="MIME 类型（以魔数校验结果为准）")
    size_bytes: int = Field(ge=0)
    download_url: str = Field(description="临时下载地址（GET 即可）")
    expires_in_s: int = Field(ge=1, description="URL 有效期（秒）")


class RollbackVersionRequest(BaseModel):
    """回滚到指定文档版本。

    ## 为什么要显式传 version_id 而不是"上一个版本"

    "回滚到上一版"在多版本场景下语义含糊（上一版可能是失败版本）。
    显式指定目标版本，配合详情接口返回的版本列表，用户的选择是明确的。

    ## 为什么这不是异步操作

    回滚只做三件事：把版本置 current、把旧版本的 chunk 置 `is_current=false`、
    把目标版本的 chunk 置回 `is_current=true`。**都是元数据操作，没有
    重新解析或向量化**——这正是 `DocumentVersion` 快照设计的价值。
    """

    model_config = ConfigDict(extra="forbid")

    version_id: uuid.UUID = Field(description="目标版本 ID（必须是该文档的 ready 版本）")


__all__ = [
    "CreateDocumentRequest",
    "CreateDocumentResponse",
    "CreateKnowledgeBaseRequest",
    "DocumentDetail",
    "DocumentDownloadResponse",
    "DocumentItem",
    "DocumentVersionItem",
    "KnowledgeBaseItem",
    "RollbackVersionRequest",
    "UpdateKnowledgeBaseRequest",
]
