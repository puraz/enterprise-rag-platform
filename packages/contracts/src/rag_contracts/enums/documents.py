"""知识库与文档域枚举。

对应模型：`rag_db.models.knowledge` / `document` / `blob`。
迁移任何一项状态流转前，先看这里的注释——状态机是文档管线的核心约束。
"""

from __future__ import annotations

from enum import StrEnum


class KnowledgeBaseStatus(StrEnum):
    """知识库状态。"""

    ACTIVE = "active"

    ARCHIVED = "archived"
    """归档：保留数据与索引，不再允许新增文档，检索默认排除。"""

    DELETED = "deleted"
    """逻辑删除：索引与 chunk 待异步清理。"""


class KnowledgeBaseVisibility(StrEnum):
    """知识库可见范围。

    这是**库级**的粗粒度控制；文档级的 ACL 与密级是更细的一层（A7 实现）。
    两层并存的原因：绝大多数场景只需库级控制（整个部门库），
    逐文档配 ACL 成本太高；但敏感文档确实需要单独收紧。
    """

    PRIVATE = "private"
    """私有：仅创建者可见（个人草稿库）。"""

    TENANT = "tenant"
    """租户内公开：所有成员可见。"""

    DEPARTMENT = "department"
    """部门内可见：需要 department_ids 配合（A7 的 ACL 层实现）。"""


class ConfidentialityLevel(StrEnum):
    """密级。数值越大越敏感。

    密级与 ACL 是**正交**的两道门：
    - ACL 回答"你能不能看这个库/文档"；
    - 密级回答"你这个人的密级够不够"。
    缺任何一道都会出事：只有 ACL 时，"有权限的人"可能是实习生；
    只有密级时，"够级的人"可能不在这个部门。
    """

    PUBLIC = "public"
    """公开：可对外。"""

    INTERNAL = "internal"
    """内部：默认级别。"""

    CONFIDENTIAL = "confidential"
    """机密：核心业务数据。"""

    SECRET = "secret"
    """绝密：高管/合规相关，访问必留审计。"""


class DocumentStatus(StrEnum):
    """文档处理状态机。

    完整流转：

        PENDING ──▶ PARSING ──▶ CHUNKING ──▶ EMBEDDING ──▶ INDEXING ──▶ READY
           │           │           │            │            │
           └───────────┴───────────┴────────────┴────────────┴──▶ FAILED
                                                                  │
                                                              (重试) ──▶ PENDING

    约束（A3-4 的任务表会用同样的状态集）：
    - 只有 READY 的文档能参与检索；
    - FAILED 的文档保留原始 blob，允许重试而不必重新上传；
    - 状态流转必须走乐观锁（`version`），否则并发重试会互相覆盖。
    """

    PENDING = "pending"
    """已上传，等待解析。"""

    PARSING = "parsing"
    """解析中（PDF → 文本 + 位置信息）。"""

    CHUNKING = "chunking"
    """切分中。"""

    EMBEDDING = "embedding"
    """向量化中。"""

    INDEXING = "indexing"
    """写入索引中。"""

    READY = "ready"
    """可检索。"""

    FAILED = "failed"
    """处理失败，错误详情见 error_code / error_message。"""


class DocumentVersionStatus(StrEnum):
    """文档版本状态。

    与 `DocumentStatus` 分开：文档整体状态取"当前版本"的状态，
    但历史版本有独立生命周期（可能已归档、已被新版本取代）。
    """

    PROCESSING = "processing"
    """处理中。"""

    READY = "ready"
    """可用。"""

    FAILED = "failed"
    """处理失败。"""

    SUPERSEDED = "superseded"
    """已被新版本取代，但保留供回滚与审计。"""


class SourceType(StrEnum):
    """文档来源类型。决定解析与去重策略的差异。"""

    UPLOAD = "upload"
    """用户上传的文件。"""

    URL = "url"
    """网页抓取（后续阶段）。"""

    CONNECTOR = "connector"
    """企业系统同步（Confluence / 飞书文档等，后续阶段）。"""

    API = "api"
    """通过 API 直接推送文本内容。"""


class DocumentFormat(StrEnum):
    """文档格式。决定用哪个 parser 实现。

    新增格式时必须同步：
    1. `rag_api.parsers` 下的解析器实现；
    2. 前端上传组件的 `accept` 白名单；
    3. 本文档的注释（说明该格式的已知坑）。
    """

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    IMAGE = "image"
    """图片走 OCR 分流。"""


class BlobStatus(StrEnum):
    """物理文件状态。

    单独设状态的原因：上传可能中断、校验可能失败、对象存储可能丢文件。
    文档版本引用一个 blob 时，必须确认它真的可用。
    """

    PENDING = "pending"
    """上传会话已创建，数据未完整。"""

    COMPLETE = "complete"
    """数据完整且已校验（hash 匹配）。"""

    MISSING = "missing"
    """对象存储里找不到（对账任务标记）。"""

    DELETED = "deleted"
    """已删除（可能仍被其它文档引用时先标记，等引用清零再物理删）。"""
