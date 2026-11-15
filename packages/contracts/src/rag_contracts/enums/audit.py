"""审计与成本域枚举。

对应模型：`rag_db.models.audit` / `model_call`。
"""

from __future__ import annotations

from enum import StrEnum


class AuditAction(StrEnum):
    """审计动作类型。

    **命名规则**：`<资源>.<动词>`，全小写下划线连接资源名。
    值是对外契约的一部分（管理端按此筛选），一旦发布不能改。
    """

    # ---- 认证 ----
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token_refresh"
    AUTH_LOGIN_FAILED = "auth.login_failed"

    # ---- 知识库 ----
    KB_CREATE = "knowledge_base.create"
    KB_UPDATE = "knowledge_base.update"
    KB_DELETE = "knowledge_base.delete"

    # ---- 文档 ----
    DOC_UPLOAD = "document.upload"
    DOC_READ = "document.read"
    DOC_DOWNLOAD = "document.download"
    DOC_DELETE = "document.delete"
    DOC_VERSION_ROLLBACK = "document.version_rollback"

    # ---- 检索与问答 ----
    RETRIEVAL_QUERY = "retrieval.query"
    CHAT_COMPLETION = "chat.completion"

    # ---- 安全事件（这类必须独立事务写，见 AuditRepository 说明）----
    SECURITY_ACCESS_DENIED = "security.access_denied"
    SECURITY_TENANT_MISMATCH = "security.tenant_mismatch"
    SECURITY_RATE_LIMITED = "security.rate_limited"
    SECURITY_TOKEN_REUSED = "security.token_reused"
    SECURITY_PROMPT_INJECTION = "security.prompt_injection"

    # ---- 管理面 ----
    ADMIN_INDEX_REBUILD = "admin.index_rebuild"
    ADMIN_PERMISSION_CHANGE = "admin.permission_change"
    ADMIN_PROVIDER_CHANGE = "admin.provider_change"
    ADMIN_CONFIG_CHANGE = "admin.config_change"
    ADMIN_AUDIT_QUERY = "admin.audit_query"
    """查询审计日志。**查询本身要留痕**：它是敏感操作，
    能暴露"谁在什么时候看了什么文档"。"""


class AuditOutcome(StrEnum):
    """操作结果。"""

    SUCCESS = "success"

    DENIED = "denied"
    """权限不足被拒。"""

    FAILED = "failed"
    """操作失败（业务错误或系统错误）。"""


class AuditTargetType(StrEnum):
    """被操作对象的类型。用于按对象查审计（"这个文档被谁看过"）。"""

    TENANT = "tenant"
    USER = "user"
    KNOWLEDGE_BASE = "knowledge_base"
    DOCUMENT = "document"
    DOCUMENT_VERSION = "document_version"
    CONVERSATION = "conversation"
    PROVIDER = "provider"
    CONFIG = "config"


class ModelTaskType(StrEnum):
    """模型调用的任务类型。

    成本按任务类型拆分是必需的：`docs/PROMPT.md` 要求"按租户-功能-模型
    三维成本拆解"。没有这个字段，就无法回答"钱主要花在检索改写还是回答生成上"。
    """

    CHAT = "chat"
    """答案生成（最贵，输出 token 多）。"""

    QUERY_REWRITE = "query_rewrite"
    """查询改写（便宜小模型即可）。"""

    MULTI_QUERY = "multi_query"
    """多查询生成。"""

    HYDE = "hyde"
    """假设性文档生成。"""

    CONTEXT_COMPRESS = "context_compress"
    """上下文压缩。"""

    EMBEDDING = "embedding"
    """向量化。"""

    RERANK = "rerank"
    """重排。"""

    EVALUATION = "evaluation"
    """评估打分（judge）。"""


class ModelCallStatus(StrEnum):
    """模型调用结果。"""

    SUCCESS = "success"

    FAILED = "failed"
    """调用失败（此时 token 与成本通常为 0 或未知）。"""

    TIMEOUT = "timeout"

    RATE_LIMITED = "rate_limited"

    CONTENT_FILTERED = "content_filtered"
    """被供应商内容审核拦截。仍然会计费，所以要记录。"""
