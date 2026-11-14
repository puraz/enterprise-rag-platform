"""错误码契约：四端唯一真相来源。

为什么错误码要单独成文并进 contracts：
- 前端要按 `code` 分支（弹什么提示、要不要重试、跳不跳登录页），
  不能靠 HTTP 状态码区分——同一个 400 可能对应十几种业务含义。
- 前端 TS 类型由这里生成，新增码会自动出现在前端的联合类型里，
  漏处理会编译期报错，而不是上线后没人知道 409 该干嘛。

命名分两层：
- **成员名**必须是合法 Python 标识符 → 用下划线 `COMMON_VALIDATION_FAILED`
- **字符串值**是对外契约，带点号分段 → `"COMMON.VALIDATION_FAILED"`

前端/日志/审计里一律用**字符串值**，不用成员名。两者故意不同：
成员名受语言约束会变（换成别的语言生成器时），字符串值不会。

**码一旦发布就不能改字符串值**：前端已在上线版本里硬编码分支。
需要变更语义时新增一个码、把旧码标记 deprecated，而不是原地改。
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """全平台错误码。

    字符串值形如 `<域>.<含义>`，前缀即分类，便于前端一眼判断处理策略。
    """

    # ---------------- 通用（请求本身有问题，重试无用） ----------------
    COMMON_VALIDATION_FAILED = "COMMON.VALIDATION_FAILED"
    """入参校验失败。响应里带字段级错误列表。"""

    COMMON_BAD_REQUEST = "COMMON.BAD_REQUEST"
    """请求语义不合法，但不属于字段校验。"""

    COMMON_NOT_FOUND = "COMMON.NOT_FOUND"
    """资源不存在。注意：越权访问也返回这个，不泄漏资源是否存在。"""

    COMMON_METHOD_NOT_ALLOWED = "COMMON.METHOD_NOT_ALLOWED"

    COMMON_PAYLOAD_TOO_LARGE = "COMMON.PAYLOAD_TOO_LARGE"
    """请求体超限（上传分片 / JSON body 上限）。"""

    COMMON_UNSUPPORTED_MEDIA_TYPE = "COMMON.UNSUPPORTED_MEDIA_TYPE"

    COMMON_RATE_LIMITED = "COMMON.RATE_LIMITED"
    """被限流。响应头带 Retry-After，前端应据此退避。"""

    COMMON_IDEMPOTENCY_CONFLICT = "COMMON.IDEMPOTENCY_CONFLICT"
    """同一幂等键的请求体与首次不一致。"""

    COMMON_CONFLICT = "COMMON.CONFLICT"
    """状态冲突（乐观锁失败、重复创建等，调用方需重新读取再决策）。"""

    COMMON_PRECONDITION_FAILED = "COMMON.PRECONDITION_FAILED"
    """前置条件不满足（如依赖任务未完成）。"""

    # ---------------- 认证（身份不成立，前端应引导重新登录） ----------------
    AUTH_UNAUTHENTICATED = "AUTH.UNAUTHENTICATED"

    AUTH_TOKEN_EXPIRED = "AUTH.TOKEN_EXPIRED"
    """access token 过期 → 前端用 refresh 换新，不要直接踢到登录页。"""

    AUTH_TOKEN_INVALID = "AUTH.TOKEN_INVALID"
    """签名/格式/nonce 校验失败。与 EXPIRED 区分：前者重试无用。"""

    AUTH_TOKEN_REVOKED = "AUTH.TOKEN_REVOKED"
    """已被强制下线。必须重新走 SSO。"""

    AUTH_REFRESH_REUSED = "AUTH.REFRESH_REUSED"
    """refresh token 复用：判定为凭据泄漏，整条会话链作废。"""

    AUTH_INVALID_CREDENTIALS = "AUTH.INVALID_CREDENTIALS"

    AUTH_ACCOUNT_DISABLED = "AUTH.ACCOUNT_DISABLED"

    AUTH_IDP_UNAVAILABLE = "AUTH.IDP_UNAVAILABLE"
    """身份提供商不可用。可重试，应提示稍后再试而不是报错退出。"""

    # ---------------- 授权（身份成立但没有权限） ----------------
    AUTHZ_FORBIDDEN = "AUTHZ.FORBIDDEN"

    AUTHZ_TENANT_MISMATCH = "AUTHZ.TENANT_MISMATCH"
    """跨租户访问。审计必须留痕，这类事件是安全告警信号。"""

    AUTHZ_INSUFFICIENT_ROLE = "AUTHZ.INSUFFICIENT_ROLE"

    AUTHZ_DOCUMENT_ACCESS_DENIED = "AUTHZ.DOCUMENT_ACCESS_DENIED"
    """文档级 ACL 拒绝。是否对外暴露成 404 见 COMMON_NOT_FOUND 说明。"""

    AUTHZ_CLASSIFICATION_DENIED = "AUTHZ.CLASSIFICATION_DENIED"
    """密级不足。"""

    # ---------------- 资源与业务状态 ----------------
    DOC_NOT_FOUND = "DOC.NOT_FOUND"

    DOC_DUPLICATE = "DOC.DUPLICATE"
    """同租户下内容 hash 已存在（秒传命中也可能返回这个）。"""

    DOC_STATUS_CONFLICT = "DOC.STATUS_CONFLICT"
    """当前状态不允许该操作，例如对 READY 的文档再触发解析。"""

    DOC_UNSUPPORTED_FORMAT = "DOC.UNSUPPORTED_FORMAT"

    DOC_PARSE_FAILED = "DOC.PARSE_FAILED"

    DOC_VERSION_NOT_FOUND = "DOC.VERSION_NOT_FOUND"

    CHUNK_NOT_FOUND = "CHUNK.NOT_FOUND"

    KB_NOT_FOUND = "KB.NOT_FOUND"

    KB_NAME_CONFLICT = "KB.NAME_CONFLICT"

    CONV_NOT_FOUND = "CONV.NOT_FOUND"

    CONV_BUSY = "CONV.BUSY"
    """同一会话已有进行中的问答。并发提问必须明确报错而非错乱上下文。"""

    UPLOAD_SESSION_NOT_FOUND = "UPLOAD.SESSION_NOT_FOUND"

    UPLOAD_SESSION_EXPIRED = "UPLOAD.SESSION_EXPIRED"

    UPLOAD_PART_MISMATCH = "UPLOAD.PART_MISMATCH"
    """分片序号/大小/ETag 与预期不符。"""

    UPLOAD_INCOMPLETE = "UPLOAD.INCOMPLETE"

    UPLOAD_HASH_MISMATCH = "UPLOAD.HASH_MISMATCH"
    """合并后整体 hash 与声明不符，必须作废而不是入库。"""

    UPLOAD_EMPTY_FILE = "UPLOAD.EMPTY_FILE"

    UPLOAD_FILE_TOO_LARGE = "UPLOAD.FILE_TOO_LARGE"

    UPLOAD_MIME_MISMATCH = "UPLOAD.MIME_MISMATCH"
    """魔数与声明的 Content-Type 不一致。安全校验，不是格式不支持。"""

    TASK_NOT_FOUND = "TASK.NOT_FOUND"

    TASK_NOT_CANCELLABLE = "TASK.NOT_CANCELLABLE"

    INDEX_REBUILD_IN_PROGRESS = "INDEX.REBUILD_IN_PROGRESS"

    INDEX_VERSION_NOT_FOUND = "INDEX.VERSION_NOT_FOUND"

    # ---------------- 配额与成本 ----------------
    QUOTA_EXCEEDED = "QUOTA.EXCEEDED"
    """配额用尽。与 RATE_LIMITED 区分：前者需提额，后者稍等即可。"""

    QUOTA_BUDGET_EXHAUSTED = "QUOTA.BUDGET_EXHAUSTED"
    """成本预算耗尽。"""

    # ---------------- 外部供应商 ----------------
    PROVIDER_UNAVAILABLE = "PROVIDER.UNAVAILABLE"
    """供应商暂不可用。已走降级链，本次用降级结果或明确告知。"""

    PROVIDER_TIMEOUT = "PROVIDER.TIMEOUT"

    PROVIDER_RATE_LIMITED = "PROVIDER.RATE_LIMITED"

    PROVIDER_AUTH_FAILED = "PROVIDER.AUTH_FAILED"
    """我方配置的密钥失效——运维故障，不是用户错误。"""

    PROVIDER_CONTENT_FILTERED = "PROVIDER.CONTENT_FILTERED"

    PROVIDER_INVALID_RESPONSE = "PROVIDER.INVALID_RESPONSE"
    """返回结构不符合预期（协议漂移 / 中间件改写）。"""

    # ---------------- 内部 ----------------
    INTERNAL_UNEXPECTED = "INTERNAL.UNEXPECTED"
    """兜底。对外只回这个 + request_id，绝不回堆栈。"""

    INTERNAL_CONFIG_ERROR = "INTERNAL.CONFIG_ERROR"

    INTERNAL_DEPENDENCY_UNAVAILABLE = "INTERNAL.DEPENDENCY_UNAVAILABLE"
    """自身依赖（PG / Redis / 存储）不可用。"""

    INTERNAL_NOT_IMPLEMENTED = "INTERNAL.NOT_IMPLEMENTED"


class ErrorCategory(StrEnum):
    """错误大类。决定"该谁负责"和"前端该怎么处理"。

    与 HTTP 状态码是两套体系：状态码给网关/监控看，category 给人和
    前端策略看。两者由异常类分别声明，不互相推导。
    """

    VALIDATION = "validation"
    """调用方入参错误 → 前端改输入。"""

    AUTHENTICATION = "authentication"
    """身份不成立 → 引导登录 / 刷新。"""

    AUTHORIZATION = "authorization"
    """身份成立但无权 → 不重试，记审计。"""

    NOT_FOUND = "not_found"

    CONFLICT = "conflict"
    """状态冲突 → 重新读取当前状态再决策。"""

    RATE_LIMIT = "rate_limit"
    """被限流 → 按 Retry-After 退避重试。"""

    QUOTA = "quota"
    """配额 / 预算 → 需人工提额，重试无用。"""

    DEPENDENCY = "dependency"
    """外部依赖故障 → 可重试，走降级。"""

    INTERNAL = "internal"
    """我方 bug → 必须告警 + 留 request_id。"""


class Severity(StrEnum):
    """日志与告警级别。把"要不要半夜叫醒人"编码进错误定义。"""

    INFO = "info"
    """预期内的业务拒绝（404、配额不足）。只记日志。"""

    WARNING = "warning"
    """需要观察（限流、供应商降级、并发冲突重试）。进看板。"""

    ERROR = "error"
    """我方内部错误或依赖故障。进告警。"""

    CRITICAL = "critical"
    """数据安全 / 越权 / 一致性破坏。必须立即人工介入。"""
