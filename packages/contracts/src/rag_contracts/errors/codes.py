"""业务错误码体系：数字码 + 语义名双轨。

## 为什么是双轨

| 字段 | 类型 | 消费者 | 例子 |
|---|---|---|---|
| `code` | `int` | 前端 `switch` 分支、监控告警规则 | `10101` |
| `code_name` | `str` | 日志、审计、排障、文档 | `AUTH_TOKEN_EXPIRED` |
| HTTP 状态 | `int` | 网关、浏览器、前端拦截器 | `401` |

只有数字码的问题：日志里 `code=10101` 完全不可读，排障必须查表。
只有字符串的问题：前端要写长字符串比较，且无法用数值区间做规则
（比如"所有 1xxxx 都走登录重定向"）。

### 为什么不直接用 HTTP 状态码当前缀（40101 这种）

1. **419/420 这类非标状态码会与业务语义撞车**——我们实际用的是 401；
2. HTTP 状态码只有 60 来个，业务错误有上百个，前缀会立刻耗尽；
3. **状态码会变，业务码不该变**：同一个"token 过期"，网关可能希望
   返回 401 让浏览器处理，也可能在特定路由返回 403 防止重定向循环——
   业务码必须稳定，HTTP 码才可以按场景调整。

## 号段规划（每域留 10000 个）

| 号段 | 域 | 对应模块 |
|---|---|---|
| `0xxxx` | 通用/框架 | 校验、路由、限流 |
| `1xxxx` | 认证 AuthN | token、登录、IdP |
| `2xxxx` | 授权 AuthZ | 角色、ACL、密级、跨租户 |
| `3xxxx` | 知识库与文档 | KB、document、version、blob |
| `4xxxx` | 分块/索引/检索 | chunk、index、retrieval |
| `5xxxx` | 上传与存储 | upload session、multipart、存储后端 |
| `6xxxx` | 任务与队列 | task、outbox |
| `7xxxx` | 配额与计费 | quota、budget、成本 |
| `8xxxx` | 外部供应商 | LLM、embedding、reranker、IdP 服务 |
| `9xxxx` | 内部系统 | 未预期错误、配置、依赖 |

**新增域从号段表申请，不重排已有码。** 10000 个/域足够用很久。

## 编码规则

`code` 的构成：`<域首位><3位子域><2位序号>`

例：`10101`
- `1` → 认证域
- `01` → token 相关子域
- `01` → 该子域第 1 个错误

子域划分让"token 相关的所有错误"集中在 `101xx`，
前端可以写 `code >= 10100 && code < 10200` 做批量判断。

## 一旦发布不能改

前端会硬编码 `if (code === 10101)`。需要变更语义时：
- **新增**一个码，把旧码标记 deprecated；
- 绝不改变已有码的含义，也不复用已废弃的码。

`tests/test_error_codes.py` 会检查号段合法性、唯一性、以及与
`code_name` 的映射一致性。
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class ErrorCode(IntEnum):
    """全平台业务错误码。

    成员名是语义名（对应 `code_name`），值是数字码。
    用 `IntEnum` 而不是 `StrEnum`：序列化时自然是数字，
    且可以做数值比较（前端按号段判断）。
    """

    # ================= 0xxxx 通用 =================
    COMMON_VALIDATION_FAILED = 1
    """请求参数校验失败。响应里带字段级明细。"""

    COMMON_BAD_REQUEST = 2
    """请求语义不合法，但不属于字段校验。"""

    COMMON_NOT_FOUND = 3
    """资源不存在。**越权访问也返回此码**，不泄漏资源是否存在。"""

    COMMON_METHOD_NOT_ALLOWED = 4
    COMMON_PAYLOAD_TOO_LARGE = 5
    COMMON_UNSUPPORTED_MEDIA_TYPE = 6
    COMMON_RATE_LIMITED = 7
    """被限流。响应头带 Retry-After，前端据此退避。"""

    COMMON_IDEMPOTENCY_CONFLICT = 8
    """同一幂等键的请求体与首次不一致。"""

    COMMON_CONFLICT = 9
    """状态冲突（乐观锁失败、重复创建等，调用方需重新读取再决策）。"""

    COMMON_PRECONDITION_FAILED = 10

    # ================= 1xxxx 认证 =================
    AUTH_UNAUTHENTICATED = 10001
    AUTH_TOKEN_EXPIRED = 10101
    """access token 过期 → 前端用 refresh 换新，不要直接踢到登录页。"""

    AUTH_TOKEN_INVALID = 10102
    """签名/格式/issuer 校验失败。与 EXPIRED 区分：前者重试无用。"""

    AUTH_TOKEN_REVOKED = 10103
    """已被强制下线。必须重新走 SSO。"""

    AUTH_REFRESH_REUSED = 10104
    """refresh token 复用：判定为凭据泄漏，整条会话链作废。"""

    AUTH_TOKEN_MISSING = 10105
    """缺少 Authorization 头。"""

    AUTH_INVALID_CREDENTIALS = 10201
    AUTH_ACCOUNT_DISABLED = 10202
    AUTH_IDP_UNAVAILABLE = 10301
    """身份提供商不可用。可重试，应提示稍后再试而不是报错退出。"""

    AUTH_DEV_DISABLED = 10302
    """dev 登录端点未启用（生产环境默认）。"""

    # ================= 2xxxx 授权 =================
    AUTHZ_FORBIDDEN = 20001
    AUTHZ_TENANT_MISMATCH = 20101
    """跨租户访问。审计必须留痕，这类事件是安全告警信号。"""

    AUTHZ_INSUFFICIENT_ROLE = 20201
    AUTHZ_DOCUMENT_ACCESS_DENIED = 20301
    AUTHZ_CLASSIFICATION_DENIED = 20302
    """密级不足。"""

    # ================= 3xxxx 知识库与文档 =================
    KB_NOT_FOUND = 30001
    KB_NAME_CONFLICT = 30002
    KB_ARCHIVED = 30003
    """知识库已归档，不允许新增文档。"""

    DOC_NOT_FOUND = 31001
    DOC_DUPLICATE = 31002
    """同租户下内容 hash 已存在（秒传命中时也可能返回这个）。"""

    DOC_STATUS_CONFLICT = 31003
    """当前状态不允许该操作，例如对 READY 的文档再触发解析。"""

    DOC_UNSUPPORTED_FORMAT = 31004
    DOC_PARSE_FAILED = 31005
    DOC_VERSION_NOT_FOUND = 31101
    DOC_VERSION_CONFLICT = 31102
    """版本号冲突（并发创建同版本）。"""

    BLOB_NOT_FOUND = 32001
    BLOB_INCOMPLETE = 32002
    """物理文件未完整（status != complete）。"""

    BLOB_MISSING = 32003
    """对象存储里找不到（对账发现）。"""

    # ================= 4xxxx 分块/索引/检索 =================
    CHUNK_NOT_FOUND = 40001
    CHUNK_EMPTY = 40002
    """切分结果为空（文档无有效文本）。"""

    INDEX_REBUILD_IN_PROGRESS = 41001
    INDEX_VERSION_NOT_FOUND = 41002
    INDEX_NOT_READY = 41003
    """索引版本尚未就绪（building 状态）。"""

    RETRIEVAL_NO_EVIDENCE = 42001
    """检索无依据 → 应明确拒答而不是编造。"""

    RETRIEVAL_QUERY_TOO_LONG = 42002

    # ================= 5xxxx 上传与存储 =================
    UPLOAD_SESSION_NOT_FOUND = 50001
    UPLOAD_SESSION_EXPIRED = 50002
    UPLOAD_PART_MISMATCH = 50101
    """分片序号/大小/ETag 与预期不符。"""

    UPLOAD_INCOMPLETE = 50102
    UPLOAD_HASH_MISMATCH = 50103
    """合并后整体 hash 与声明不符，必须作废而不是入库。"""

    UPLOAD_EMPTY_FILE = 50201
    UPLOAD_FILE_TOO_LARGE = 50202
    UPLOAD_MIME_MISMATCH = 50203
    """魔数与声明的 Content-Type 不一致。安全校验，不是格式不支持。"""

    UPLOAD_PATH_TRAVERSAL = 50204
    """文件名含路径穿越字符。"""

    STORAGE_UNAVAILABLE = 50301
    STORAGE_OBJECT_MISSING = 50302

    # ================= 6xxxx 任务与队列 =================
    TASK_NOT_FOUND = 60001
    TASK_NOT_CANCELLABLE = 60002
    TASK_ALREADY_RUNNING = 60003
    TASK_DEAD = 60101
    """任务已进死信，需人工介入。"""

    OUTBOX_BACKLOG = 60201
    """outbox 积压超阈值（告警用）。"""

    # ================= 7xxxx 配额与计费 =================
    QUOTA_EXCEEDED = 70001
    """配额用尽。与 RATE_LIMITED 区分：前者需提额，后者稍等即可。"""

    QUOTA_BUDGET_EXHAUSTED = 70002
    """成本预算耗尽。"""

    QUOTA_TENANT_CONCURRENCY_EXCEEDED = 70003

    # ================= 8xxxx 外部供应商 =================
    PROVIDER_UNAVAILABLE = 80001
    """供应商暂不可用。已走降级链，本次用降级结果或明确告知。"""

    PROVIDER_TIMEOUT = 80101
    PROVIDER_RATE_LIMITED = 80102
    PROVIDER_AUTH_FAILED = 80201
    """我方配置的密钥失效——运维故障，不是用户错误。"""

    PROVIDER_CONTENT_FILTERED = 80301
    PROVIDER_INVALID_RESPONSE = 80302
    """返回结构不符合预期（协议漂移 / 中间件改写）。"""

    PROVIDER_CONTEXT_TOO_LONG = 80303
    """输入超出模型上下文窗口。"""

    # ================= 9xxxx 内部 =================
    INTERNAL_UNEXPECTED = 90001
    """兜底。对外只回这个 + request_id，绝不回堆栈。"""

    INTERNAL_CONFIG_ERROR = 90101
    INTERNAL_DEPENDENCY_UNAVAILABLE = 90201
    """自身依赖（PG / Redis / 存储）不可用。"""

    INTERNAL_NOT_IMPLEMENTED = 90301
    INTERNAL_RATE_LIMITER_FAILED = 90302
    """限流器自身故障（Redis 挂）。按 fail-open 还是 fail-close 由策略决定。"""

    @property
    def domain(self) -> int:
        """域号（code // 10000）。前端可用它做粗粒度分类。"""
        return int(self.value) // 10_000

    @property
    def http_status(self) -> int:
        """默认 HTTP 状态码。

        与 `AppError.category` 的推导保持一致，但放在这里让"码 → 状态"
        的映射可以独立查询（例如给前端文档生成对照表）。

        **个别场景可以覆盖**：`AppError.status_override` 优先。
        """
        from rag_core.errors import default_status_for

        return default_status_for(self)


# 语义名 → 数字码的便捷映射（日志与审计用）
_CODE_NAMES: dict[int, str] = {member.value: member.name for member in ErrorCode}


def code_name(code: int) -> str:
    """取数字码的语义名。未知码返回 `UNKNOWN_<code>`。

    用于日志与审计：`code=10101` 不可读，`AUTH_TOKEN_EXPIRED` 一眼就懂。
    """
    return _CODE_NAMES.get(code, f"UNKNOWN_{code}")


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
