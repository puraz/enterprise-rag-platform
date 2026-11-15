/**
 * 本文件由 `make gen-contracts` 自动生成，请勿手改。
 *
 * 源头：packages/contracts/src/rag_contracts/**
 * 生成器：packages/contracts/scripts/gen_ts.py
 *
 * 改契约的正确姿势：
 *   1. 改 Pydantic 模型 / 枚举
 *   2. make gen-contracts
 *   3. 提交生成产物（前端 clone 后无需 Python 环境）
 */

// ============================ 枚举 ============================

/**
 * 全平台错误码。
 * 字符串值形如 `<域>.<含义>`，前缀即分类，便于前端一眼判断处理策略。
 */
export type ErrorCode =
  | "COMMON.VALIDATION_FAILED"
  | "COMMON.BAD_REQUEST"
  | "COMMON.NOT_FOUND"
  | "COMMON.METHOD_NOT_ALLOWED"
  | "COMMON.PAYLOAD_TOO_LARGE"
  | "COMMON.UNSUPPORTED_MEDIA_TYPE"
  | "COMMON.RATE_LIMITED"
  | "COMMON.IDEMPOTENCY_CONFLICT"
  | "COMMON.CONFLICT"
  | "COMMON.PRECONDITION_FAILED"
  | "AUTH.UNAUTHENTICATED"
  | "AUTH.TOKEN_EXPIRED"
  | "AUTH.TOKEN_INVALID"
  | "AUTH.TOKEN_REVOKED"
  | "AUTH.REFRESH_REUSED"
  | "AUTH.INVALID_CREDENTIALS"
  | "AUTH.ACCOUNT_DISABLED"
  | "AUTH.IDP_UNAVAILABLE"
  | "AUTHZ.FORBIDDEN"
  | "AUTHZ.TENANT_MISMATCH"
  | "AUTHZ.INSUFFICIENT_ROLE"
  | "AUTHZ.DOCUMENT_ACCESS_DENIED"
  | "AUTHZ.CLASSIFICATION_DENIED"
  | "DOC.NOT_FOUND"
  | "DOC.DUPLICATE"
  | "DOC.STATUS_CONFLICT"
  | "DOC.UNSUPPORTED_FORMAT"
  | "DOC.PARSE_FAILED"
  | "DOC.VERSION_NOT_FOUND"
  | "CHUNK.NOT_FOUND"
  | "KB.NOT_FOUND"
  | "KB.NAME_CONFLICT"
  | "CONV.NOT_FOUND"
  | "CONV.BUSY"
  | "UPLOAD.SESSION_NOT_FOUND"
  | "UPLOAD.SESSION_EXPIRED"
  | "UPLOAD.PART_MISMATCH"
  | "UPLOAD.INCOMPLETE"
  | "UPLOAD.HASH_MISMATCH"
  | "UPLOAD.EMPTY_FILE"
  | "UPLOAD.FILE_TOO_LARGE"
  | "UPLOAD.MIME_MISMATCH"
  | "TASK.NOT_FOUND"
  | "TASK.NOT_CANCELLABLE"
  | "INDEX.REBUILD_IN_PROGRESS"
  | "INDEX.VERSION_NOT_FOUND"
  | "QUOTA.EXCEEDED"
  | "QUOTA.BUDGET_EXHAUSTED"
  | "PROVIDER.UNAVAILABLE"
  | "PROVIDER.TIMEOUT"
  | "PROVIDER.RATE_LIMITED"
  | "PROVIDER.AUTH_FAILED"
  | "PROVIDER.CONTENT_FILTERED"
  | "PROVIDER.INVALID_RESPONSE"
  | "INTERNAL.UNEXPECTED"
  | "INTERNAL.CONFIG_ERROR"
  | "INTERNAL.DEPENDENCY_UNAVAILABLE"
  | "INTERNAL.NOT_IMPLEMENTED";

/**
 * 错误大类。决定"该谁负责"和"前端该怎么处理"。
 * 与 HTTP 状态码是两套体系：状态码给网关/监控看，category 给人和
 * 前端策略看。两者由异常类分别声明，不互相推导。
 */
export type ErrorCategory =
  | "validation"
  | "authentication"
  | "authorization"
  | "not_found"
  | "conflict"
  | "rate_limit"
  | "quota"
  | "dependency"
  | "internal";

/** 日志与告警级别。把"要不要半夜叫醒人"编码进错误定义。 */
export type Severity =
  | "info"
  | "warning"
  | "error"
  | "critical";

/** 租户状态。 */
export type TenantStatus =
  | "active"
  | "suspended"
  | "deleted";

/**
 * 用户全局状态。
 * 注意与 `MembershipStatus` 的区别：这个字段管"这个人还能不能用"，
 * `MembershipStatus` 管"这个人在某个租户里是否生效"。
 */
export type UserStatus =
  | "active"
  | "disabled"
  | "pending";

/**
 * 租户内角色，权限从高到低。
 * **这是租户内角色，不是平台角色**——平台侧看 `PlatformRole`，两者正交。
 */
export type TenantRole =
  | "owner"
  | "admin"
  | "knowledge_admin"
  | "member"
  | "viewer";

/**
 * 平台级角色（管理端）。
 * 与 `TenantRole` 正交：平台超管未必是任何租户的成员。
 * 两者分开是因为「跨租户治理」和「租户内协作」是两种权力来源，
 * 混成一个字段会导致越权判断写成一团 if-else。
 */
export type PlatformRole =
  | "none"
  | "super_admin"
  | "platform_operator"
  | "auditor";

/**
 * 成员关系状态。
 * 与 `UserStatus` 分开：用户可以是 active，但在这个租户里被移除。
 * 把两者合成一个字段会导致「禁用某人」和「从租户移除某人」无法区分。
 */
export type MembershipStatus =
  | "active"
  | "suspended"
  | "removed";

/** 知识库状态。 */
export type KnowledgeBaseStatus =
  | "active"
  | "archived"
  | "deleted";

/**
 * 知识库可见范围。
 * 这是**库级**的粗粒度控制；文档级的 ACL 与密级是更细的一层（A7 实现）。
 * 两层并存的原因：绝大多数场景只需库级控制（整个部门库），
 * 逐文档配 ACL 成本太高；但敏感文档确实需要单独收紧。
 */
export type KnowledgeBaseVisibility =
  | "private"
  | "tenant"
  | "department";

/**
 * 密级。数值越大越敏感。
 * 密级与 ACL 是**正交**的两道门：
 * - ACL 回答"你能不能看这个库/文档"；
 * - 密级回答"你这个人的密级够不够"。
 * 缺任何一道都会出事：只有 ACL 时，"有权限的人"可能是实习生；
 * 只有密级时，"够级的人"可能不在这个部门。
 */
export type ConfidentialityLevel =
  | "public"
  | "internal"
  | "confidential"
  | "secret";

/**
 * 文档处理状态机。
 * 完整流转：
 * PENDING ──▶ PARSING ──▶ CHUNKING ──▶ EMBEDDING ──▶ INDEXING ──▶ READY
 * │           │           │            │            │
 * └───────────┴───────────┴────────────┴────────────┴──▶ FAILED
 * │
 * (重试) ──▶ PENDING
 * 约束（A3-4 的任务表会用同样的状态集）：
 * - 只有 READY 的文档能参与检索；
 * - FAILED 的文档保留原始 blob，允许重试而不必重新上传；
 * - 状态流转必须走乐观锁（`version`），否则并发重试会互相覆盖。
 */
export type DocumentStatus =
  | "pending"
  | "parsing"
  | "chunking"
  | "embedding"
  | "indexing"
  | "ready"
  | "failed";

/**
 * 文档版本状态。
 * 与 `DocumentStatus` 分开：文档整体状态取"当前版本"的状态，
 * 但历史版本有独立生命周期（可能已归档、已被新版本取代）。
 */
export type DocumentVersionStatus =
  | "processing"
  | "ready"
  | "failed"
  | "superseded";

/**
 * 文档格式。决定用哪个 parser 实现。
 * 新增格式时必须同步：
 * 1. `rag_api.parsers` 下的解析器实现；
 * 2. 前端上传组件的 `accept` 白名单；
 * 3. 本文档的注释（说明该格式的已知坑）。
 */
export type DocumentFormat =
  | "pdf"
  | "docx"
  | "pptx"
  | "xlsx"
  | "csv"
  | "markdown"
  | "html"
  | "text"
  | "image";

/** 文档来源类型。决定解析与去重策略的差异。 */
export type SourceType =
  | "upload"
  | "url"
  | "connector"
  | "api";

/**
 * 物理文件状态。
 * 单独设状态的原因：上传可能中断、校验可能失败、对象存储可能丢文件。
 * 文档版本引用一个 blob 时，必须确认它真的可用。
 */
export type BlobStatus =
  | "pending"
  | "complete"
  | "missing"
  | "deleted";

/**
 * chunk 的内容类型。
 * 整块不切的类型（TABLE / CODE / FORMULA）在切分时会被保护，
 * 不参与递归合并 —— 表格被切断会让行列错位，代码被切断会失去语义。
 */
export type ChunkType =
  | "text"
  | "title"
  | "table"
  | "code"
  | "formula"
  | "image_caption"
  | "list";

/**
 * 切分策略标识。
 * 值必须与 `rag_api.chunkers` 注册表里的名字一致。
 * 新增策略时要同时更新这里与注册表——不一致会导致 worker 找不到实现，
 * 报错信息是"未知策略"，但实际是名字拼写不同。
 */
export type ChunkingStrategy =
  | "recursive"
  | "structured"
  | "semantic"
  | "sentence_window"
  | "parent_child"
  | "fixed";

/**
 * 向量索引类型。
 * HNSW vs IVFFlat 的取舍：
 * - HNSW：查询快、召回高、无需训练，但构建慢、内存占用大；
 * - IVFFlat：构建快、内存小，但需要数据量足够才能训练好，
 * 且召回受 probes 影响明显。
 * 本项目默认 HNSW：知识库规模通常是十万到百万级 chunk，
 * HNSW 的查询延迟优势更重要。
 */
export type IndexType =
  | "hnsw"
  | "ivfflat"
  | "none";

/**
 * 索引版本状态（A10 零停机重建用）。
 * 双版本 + alias 切换的完整生命周期：
 * BUILDING → READY → ACTIVE → RETIRED
 * │                 ▲
 * └──(切换前可回滚)──┘
 * 重建期间双写两个版本，切换瞬间只改 alias 指向。
 */
export type IndexVersionStatus =
  | "building"
  | "ready"
  | "active"
  | "retired"
  | "failed";

/**
 * 向量距离度量。
 * **必须与 embedding 模型的训练目标一致**：
 * - bge 系列、多数字段中文模型：用余弦（cosine）；
 * - OpenAI text-embedding-3：用余弦；
 * - 少数模型（如某些 e5 变体）训练时用内积。
 * 用错度量不会报错，只会让召回悄悄变差——这是最难发现的检索问题之一。
 */
export type DistanceMetric =
  | "cosine"
  | "l2"
  | "inner_product";

/** 检索模式（A11 用）。 */
export type RetrievalMode =
  | "vector"
  | "keyword"
  | "hybrid";

// =========================== 数据模型 ===========================

/** 统一错误响应体。所有非 2xx 响应都必须是这个形状。 */
export interface ErrorResponse {
  /** 业务错误码，前端按此分支处理 */
  code: ErrorCode;
  /** 面向用户的中文错误信息 */
  message: string;
  /** 请求追踪 ID，报障时提供此值 */
  request_id: string;
  /** 字段级错误明细 */
  fields?: FieldError[];
  /** 可公开的结构化补充信息 */
  details?: Record<string, string | number | boolean>;
}

/** 字段级校验错误。供前端把错误标到具体输入框上。 */
export interface FieldError {
  /** 错误字段位置路径 */
  location?: (string | number)[];
  /** 给用户看的中文提示 */
  message: string;
  /** 校验器错误类型标识 */
  type: string;
}
