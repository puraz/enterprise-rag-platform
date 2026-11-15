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
 * 全平台业务错误码。
 * 成员名是语义名（对应 `code_name`），值是数字码。
 * 用 `IntEnum` 而不是 `StrEnum`：序列化时自然是数字，
 * 且可以做数值比较（前端按号段判断）。
 */
export type ErrorCode =
  | 1
  | 2
  | 3
  | 4
  | 5
  | 6
  | 7
  | 8
  | 9
  | 10
  | 10001
  | 10101
  | 10102
  | 10103
  | 10104
  | 10105
  | 10201
  | 10202
  | 10301
  | 10302
  | 20001
  | 20101
  | 20201
  | 20301
  | 20302
  | 30001
  | 30002
  | 30003
  | 31001
  | 31002
  | 31003
  | 31004
  | 31005
  | 31101
  | 31102
  | 32001
  | 32002
  | 32003
  | 40001
  | 40002
  | 41001
  | 41002
  | 41003
  | 42001
  | 42002
  | 50001
  | 50002
  | 50101
  | 50102
  | 50103
  | 50201
  | 50202
  | 50203
  | 50204
  | 50301
  | 50302
  | 60001
  | 60002
  | 60003
  | 60101
  | 60201
  | 70001
  | 70002
  | 70003
  | 80001
  | 80101
  | 80102
  | 80201
  | 80301
  | 80302
  | 80303
  | 90001
  | 90101
  | 90201
  | 90301
  | 90302;

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
 * 1. `rag_adapters.parsers` 下的解析器实现；
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

/**
 * 分片上传会话状态机。
 * INITIATED ──▶ UPLOADING ──▶ COMPLETING ──▶ COMPLETED
 * │             │              │
 * │             │              └──▶ FAILED（合并/校验失败）
 * │             └──▶ EXPIRED（超时未完成）
 * └──▶ ABORTED（用户取消）
 * 关键约束：
 * - COMPLETED 之后不能再收分片（幂等返回成功）；
 * - EXPIRED 与 ABORTED 的分片由清理任务回收；
 * - COMPLETING 是"合并中"的中间态，防止两个请求同时 complete。
 */
export type UploadSessionStatus =
  | "initiated"
  | "uploading"
  | "completing"
  | "completed"
  | "failed"
  | "expired"
  | "aborted";

/**
 * 异步任务类型。
 * 值必须与 worker 侧的任务注册表一致——不一致时报错是
 * "未知任务类型"，但实际是名字拼写不同，很难定位。
 */
export type TaskType =
  | "parse_document"
  | "chunk_document"
  | "embed_document"
  | "index_document"
  | "rebuild_index"
  | "cleanup_upload_sessions"
  | "cleanup_orphan_blobs"
  | "dispatch_outbox";

/**
 * 任务执行状态。
 * 与 `DocumentStatus` 分开：任务表记录"某次执行的进展"，
 * 文档表记录"业务上的最终状态"。一次文档处理可能产生多次任务执行
 * （每次重试一条），管理端需要看到历史。
 */
export type TaskStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "dead"
  | "cancelled";

/**
 * outbox 消息状态。
 * 至少一次送达：只要消息没有标记为 dispatched，就会一直被轮询重发。
 * 消费者必须幂等——这是"至少一次"语义的必然要求。
 */
export type OutboxStatus =
  | "pending"
  | "dispatched"
  | "failed";

/**
 * 审计动作类型。
 * **命名规则**：`<资源>.<动词>`，全小写下划线连接资源名。
 * 值是对外契约的一部分（管理端按此筛选），一旦发布不能改。
 */
export type AuditAction =
  | "auth.login"
  | "auth.logout"
  | "auth.token_refresh"
  | "auth.login_failed"
  | "knowledge_base.create"
  | "knowledge_base.update"
  | "knowledge_base.delete"
  | "document.upload"
  | "document.read"
  | "document.download"
  | "document.delete"
  | "document.version_rollback"
  | "retrieval.query"
  | "chat.completion"
  | "security.access_denied"
  | "security.tenant_mismatch"
  | "security.rate_limited"
  | "security.token_reused"
  | "security.prompt_injection"
  | "admin.index_rebuild"
  | "admin.permission_change"
  | "admin.provider_change"
  | "admin.config_change"
  | "admin.audit_query";

/** 操作结果。 */
export type AuditOutcome =
  | "success"
  | "denied"
  | "failed";

/** 被操作对象的类型。用于按对象查审计（"这个文档被谁看过"）。 */
export type AuditTargetType =
  | "tenant"
  | "user"
  | "knowledge_base"
  | "document"
  | "document_version"
  | "conversation"
  | "provider"
  | "config";

/**
 * 模型调用的任务类型。
 * 成本按任务类型拆分是必需的：`docs/PROMPT.md` 要求"按租户-功能-模型
 * 三维成本拆解"。没有这个字段，就无法回答"钱主要花在检索改写还是回答生成上"。
 */
export type ModelTaskType =
  | "chat"
  | "query_rewrite"
  | "multi_query"
  | "hyde"
  | "context_compress"
  | "embedding"
  | "rerank"
  | "evaluation";

/** 模型调用结果。 */
export type ModelCallStatus =
  | "success"
  | "failed"
  | "timeout"
  | "rate_limited"
  | "content_filtered";

/** 会话状态。 */
export type ConversationStatus =
  | "active"
  | "archived";

/**
 * 消息角色。
 * 与 `LLMRole`（provider 层）分开：这里是**业务角色**，
 * 将来可能要加 `tool`（工具返回）等业务概念，而 provider 层
 * 只认识 system/user/assistant 三种。
 */
export type MessageRole =
  | "user"
  | "assistant";

/**
 * 消息状态。
 * ## 为什么助手消息需要 5 种状态
 * 流式回答不是原子的：可能生成到一半断连、可能被自省判为无依据、
 * 可能供应商报错。用 `success`/`failed` 两态无法表达
 * "用户看到了部分内容但没写完"——而那恰恰是最需要记录的情况
 * （前端要显示"回答中断"，且这条消息不该进评估集）。
 */
export type MessageStatus =
  | "pending"
  | "streaming"
  | "completed"
  | "interrupted"
  | "failed";

/**
 * 拒答原因。
 * `docs/PROMPT.md` 要求"无依据拒答"。拒答必须记录原因，
 * 否则无法回答"为什么这个月拒答率涨了 3 倍"。
 */
export type RefusalReason =
  | "no_materials"
  | "insufficient_evidence"
  | "prompt_injection_suspected";

/** 用户反馈。 */
export type FeedbackRating =
  | "like"
  | "dislike";

/**
 * 点踩原因（可选，用于构建改进数据集）。
 * 固定枚举而不是自由文本：自由文本无法聚合统计，
 * 而"哪类问题最常被踩"是优化 prompt 的关键输入。
 */
export type FeedbackReason =
  | "not_relevant"
  | "wrong_fact"
  | "incomplete"
  | "no_citation"
  | "too_verbose"
  | "other";

// =========================== 数据模型 ===========================

/** 统一错误响应体。所有非 2xx 响应都必须是这个形状。 */
export interface ErrorResponse {
  /** 业务错误码（数字）。按域号段编码：0xxxx 通用、1xxxx 认证、2xxxx 授权、3xxxx 文档、4xxxx 检索、5xxxx 上传、6xxxx 任务、7xxxx 配额、8xxxx 供应商、9xxxx 内部。前端按此分支处理，**一旦发布不可变更语义** */
  code: number;
  /** 错误码的语义名，便于日志检索与排障（如 AUTH_TOKEN_EXPIRED） */
  code_name: string;
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

/** 开发登录请求（仅非生产可用）。 */
export interface DevLoginRequest {
  /** dev 凭证，格式 `dev:<subject>` 或 `dev:<issuer>:<subject>` */
  credential: string;
}

/** 登录成功的响应。 */
export interface LoginResponse {
  /** 本系统签发的 access token */
  access_token: string;
  /** 固定为 Bearer */
  token_type?: string;
  /** token 过期时间（Unix 秒），前端据此提前刷新 */
  expires_at: number;
  /** 本地用户 ID */
  user_id: string;
  /** 当前激活的租户 ID */
  tenant_id: string;
  email: string;
  display_name: string;
  /** 是否本次登录新建的用户（首登） */
  is_new_user: boolean;
}

/** `GET /me` 的响应。 */
export interface CurrentUserResponse {
  user_id: string;
  tenant_id: string;
  email: string;
  display_name: string;
  status: string;
  platform_role: string;
  /** 当前租户内的角色（owner/admin/…），非成员时为 null */
  tenant_role?: string | null;
}

/** 健康检查响应。 */
export interface HealthResponse {
  /** ok / degraded / error */
  status: string;
  version: string;
  environment: string;
  /** 各依赖的检查结果（ok / 错误摘要） */
  checks?: Record<string, string>;
}

/** 分页元信息。 */
export interface PageMeta {
  /** 每页条数（服务端上限 100） */
  limit: number;
  /** 跳过的条数 */
  offset: number;
  /** 总条数。仅在请求带 `include_total=true` 时返回，否则为 null —— 大表上 COUNT(*) 很慢，不应默认执行 */
  total?: number | null;
  /** 是否还有下一页。无限滚动只需看这个字段 */
  has_more: boolean;
}

/**
 * 分页请求参数（可作为 query 依赖使用）。
 * 路由层用法：
 * async def list_documents(page: PageRequest = Depends(page_query)): ...
 * 校验规则放在这里而不是每个路由各写一遍——否则不同接口的
 * 上限会不一致（A 接口 50、B 接口 1000），前端无法统一处理。
 */
export interface PageRequest {
  /** 每页条数，上限 100 */
  limit?: number;
  /** 跳过的条数 */
  offset?: number;
  /** 是否返回总条数（会触发额外 COUNT 查询） */
  include_total?: boolean;
}

/**
 * 异步任务已接受的响应（对应 HTTP 202）。
 * 用于"提交后立即返回、后台处理"的接口：文档解析、索引重建、批量操作。
 * **必须带 `task_id`**：前端靠它轮询 `GET /api/v1/tasks/{id}` 拿进度。
 * 没有 task_id 的 202 等于告诉前端"等着吧"，但它不知道等什么。
 */
export interface TaskAccepted {
  /** 任务 ID，用于轮询进度 */
  task_id: string;
  /** 任务初始状态 */
  status?: string;
  /** 建议的轮询地址（绝对路径），前端可直接 GET */
  poll_url?: string | null;
}

/**
 * 创建成功的响应基类（对应 HTTP 201）。
 * 子类声明具体资源字段；本类只约束"必须能被定位"。
 * 用法：
 * class DocumentCreated(CreatedResource):
 * title: str
 * format: str
 */
export interface CreatedResource {
  /** 新资源的 ID */
  id: string;
}

/** 初始化上传请求。 */
export interface InitiateUploadRequest {
  /** 原始文件名（服务端会净化） */
  filename: string;
  /** 文件总字节数 */
  total_size: number;
  /** MIME 类型（仅供参考，服务端以魔数为准） */
  content_type: string;
  /** 整体 SHA-256（小写十六进制）。提供时启用秒传：同租户内已有相同内容的文件则直接返回 blob_id，无需上传 */
  content_hash?: string | null;
  /** 目标知识库（可稍后在 complete 时指定） */
  knowledge_base_id?: string | null;
}

/**
 * 初始化上传响应。
 * 两种形态（用 `deduplicated` 区分）：
 * - **秒传**：`deduplicated=true`、`blob_id` 有值、`upload_id` 为 null；
 * - **需上传**：`deduplicated=false`、`upload_id` 有值、`blob_id` 为 null。
 */
export interface InitiateUploadResponse {
  /** 是否秒传命中（无需上传） */
  deduplicated: boolean;
  /** 存储层上传会话 ID */
  upload_id?: string | null;
  /** 秒传命中的 blob ID */
  blob_id?: string | null;
  /** 建议的分片大小（字节） */
  part_size: number;
  /** 总分片数 */
  total_parts: number;
  /** 已上传的分片号（续传场景：客户端只需补传缺失的） */
  uploaded_parts?: number[];
}

/** 分片上传响应。 */
export interface UploadPartResponse {
  /** 分片编号（从 1 开始） */
  part_number: number;
  /** 分片 ETag，complete 时需带回 */
  etag: string;
  /** 分片字节数 */
  size_bytes: number;
}

/** 上传会话状态（断点续传查询）。 */
export interface UploadSessionStatusResponse {
  /** 上传会话 ID */
  upload_id: string;
  /** 会话状态：initiated/uploading/completing/completed/... */
  status: string;
  total_parts: number;
  part_size: number;
  /** 已上传的分片（**以对象存储为准**，不是数据库计数） */
  uploaded_parts: UploadPartResponse[];
  /** 尚未上传的分片号，客户端只需补这些 */
  missing_parts: number[];
}

/** 完成上传请求。 */
export interface CompleteUploadRequest {
  /** 整体 SHA-256。提供时服务端会重新计算已合并文件的 hash 并比对；不匹配则作废上传。**强烈建议提供**——分片级 ETag 无法证明整体正确 */
  content_hash?: string | null;
  /** 目标知识库（覆盖 initiate 时的值） */
  knowledge_base_id?: string | null;
}

/** 完成上传响应。 */
export interface CompleteUploadResponse {
  /** 落库的物理文件 ID */
  blob_id: string;
  /** 文件大小 */
  size_bytes: number;
  /** 服务端计算的内容 SHA-256 */
  content_hash: string;
  /** 服务端确认的 MIME 类型 */
  content_type: string;
}

/** 一条审计记录。 */
export interface AuditLogItem {
  id: string;
  /** 链内序号（同一租户内严格递增） */
  seq: number;
  /** 动作类型，见 AuditAction */
  action: string;
  /** 结果：success/denied/failed */
  outcome: string;
  /** 操作者 ID（用户删除后为 null，但记录保留） */
  actor_user_id?: string | null;
  /** 操作者邮箱快照（用户改名后仍可追溯） */
  actor_email?: string | null;
  /** 目标对象类型 */
  target_type?: string | null;
  /** 目标对象 ID */
  target_id?: string | null;
  /** 目标名称快照 */
  target_name?: string | null;
  /** 关联的请求 ID，可回查完整链路日志 */
  request_id?: string | null;
  /** 客户端 IP */
  ip_address?: string | null;
  /** 失败/拒绝时的错误码 */
  error_code?: string | null;
  /** 结构化详情（服务端已脱敏 PII，不含文档正文） */
  details?: Record<string, unknown>;
  /** 事件时间（UTC） */
  created_at: string;
}

/**
 * 审计查询响应。
 * 扁平结构（与 A4 的分页契约一致）：`{items, total, limit, offset, has_more}`。
 */
export interface AuditQueryResponse {
  items: AuditLogItem[];
  /** 总条数。**默认 null**：审计表很大，COUNT(*) 需要全表扫描。管理端的页码器改为「加载更多」即可 */
  total?: number | null;
  limit: number;
  offset: number;
  /** 是否还有下一页 */
  has_more: boolean;
}

/** 一条引用：答案里的某个结论来自原文的哪个位置。 */
export interface Citation {
  /** 引用编号（从 1 开始）。答案正文里的 `[1]` 对应这个字段。 */
  index: number;
  /** 来源文档 ID */
  document_id: string;
  /** 文档标题（前端展示用，避免再查一次库） */
  document_title: string;
  /** 来源 chunk ID */
  chunk_id: string;
  /** chunk 在文档版本内的序号。引用可能跨多个 chunk 时按最小序号记。 */
  chunk_index: number;
  /** 页码（**从 1 开始**，与用户看到的页码一致）。非分页格式为 null。 */
  page_no?: number | null;
  /** 在文档纯文本中的起始字符偏移（含） */
  char_start?: number | null;
  /** 在文档纯文本中的结束字符偏移（不含） */
  char_end?: number | null;
  /** 版面坐标 {x0,y0,x1,y1}，单位 PDF point。只有 PDF 类画布式文档有；前端据此在原图上框选。 */
  bbox?: Record<string, number> | null;
  /** 章节路径（如 ['第三章', '3.2 报销流程']），用于面包屑展示 */
  section_path?: string[];
  /** 引用处的原文片段（截断到 300 字符）。坐标缺失或不准时，前端可据此在全文里搜索定位；也是人工审计「引用是否错配」的锚点。 */
  quote: string;
  /** 相关性分数，**余弦距离语义：0 最相关，2 最不相关**（越小越好）。不在此处转成相似度——两者语义相反，转换点越多越容易把排序搞反。 */
  relevance_score?: number | null;
}

/**
 * 带引用的答案。
 * 这是**问答接口的成功响应体**（`POST /api/v1/chat` 之类）。
 * 为什么 `citations` 可能是空列表：
 * - 模型拒答（"根据现有资料无法回答"）；
 * - 答案不需要引用（如"你好"这类寒暄）。
 * 前端不能假设 `citations` 非空。**拒答时 `refused=True`**，
 * 让前端能区分"拒答"与"回答了但没有引用"——这两者的 UI 处理完全不同。
 */
export interface AnswerWithCitations {
  /** 答案正文。其中的 `[n]` 对应 citations 里的 index。 */
  answer: string;
  /** 是否为拒答（材料不足，没有编造）。**与「citations 为空」不等价**：答案可以没有引用但不是拒答。 */
  refused?: boolean;
  /** 引用列表，按 index 升序 */
  citations?: Citation[];
  /** 是否走了降级路径（如 reranker 挂→只用 RRF、主模型挂→备用模型）。降级时前端应给用户明确提示，不能悄悄降级。 */
  degraded?: boolean;
}

/**
 * 发起一次问答。
 * ## 为什么 `conversation_id` 可选
 * - 空：新建会话（首轮问答）；
 * - 非空：在该会话内追问（服务端取历史做独立化改写）。
 * 服务端会校验该会话属于当前用户——**跨用户访问返回 404**
 * （不泄漏"这个 ID 是否存在"）。
 */
export interface ChatRequest {
  /** 用户问题。上限 4000 字符——超过这个长度的问题本身不合法（应拆成多次提问） */
  query: string;
  /** 会话 ID。空则新建会话 */
  conversation_id?: string | null;
  /** 限定的知识库。空表示租户内全部可见库。上限 20：太多库会让检索范围过宽、召回质量下降 */
  knowledge_base_ids?: string[];
}

/** 会话列表项。 */
export interface ConversationItem {
  id: string;
  title: string;
  status: ConversationStatus;
  knowledge_base_id?: string | null;
  message_count: number;
  last_message_at?: string | null;
  created_at: string;
  updated_at: string;
  /** 本会话累计成本（展示用） */
  total_cost_usd: number;
}

/** 消息项。 */
export interface MessageItem {
  id: string;
  conversation_id: string;
  sequence: number;
  role: MessageRole;
  content: string;
  status: MessageStatus;
  refused?: boolean;
  refusal_reason?: RefusalReason | null;
  /** 引用列表（只含本轮检索来源） */
  citations?: Citation[];
  created_at: string;
  input_tokens?: number;
  output_tokens?: number;
  cost_usd?: number;
  latency_ms?: number | null;
}

/** SSE `done` 事件的 payload。 */
export interface ChatDonePayload {
  message_id: string;
  conversation_id: string;
  refused?: boolean;
  refusal_reason?: RefusalReason | null;
  citations?: Citation[];
  input_tokens?: number;
  output_tokens?: number;
  cost_usd?: number;
  latency_ms?: number;
  /** 是否走了降级路径（供应商降级/rerank 挂） */
  degraded?: boolean;
}
