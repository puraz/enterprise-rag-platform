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
