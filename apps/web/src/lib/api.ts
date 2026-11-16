/**
 * API 客户端：统一请求封装 + 错误码映射 + 401 刷新重试。
 *
 * ## 为什么自己封装而不是直接用 fetch
 *
 * 四件事每个请求都要做，散落在各调用点必然漏：
 * 1. 带 Authorization 头；
 * 2. 把 `ErrorResponse` 转成可判断的错误对象（前端要按 `code` 分支）；
 * 3. 401 时尝试刷新 token 并重放原请求；
 * 4. 限流（429）时提取 `retry_after_s` 给用户明确提示。
 *
 * ## 错误形状
 *
 * 后端契约（AGENTS.md 4.1）：失败响应统一是
 * `{code, code_name, message, request_id, fields, details}`。
 * 前端把它包成 `ApiError` 类，保留 `code`（数字）与 `codeName`（语义名）：
 * - `code` 用于 `switch` 与号段判断（如 `1xxxx` 都是认证问题）；
 * - `codeName` 用于日志与用户提示的兜底文案。
 */

import type { ErrorResponse, FieldError } from "@contracts";

/** 后端返回的业务错误。 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: number;
  readonly codeName: string;
  readonly requestId: string | undefined;
  readonly fields: FieldError[];
  readonly details: Record<string, unknown>;

  constructor(params: {
    status: number;
    code: number;
    codeName: string;
    message: string;
    requestId?: string;
    fields?: FieldError[];
    details?: Record<string, unknown>;
  }) {
    super(params.message);
    this.name = "ApiError";
    this.status = params.status;
    this.code = params.code;
    this.codeName = params.codeName;
    this.requestId = params.requestId;
    this.fields = params.fields ?? [];
    this.details = params.details ?? {};
  }

  /** 是否是认证问题（1xxxx 号段：token 过期/无效/未登录）。 */
  get isAuthError(): boolean {
    return this.code >= 10000 && this.code < 20000;
  }

  /** 是否是限流（前端应展示"请稍后重试"+ 精确秒数）。 */
  get isRateLimited(): boolean {
    return this.code === 40001 || this.status === 429;
  }

  /** 限流建议的重试秒数（服务端在 details 里给）。 */
  get retryAfterSeconds(): number | undefined {
    const value = this.details["retry_after_s"];
    return typeof value === "number" ? value : undefined;
  }
}

/** token 存取。**内存 + localStorage 双写**：见下方说明。 */
const TOKEN_KEY = "rag.access_token";
const REFRESH_KEY = "rag.refresh_token";

/**
 * ## 为什么 access token 也放 localStorage
 *
 * 严格安全做法是 access token 只放内存（XSS 拿不到），refresh token 放
 * httpOnly cookie（JS 读不到）。但那需要后端支持 cookie 且处理 CSRF。
 *
 * 本项目后端用 Bearer token（`Authorization` 头），刷新也走 JSON body，
 * 所以只能放 JS 可读的地方。为了降低 XSS 风险，采取：
 * - access token 短有效期（15 分钟，后端配置）；
 * - 刷新时轮换 refresh token（后端已实现复用检测）；
 * - 渲染 markdown 时用 `rehype-sanitize`（防 XSS 的主要手段，见 Markdown.tsx）。
 *
 * **这不是最安全方案**，是在"后端契约已定"前提下的合理取舍。
 * 若将来后端支持 httpOnly cookie，应迁移。
 */
export const tokenStore = {
  get access(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  },
  get refresh(): string | null {
    return localStorage.getItem(REFRESH_KEY);
  },
  set(access: string, refresh: string): void {
    localStorage.setItem(TOKEN_KEY, access);
    localStorage.setItem(REFRESH_KEY, refresh);
  },
  clear(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_KEY);
  },
};

/** 单飞刷新：并发 401 时只发一次刷新请求。 */
let refreshPromise: Promise<boolean> | null = null;

async function refreshToken(): Promise<boolean> {
  // 已有刷新在跑：复用同一个 Promise（否则 5 个并发 401 会发 5 次刷新，
  // 而 refresh token 是**轮换**的——后 4 次会因"已使用"而失败，
  // 触发后端的"复用检测"把整族作废，用户被迫重新登录）
  if (refreshPromise) return refreshPromise;

  refreshPromise = (async () => {
    const refresh = tokenStore.refresh;
    if (!refresh) return false;
    try {
      const response = await fetch("/api/v1/auth/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refresh }),
      });
      if (!response.ok) return false;
      const data = (await response.json()) as {
        access_token: string;
        refresh_token: string;
      };
      tokenStore.set(data.access_token, data.refresh_token);
      return true;
    } catch {
      return false;
    } finally {
      // 清空以便下次重新发起（不能不清：token 再过期时不会刷新了）
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}

export interface RequestOptions {
  method?: "GET" | "POST" | "PATCH" | "PUT" | "DELETE";
  body?: unknown;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  /** 内部用：标记这是重放（避免无限刷新循环）。 */
  _retried?: boolean;
}

/**
 * 发一次 API 请求。
 *
 * 成功返回解析后的 JSON（204 返回 undefined）。
 * 失败抛 `ApiError`（网络错误抛原始 Error）。
 */
export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, headers = {}, signal, _retried = false } = options;

  const accessToken = tokenStore.access;
  const response = await fetch(path, {
    method,
    headers: {
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
      ...headers,
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    ...(signal ? { signal } : {}),
  });

  if (response.status === 204) {
    return undefined as T;
  }

  if (response.ok) {
    return (await response.json()) as T;
  }

  // 401 且未重放过：尝试刷新后重放一次
  if (response.status === 401 && !_retried) {
    const refreshed = await refreshToken();
    if (refreshed) {
      return apiFetch<T>(path, { ...options, _retried: true });
    }
    // 刷新失败：清空 token（避免带着废 token 反复请求）
    tokenStore.clear();
  }

  throw await toApiError(response);
}

/** 把错误响应转成 ApiError。 */
async function toApiError(response: Response): Promise<ApiError> {
  let payload: Partial<ErrorResponse> = {};
  try {
    payload = (await response.json()) as Partial<ErrorResponse>;
  } catch {
    // 非 JSON 响应（如网关的 HTML 502）：保留状态码
  }

  return new ApiError({
    status: response.status,
    code: payload.code ?? 0,
    codeName: payload.code_name ?? "UNKNOWN",
    message: payload.message ?? defaultMessage(response.status),
    ...(payload.request_id ? { requestId: payload.request_id } : {}),
    ...(payload.fields ? { fields: payload.fields } : {}),
    ...(payload.details ? { details: payload.details as Record<string, unknown> } : {}),
  });
}

function defaultMessage(status: number): string {
  if (status === 401) return "登录已过期，请重新登录";
  if (status === 403) return "没有权限执行此操作";
  if (status === 404) return "资源不存在";
  if (status === 429) return "请求过于频繁，请稍后重试";
  if (status >= 500) return "服务暂时不可用，请稍后重试";
  return "请求失败";
}
