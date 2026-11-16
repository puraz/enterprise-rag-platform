/**
 * SSE 客户端：POST + 自定义头 + 手写解析。
 *
 * ## 为什么不用浏览器内置的 EventSource
 *
 * `EventSource` 有三个硬限制，每一个都让它在真实项目里不可用：
 *
 * 1. **只能 GET**：我们的 `/api/v1/chat` 需要 POST body（问题文本可能 4000 字，
 *    URL 装不下且会被日志记录）。
 * 2. **不能设自定义头**：必须带 `Authorization: Bearer <token>`。
 * 3. **不能读响应头/状态码**：401 时无法区分"token 过期"和"限流"，
 *    只能看到一次无信息的 `error` 事件。
 *
 * 所以用 `fetch` + `ReadableStream` 手写。核心难点是**跨 chunk 的半行**：
 * TCP 不保证按行切分，一个 `data: {...}` 可能被拆成两个 chunk。
 *
 * ## 三个必须处理的情况
 *
 * - **半行缓冲**：保留最后一个不完整的行，下次拼接（`buffer` 变量）。
 * - **空行分隔事件**：SSE 用空行分隔事件，不是换行。
 * - **`[DONE]` 哨兵**：有些服务端用它结束流（我们用自己的 `done` 事件，
 *   但也兼容）。
 */

export interface SseHandlers {
  /** 收到一个完整事件。`event.event` 是事件名，`data` 已 JSON 解析。 */
  onEvent: (event: { event: string; data: unknown }) => void;
  /** 流正常结束（服务端关闭连接）。 */
  onClose?: () => void;
  /** 网络/解析错误。注意：业务错误走 `onEvent` 的 `error` 事件，不是这里。 */
  onError?: (error: Error) => void;
}

export interface SseOptions {
  url: string;
  body: unknown;
  headers?: Record<string, string>;
  /** 由调用方控制取消（组件卸载 / 用户点"停止"）。 */
  signal?: AbortSignal;
}

/**
 * 发起一次 POST SSE 请求。
 *
 * 返回 Promise：在流**完全读完**（或出错）时 resolve/reject。
 * 事件通过 handlers 回调推送——不做成 AsyncIterable 是因为
 * React 里用回调更自然（避免在 effect 里手写 for-await + 取消逻辑）。
 */
export async function postSse(options: SseOptions, handlers: SseHandlers): Promise<void> {
  const { url, body, headers = {}, signal } = options;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...headers,
      },
      body: JSON.stringify(body),
      // exactOptionalPropertyTypes 下不能传 `undefined`，
      // 必须显式给 null 或省略这个键
      ...(signal ? { signal } : {}),
    });
  } catch (error) {
    // 主动取消不算错误：组件卸载时会 abort，不该弹提示
    if (isAbortError(error)) return;
    handlers.onError?.(error instanceof Error ? error : new Error(String(error)));
    return;
  }

  // 非 2xx：服务端返回的是普通 JSON 错误体（流还没开始）
  if (!response.ok) {
    const payload = await safeJson(response);
    handlers.onEvent({ event: "http_error", data: { status: response.status, ...payload } });
    return;
  }

  if (!response.body) {
    handlers.onError?.(new Error("响应没有 body（服务端未返回流）"));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // `stream: true` 让解码器正确处理跨 chunk 的多字节字符
      // （中文一个字符 3 字节，可能被拆分到两个 chunk）
      buffer += decoder.decode(value, { stream: true });

      // SSE 事件以**空行**分隔（`\n\n` 或 `\r\n\r\n`）
      let boundary = findBoundary(buffer);
      while (boundary !== -1) {
        const rawEvent = buffer.slice(0, boundary.index);
        // 必须**跨过分隔符**再继续。只 slice 到 boundary.index 会让 buffer
        // 仍以 `\n\n` 开头 → findBoundary 恒返回 0 → 死循环卡死主线程。
        buffer = buffer.slice(boundary.index + boundary.length);
        const parsed = parseEvent(rawEvent);
        if (parsed) handlers.onEvent(parsed);
        boundary = findBoundary(buffer);
      }
    }

    // 流结束：处理缓冲区里可能残留的最后一个事件（服务端未加空行时）
    const tail = parseEvent(buffer);
    if (tail) handlers.onEvent(tail);

    handlers.onClose?.();
  } catch (error) {
    if (isAbortError(error)) return;
    handlers.onError?.(error instanceof Error ? error : new Error(String(error)));
  } finally {
    // 必须释放 reader：不释放会让连接保持打开，泄漏连接
    reader.releaseLock();
  }
}

interface Boundary {
  index: number;
  length: number;
}

/** 找事件边界（空行）。返回边界**起点**与分隔符长度；没有则返回 -1。 */
function findBoundary(buffer: string): Boundary | -1 {
  const lf = buffer.indexOf("\n\n");
  const crlf = buffer.indexOf("\r\n\r\n");
  if (lf === -1 && crlf === -1) return -1;
  if (crlf !== -1 && (lf === -1 || crlf < lf)) {
    return { index: crlf, length: 4 };
  }
  return { index: lf, length: 2 };
}

/**
 * 解析一个 SSE 事件块。
 *
 * 格式：
 * ```
 * event: delta
 * data: {"text":"住"}
 * ```
 *
 * 规范允许 `data` 出现多行（用 `\n` 连接），也允许只有 `data` 没有 `event`
 * （此时事件名默认为 `message`）。两种都处理。
 */
function parseEvent(raw: string): { event: string; data: unknown } | null {
  const lines = raw.split(/\r?\n/);
  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) continue; // 空行与注释（心跳）跳过
    const colon = line.indexOf(":");
    if (colon === -1) continue;

    const field = line.slice(0, colon);
    // 规范：冒号后若有单个空格，去掉它
    let value = line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") eventName = value;
    else if (field === "data") dataLines.push(value);
  }

  if (dataLines.length === 0) return null;

  const payload = dataLines.join("\n");
  if (payload === "[DONE]") return { event: "done", data: null };

  try {
    return { event: eventName, data: JSON.parse(payload) };
  } catch {
    // 解析失败不抛异常（会让整个流中断）：当成纯文本事件交给上层
    return { event: eventName, data: payload };
  }
}

function isAbortError(error: unknown): boolean {
  return (
    error instanceof DOMException &&
    (error.name === "AbortError" || error.name === "TimeoutError")
  );
}

async function safeJson(response: Response): Promise<Record<string, unknown>> {
  try {
    return (await response.json()) as Record<string, unknown>;
  } catch {
    return {};
  }
}
