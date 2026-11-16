/**
 * 对话流：SSE 订阅 + 增量状态管理。
 *
 * ## 核心难点：流式增量怎么进 React 状态
 *
 * SSE 一次请求会推送几十上百个 `delta`。如果每个 delta 都触发一次
 * `setState`，React 会渲染上百次（虽然 React 18 有批处理，但异步回调里的
 * 更新不会自动批处理）。中文回答通常 200~500 token，意味着几百次渲染。
 *
 * 解法两层：
 * 1. **用 `useRef` 累积文本**，不直接进 state；
 * 2. **用 30ms 节流刷新 state**（约 33fps，肉眼完全流畅，渲染次数降到 1/10）。
 *
 * 流结束时立即 flush 一次（避免最后几个字没刷出来）。
 *
 * ## 为什么用 reducer 而不是多个 useState
 *
 * 一次对话有 5 个相关状态（消息列表、流式文本、引用、状态、错误）。
 * 分成 5 个 useState 会出现"更新 3 个 setState 中途抛错 → 状态不一致"。
 * 一个 reducer 保证状态转移原子。
 */

import { useCallback, useEffect, useReducer, useRef } from "react";
import type { Citation } from "@contracts";
import { postSse } from "@/lib/sse";
import { tokenStore } from "@/lib/api";

/** 流式状态（与后端 `MessageStatus` 对齐，但前端多了 idle）。 */
export type StreamPhase = "idle" | "retrieving" | "streaming" | "done" | "error";

export interface StreamState {
  phase: StreamPhase;
  /**
   * 本轮提问原文。
   *
   * 为什么放进 state 而不是让调用方从输入框反推：发送后输入框立刻清空，
   * 从 `draft` 反推只能拿到空串。初版就是这么错的——新会话第一次提问时
   * 用户气泡是**空的**（没有历史消息可回退）。
   */
  query: string;
  /** 已生成的文本（流式过程中持续增长）。 */
  text: string;
  /**
   * 检索命中的材料段数（`retrieval` 事件给）。
   *
   * **不是引用数**：检索 5 段不代表答案引用了 5 段。只用于在等待首
   * token 时显示"已检索到 N 段资料"，让等待可感知但不给出错误暗示。
   */
  sourcesFound: number;
  /**
   * 引用列表（`done` 事件给）。
   *
   * ## 为什么只在结束时才填
   *
   * 初版在 `retrieval` 事件里就填了引用（当时是"全部检索命中"），
   * 于是答案还没生成就显示「来源 1 2 3 4 5」，而且那些编号与答案里的
   * `[1]` `[2]` 完全不对应——点开引用会看到错的原文。
   *
   * 引用是"答案的哪句话依据了哪段原文"，答案没定稿时这个映射不存在。
   * 所以后端改为在生成结束后按答案里的 `[n]` 标记组装，前端只在
   * `done` 时渲染。
   */
  citations: Citation[];
  /** 后端返回的消息 ID（`done` 事件给，用于点赞点踩）。 */
  messageId: string | null;
  conversationId: string | null;
  refused: boolean;
  errorMessage: string | null;
  /** 是否已收到至少一个 delta（用于判断"首个 token 还没来"）。 */
  receivedFirstToken: boolean;
}

type Action =
  | { type: "start"; query: string }
  | { type: "sourcesReady"; count: number; refused: boolean }
  | { type: "delta"; text: string }
  | {
      type: "done";
      messageId: string;
      conversationId: string;
      citations: Citation[];
      refused: boolean;
    }
  | { type: "error"; message: string }
  | { type: "reset" };

const INITIAL: StreamState = {
  phase: "idle",
  query: "",
  text: "",
  sourcesFound: 0,
  citations: [],
  messageId: null,
  conversationId: null,
  refused: false,
  errorMessage: null,
  receivedFirstToken: false,
};

function reducer(state: StreamState, action: Action): StreamState {
  switch (action.type) {
    case "start":
      return { ...INITIAL, phase: "retrieving", query: action.query };
    case "sourcesReady":
      // 检索完成：只切阶段、记数量。**不碰 citations**——那时还没有引用。
      return {
        ...state,
        phase: "streaming",
        sourcesFound: action.count,
        refused: action.refused,
      };
    case "delta":
      return {
        ...state,
        text: state.text + action.text,
        receivedFirstToken: true,
      };
    case "done":
      return {
        ...state,
        phase: "done",
        messageId: action.messageId,
        conversationId: action.conversationId,
        // 无条件覆盖：`done` 的 citations 是**答案实际引用的**，
        // 为空就是"答案没引用任何材料"（正确结果，不是缺失）。
        // 初版有 `length > 0 ? ... : state.citations` 的回退，
        // 那是为了保留 retrieval 先发的那批——现在没有可保留的了。
        citations: action.citations,
        refused: action.refused,
      };
    case "error":
      return { ...state, phase: "error", errorMessage: action.message };
    case "reset":
      return INITIAL;
  }
}

/** 增量刷新节流间隔（毫秒）。33fps 对文字滚动完全够，且渲染量降一个数量级。 */
const FLUSH_INTERVAL_MS = 30;

export interface SendOptions {
  query: string;
  conversationId?: string | null;
  knowledgeBaseIds?: string[];
}

export function useChat() {
  const [state, dispatch] = useReducer(reducer, INITIAL);

  /**
   * 累积缓冲区。
   *
   * 为什么不直接 `dispatch({type:"delta"})`：见模块 docstring 的渲染次数说明。
   * ref 不触发渲染，所以累积本身是零成本的。
   */
  const pendingText = useRef("");
  const flushTimer = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const flush = useCallback(() => {
    if (pendingText.current.length === 0) return;
    const chunk = pendingText.current;
    pendingText.current = "";
    dispatch({ type: "delta", text: chunk });
  }, []);

  const scheduleFlush = useCallback(() => {
    if (flushTimer.current !== null) return;
    flushTimer.current = window.setTimeout(() => {
      flushTimer.current = null;
      flush();
    }, FLUSH_INTERVAL_MS);
  }, [flush]);

  const stop = useCallback(() => {
    // 主动中止：服务端会感知断连并落库部分内容（A16 已实现）
    abortRef.current?.abort();
    abortRef.current = null;
    if (flushTimer.current !== null) {
      window.clearTimeout(flushTimer.current);
      flushTimer.current = null;
    }
    flush();
  }, [flush]);

  const send = useCallback(
    async (options: SendOptions) => {
      // 上一次还在流：先停掉（避免两条流同时往一个 state 写，文本会交错）
      stop();
      dispatch({ type: "start", query: options.query });

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await postSse(
          {
            url: "/api/v1/chat",
            body: {
              query: options.query,
              conversation_id: options.conversationId ?? null,
              knowledge_base_ids: options.knowledgeBaseIds ?? [],
            },
            headers: tokenStore.access
              ? { Authorization: `Bearer ${tokenStore.access}` }
              : {},
            signal: controller.signal,
          },
          {
            onEvent: ({ event, data }) => {
              const payload = data as Record<string, unknown>;
              switch (event) {
                case "retrieval":
                  // 只拿数量，不拿引用——后端已经不再在这个事件里发 citations。
                  // 用 `Number.isFinite` 兜底：字段缺失时 `Number(undefined)`
                  // 是 NaN，会让 UI 显示"已检索到 NaN 段"。
                  dispatch({
                    type: "sourcesReady",
                    count: Number.isFinite(Number(payload["sources_found"]))
                      ? Number(payload["sources_found"])
                      : 0,
                    refused: Boolean(payload["refused"]),
                  });
                  break;

                case "delta": {
                  const text = String(payload["text"] ?? "");
                  if (!text) break;
                  pendingText.current += text;
                  scheduleFlush();
                  break;
                }

                case "done": {
                  // 结束时必须 flush：否则最后 30ms 内的文字会丢
                  flush();
                  dispatch({
                    type: "done",
                    messageId: String(payload["message_id"] ?? ""),
                    conversationId: String(payload["conversation_id"] ?? ""),
                    citations: (payload["citations"] as Citation[]) ?? [],
                    refused: Boolean(payload["refused"]),
                  });
                  break;
                }

                case "error":
                  flush();
                  dispatch({
                    type: "error",
                    message: String(payload["message"] ?? "回答生成失败"),
                  });
                  break;

                case "http_error":
                  // 流还没开始就失败（401/409/429），错误体是普通 JSON
                  dispatch({
                    type: "error",
                    message: String(payload["message"] ?? "请求失败"),
                  });
                  break;

                default:
                  break;
              }
            },
            onError: (error) => {
              flush();
              dispatch({ type: "error", message: error.message });
            },
          },
        );
      } catch (error) {
        flush();
        dispatch({
          type: "error",
          message: error instanceof Error ? error.message : "网络异常",
        });
      } finally {
        abortRef.current = null;
      }
    },
    [flush, scheduleFlush, stop],
  );

  const reset = useCallback(() => {
    stop();
    dispatch({ type: "reset" });
  }, [stop]);

  // 组件卸载时中止流：不中止会让服务端继续生成（浪费成本）
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (flushTimer.current !== null) window.clearTimeout(flushTimer.current);
    };
  }, []);

  return { state, send, stop, reset };
}
