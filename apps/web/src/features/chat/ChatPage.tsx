/**
 * 对话页：消息区 + 输入区。
 *
 * ## 布局的两个关键决定
 *
 * ### 1. 输入框固定在底部，消息区独立滚动
 *
 * 用 `flex-col` + `flex-1 overflow-y-auto` 让消息区滚动、输入区固定。
 * 不用 `position: fixed`——那会让输入区与内容区宽度不一致（侧栏宽度变化时）。
 *
 * ### 2. 输入框随内容长高，但有上限
 *
 * textarea 从 1 行开始，随输入长高到 6 行（约 140px）后内部滚动。
 * 固定高度会让长问题看不全；无限长高会吃掉消息区。
 *
 * ## 键盘交互
 *
 * - `Enter` 发送，`Shift+Enter` 换行（这是聊天界面的通用约定）
 * - 输入法组合中（`isComposing`）不发送——中文输入时按 Enter 是选词，
 *   不判断会**把半截拼音发出去**。这是中文应用的必做检查。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUp, Square } from "lucide-react";
import type { MessageItem, ConversationItem } from "@contracts";
import type { Page } from "@/lib/types";
import { apiFetch } from "@/lib/api";
import { useCurrentUser } from "@/features/auth/useAuth";
import { AppShell } from "@/components/layout/AppShell";
import { MessageList } from "./MessageList";
import { useChat } from "./useChat";

export function ChatPage() {
  const { data: user } = useCurrentUser();
  const queryClient = useQueryClient();
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { state, send, stop, reset } = useChat();

  // 历史消息（仅当有会话 ID 时拉）
  const { data: history } = useQuery<Page<MessageItem>>({
    queryKey: ["chat", "messages", conversationId],
    queryFn: () =>
      apiFetch<Page<MessageItem>>(
        `/api/v1/conversations/${conversationId}/messages?limit=50`,
      ),
    enabled: conversationId !== null,
  });

  // 会话列表（侧栏）
  const { data: conversations } = useQuery<Page<ConversationItem>>({
    queryKey: ["chat", "conversations"],
    queryFn: () => apiFetch<Page<ConversationItem>>("/api/v1/conversations?limit=30"),
  });

  // 流结束后刷新历史与会话列表（新消息要出现在列表里）
  useEffect(() => {
    if (state.phase === "done" && state.conversationId) {
      setConversationId(state.conversationId);
      void queryClient.invalidateQueries({ queryKey: ["chat", "messages"] });
      void queryClient.invalidateQueries({ queryKey: ["chat", "conversations"] });
    }
  }, [state.phase, state.conversationId, queryClient]);

  const submit = useCallback(() => {
    const query = draft.trim();
    if (!query || state.phase === "streaming" || state.phase === "retrieving") return;
    setDraft("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    void send({ query, conversationId });
  }, [draft, state.phase, send, conversationId]);

  const handleFeedback = useCallback(
    (messageId: string, rating: "like" | "dislike") => {
      void apiFetch(`/api/v1/messages/${messageId}/feedback`, {
        method: "POST",
        body: { rating, ...(rating === "dislike" ? { reason: "other" } : {}) },
      })
        .then(() => {
          void queryClient.invalidateQueries({ queryKey: ["chat", "messages"] });
        })
        .catch((error: unknown) => {
          /*
           * 必须把失败暴露出来。
           *
           * 初版这里没有 catch，422 变成一个 unhandled rejection：
           * 用户点了没反应、图标不变、控制台之外无迹可寻。
           * 反馈是评估集的输入，静默丢失比报错更糟。
           */
          window.alert(
            error instanceof Error ? `反馈提交失败：${error.message}` : "反馈提交失败",
          );
        });
    },
    [queryClient],
  );

  const startNew = useCallback(() => {
    reset();
    setConversationId(null);
  }, [reset]);

  const isBusy = state.phase === "streaming" || state.phase === "retrieving";
  const messages = history?.items ?? [];

  /**
   * 流式行何时隐藏。
   *
   * ## 为什么不能用 `phase === "done"` 直接隐藏
   *
   * `done` 之后我们才 invalidate 历史查询，重新拉到数据还有一段网络时间。
   * 立刻隐藏 → 回答先消失再出现（闪烁）。
   *
   * ## 为什么也不能一直显示
   *
   * 初版只判断 `phase !== "idle"`，结果 `done` 后历史已含新消息，
   * **同一个回答渲染两遍**（实测截图里能看到）。
   *
   * 正确判据是"历史里已经出现这条 message_id"——它是后端落库的权威结果，
   * 出现了说明流式行的使命结束。
   */
  const streamAlreadyPersisted =
    state.messageId !== null && messages.some((message) => message.id === state.messageId);
  const showPending = state.phase !== "idle" && !streamAlreadyPersisted;

  return (
    <AppShell
      userEmail={user?.email}
      sidebar={
        <div className="py-1">
          <p className="px-2 pb-1 text-[11px] font-medium uppercase tracking-wider text-ink-500">
            最近对话
          </p>
          {conversations?.items.length === 0 && (
            <p className="px-2 py-2 text-[12px] text-ink-500">还没有对话</p>
          )}
          {conversations?.items.map((conversation) => (
            <button
              key={conversation.id}
              type="button"
              onClick={() => {
                reset();
                setConversationId(conversation.id);
              }}
              className={`block w-full truncate rounded-md px-2 py-1.5 text-left
                          text-[12px] transition-colors duration-100
                          ${
                            conversationId === conversation.id
                              ? "bg-ink-900 text-white"
                              : "text-ink-400 hover:bg-ink-900 hover:text-white"
                          }`}
            >
              {conversation.title}
            </button>
          ))}
        </div>
      }
    >
      <div className="flex h-full flex-col">
        {/* 消息区 */}
        <div className="min-h-0 flex-1 overflow-y-auto">
          {messages.length === 0 && !showPending ? (
            <EmptyState onPick={(text) => setDraft(text)} />
          ) : (
            <MessageList
              messages={messages}
              pending={
                showPending
                  ? {
                      query: state.query,
                      phase: state.phase,
                      text: state.text,
                      sourcesFound: state.sourcesFound,
                      receivedFirstToken: state.receivedFirstToken,
                      errorMessage: state.errorMessage,
                      refused: state.refused,
                    }
                  : null
              }
              onFeedback={handleFeedback}
              onStop={stop}
            />
          )}
        </div>

        {/* 输入区 */}
        <div className="border-t border-ink-200 bg-white">
          <div className="mx-auto w-full max-w-3xl px-6 py-3">
            <div className="flex items-end gap-2 rounded-md border border-ink-200
                            bg-white px-2.5 py-1.5
                            focus-within:border-ink-400 transition-colors duration-100">
              <textarea
                ref={textareaRef}
                value={draft}
                rows={1}
                placeholder="问点什么…"
                className="max-h-[140px] min-h-[24px] flex-1 resize-none bg-transparent
                           text-[14px] leading-[1.5] text-ink-900
                           placeholder:text-ink-400 focus:outline-none"
                onChange={(event) => {
                  setDraft(event.target.value);
                  // 自适应高度：先重置再按 scrollHeight 设（否则只能长不能缩）
                  const element = event.target;
                  element.style.height = "auto";
                  element.style.height = `${Math.min(element.scrollHeight, 140)}px`;
                }}
                onKeyDown={(event) => {
                  // 输入法组合中：Enter 是选词，不发送
                  if (event.nativeEvent.isComposing) return;
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    submit();
                  }
                }}
              />
              {isBusy ? (
                <button
                  type="button"
                  aria-label="停止生成"
                  className="mb-0.5 flex h-7 w-7 shrink-0 items-center justify-center
                             rounded-md bg-ink-900 text-white hover:bg-ink-800"
                  onClick={stop}
                >
                  <Square size={11} fill="currentColor" />
                </button>
              ) : (
                <button
                  type="button"
                  aria-label="发送"
                  disabled={draft.trim().length === 0}
                  className="mb-0.5 flex h-7 w-7 shrink-0 items-center justify-center
                             rounded-md bg-ink-900 text-white
                             hover:bg-ink-800 disabled:opacity-30
                             transition-colors duration-100"
                  onClick={submit}
                >
                  <ArrowUp size={15} />
                </button>
              )}
            </div>
            <p className="mt-1.5 text-[11px] text-ink-400">
              回答基于知识库内容，请核对引用来源。Enter 发送，Shift + Enter 换行。
              {conversationId !== null && (
                <button
                  type="button"
                  className="ml-2 text-ink-500 hover:text-ink-800 underline underline-offset-2"
                  onClick={startNew}
                >
                  开始新对话
                </button>
              )}
            </p>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

/** 空态：给用户三个可点击的示例问题（比空白页更有引导性）。 */
function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  const samples = [
    "一线城市出差住宿每晚最多能报销多少？",
    "员工年假怎么计算？",
    "信息安全事件应该向谁报告？",
  ];

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col justify-center px-6 pb-24">
      <h1 className="text-[17px] font-medium text-ink-900">知识助手</h1>
      <p className="mt-1 text-[13px] text-ink-500">
        基于已上传的企业文档回答问题，并给出来源引用。
      </p>
      <div className="mt-5 flex flex-col gap-1.5">
        {samples.map((sample) => (
          <button
            key={sample}
            type="button"
            className="w-fit max-w-full truncate rounded-md border border-ink-200
                       px-3 py-1.5 text-left text-[13px] text-ink-700
                       hover:border-ink-300 hover:bg-ink-50
                       transition-colors duration-100"
            onClick={() => onPick(sample)}
          >
            {sample}
          </button>
        ))}
      </div>
    </div>
  );
}
