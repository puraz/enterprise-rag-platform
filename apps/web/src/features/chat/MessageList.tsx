/**
 * 消息列表：用户消息 + 助手回答 + 引用 + 反馈。
 *
 * ## 布局判断：为什么用户消息右对齐、助手消息左对齐
 *
 * 这是聊天界面的通用约定（微信/iMessage/ChatGPT）。但纯粹左右对齐有个问题：
 * **助手回答较长时，右缩进会浪费宽度**。
 *
 * 所以这里的做法：
 * - 用户消息：右对齐 + 气泡（浅灰底，明确区分发言方）
 * - 助手消息：**左对齐但不用气泡**（白底 + 左侧 2px 竖线标识归属）
 *
 * 不用气泡的原因：长回答被气泡包住会让每行变窄 20%，阅读长文很累。
 * 左侧竖线同样能表达"这是助手"，而且不侵蚀行宽。
 *
 * ## 流式时为什么"先显示引用再显示文字"
 *
 * 首 token 延迟通常 1~3 秒（LLM 首字延迟）。这段时间界面如果完全空白，
 * 用户会怀疑卡住了。引用在检索完成时就有（`retrieval` 事件），
 * 先渲染它能让等待变得**可感知**——用户在读引用，就不觉得在等。
 */

import { useEffect, useRef, useState } from "react";
import { ThumbsUp, ThumbsDown, Copy, Check, Square } from "lucide-react";
import type { MessageItem } from "@contracts";
import { Markdown } from "@/components/ui/Markdown";
import { CitationList } from "./CitationCard";
import type { StreamPhase } from "./useChat";

export interface PendingTurn {
  query: string;
  phase: StreamPhase;
  text: string;
  /**
   * 检索命中的段数（不是引用数）。
   *
   * 生成期间只显示"已检索到 N 段资料"，**不显示引用编号**——
   * 答案还没定稿，此时还不知道模型会引用哪几条，提前给编号会让
   * 用户看到与答案里 `[n]` 对不上的项。
   */
  sourcesFound: number;
  receivedFirstToken: boolean;
  errorMessage: string | null;
  refused: boolean;
}

export interface MessageListProps {
  messages: MessageItem[];
  pending: PendingTurn | null;
  onFeedback: (messageId: string, rating: "like" | "dislike") => void;
  onStop: () => void;
}

export function MessageList({ messages, pending, onFeedback, onStop }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // 自动滚到底：新内容出现时保持"最新在视野内"。
  // 用 `scrollIntoView({block:"end"})` 而不是 `scrollTop = scrollHeight`：
  // 后者在虚拟列表或延迟布局时会跳。
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages.length, pending?.text, pending?.phase]);

  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-6">
      {messages.map((message) => (
        <MessageRow key={message.id} message={message} onFeedback={onFeedback} />
      ))}

      {pending !== null && <PendingRow pending={pending} onStop={onStop} />}

      <div ref={bottomRef} className="h-1" />
    </div>
  );
}

function MessageRow({
  message,
  onFeedback,
}: {
  message: MessageItem;
  onFeedback: (messageId: string, rating: "like" | "dislike") => void;
}) {
  if (message.role === "user") {
    return (
      <div className="animate-msg-in mb-6 flex justify-end">
        <div className="max-w-[80%] rounded-md bg-ink-150 px-3.5 py-2.5 text-[14px] text-ink-900 whitespace-pre-wrap break-words">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="group animate-msg-in mb-6 border-l-2 border-ink-200 pl-4">
      {message.refused ? (
        <RefusalNotice reason={message.refusal_reason} />
      ) : (
        <Markdown content={message.content} />
      )}

      {message.citations && message.citations.length > 0 && (
        <CitationList citations={message.citations} />
      )}

      <MessageActions message={message} onFeedback={onFeedback} />
    </div>
  );
}

function PendingRow({ pending, onStop }: { pending: PendingTurn; onStop: () => void }) {
  const isStreaming = pending.phase === "streaming" || pending.phase === "retrieving";

  return (
    <>
      {/* 用户刚发的问题（还没落库，所以不在 messages 里） */}
      <div className="animate-msg-in mb-6 flex justify-end">
        <div className="max-w-[80%] rounded-md bg-ink-150 px-3.5 py-2.5 text-[14px] text-ink-900 whitespace-pre-wrap break-words">
          {pending.query}
        </div>
      </div>

      <div className="group animate-msg-in mb-6 border-l-2 border-ink-200 pl-4">
        {/*
         * 检索完成但首 token 还没到：显示状态文案 + 命中段数。
         *
         * **这里刻意不渲染引用编号**。答案还没定稿，模型会引用哪几条
         * 尚未确定；提前给出编号会让用户看到与答案里 `[n]` 对不上的项
         * （初版就是这个错误）。`done` 之后才由 MessageRow 渲染真正的引用。
         */}
        {!pending.receivedFirstToken && isStreaming && (
          <div className="flex items-center gap-2 text-[13px] text-ink-500">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-ink-400" />
            {pending.phase === "retrieving" ? (
              "检索资料中"
            ) : (
              <>
                生成回答中
                {pending.sourcesFound > 0 && (
                  <span className="text-ink-400">
                    · 已检索到 {pending.sourcesFound} 段资料
                  </span>
                )}
              </>
            )}
          </div>
        )}

        {pending.text.length > 0 && (
          <Markdown content={pending.text} streaming={pending.phase === "streaming"} />
        )}

        {pending.errorMessage !== null && (
          <div className="mt-2 rounded-md border border-danger/20 bg-danger-soft px-3 py-2 text-[13px] text-danger">
            {pending.errorMessage}
          </div>
        )}

        {isStreaming && (
          <button
            type="button"
            className="mt-3 inline-flex items-center gap-1.5 text-[12px] text-ink-500
                       hover:text-ink-800 transition-colors"
            onClick={onStop}
          >
            <Square size={10} fill="currentColor" aria-hidden />
            停止生成
          </button>
        )}
      </div>
    </>
  );
}

function RefusalNotice({ reason }: { reason: MessageItem["refusal_reason"] }) {
  const text =
    reason === "no_materials"
      ? "知识库中没有找到相关资料。"
      : reason === "insufficient_evidence"
        ? "找到的资料不足以回答这个问题。"
        : "无法安全地回答这个问题。";

  return (
    <div className="rounded-md border border-ink-200 bg-ink-50 px-3 py-2.5 text-[13px] text-ink-600">
      {text}
      <span className="mt-1 block text-[12px] text-ink-500">
        可以尝试换个说法，或确认相关文档已上传。
      </span>
    </div>
  );
}

function MessageActions({
  message,
  onFeedback,
}: {
  message: MessageItem;
  onFeedback: (messageId: string, rating: "like" | "dislike") => void;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // 剪贴板 API 在非安全上下文（http）下不可用：静默失败，
      // 不做提示（用户会觉得"点了没反应"，但这是浏览器限制）
    }
  };

  // 拒答消息不给点赞点踩：没有内容可评价
  if (message.refused) return null;

  return (
    <div className="mt-2 flex items-center gap-0.5 opacity-0 transition-opacity
                    duration-150 group-hover:opacity-100 focus-within:opacity-100">
      <IconAction
        label={copied ? "已复制" : "复制"}
        onClick={() => void copy()}
        active={copied}
      >
        {copied ? <Check size={13} /> : <Copy size={13} />}
      </IconAction>

      <IconAction
        label="有帮助"
        onClick={() => onFeedback(message.id, "like")}
        active={message.rating === "like"}
        activeClass="text-success"
      >
        <ThumbsUp size={13} />
      </IconAction>

      <IconAction
        label="没帮助"
        onClick={() => onFeedback(message.id, "dislike")}
        active={message.rating === "dislike"}
        activeClass="text-danger"
      >
        <ThumbsDown size={13} />
      </IconAction>
    </div>
  );
}

function IconAction({
  label,
  onClick,
  children,
  active = false,
  activeClass = "text-ink-900",
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
  active?: boolean;
  activeClass?: string;
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      className={`inline-flex h-6 w-6 items-center justify-center rounded-sm
                  text-ink-400 hover:bg-ink-100 hover:text-ink-700
                  transition-colors duration-100 ${active ? activeClass : ""}`}
    >
      {children}
    </button>
  );
}
