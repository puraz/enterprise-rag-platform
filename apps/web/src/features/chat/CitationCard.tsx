/**
 * 引用卡片：答案旁的来源标记。
 *
 * ## 设计判断：为什么是"行内编号"而不是"底部来源列表"
 *
 * 两种常见做法：
 *
 * A. 底部列出来源（类似搜索引擎）
 *    - 优点：不打断阅读
 *    - 缺点：用户看到"结论 A"时不知道它来自哪份文档，要滚动到底部再对照编号
 *
 * B. 行内编号 + 悬停/点击展开（本项目选这个）
 *    - 优点：结论与依据在同一视线内，符合"查证"这个真实动作
 *    - 缺点：编号会轻微打断阅读节奏
 *
 * 选 B 的理由：企业问答的核心场景是"我能不能信这个答案"。
 * 把依据放在结论旁边，比放在底部更符合这个动作。
 *
 * ## 交互：悬停与点击是两个状态，不能共用一个布尔值
 *
 * 初版用一个 `expanded` 同时表达"悬停展开"和"点击固定"，结果互相踩：
 *
 * ```
 * 点击 → mouseenter (expanded=true) → click (toggle) → expanded=false
 * ```
 *
 * **点一次永远打不开**（桌面端点引用毫无反应），这是实测抓到的。
 *
 * 拆成两个状态：`hovered`（鼠标在不在）与 `pinned`（点击固定）。
 *
 * ## 为什么 `pinned` 必须由父组件持有
 *
 * 每个卡片各自持有 `pinned` 时，先点引用 1 再点引用 2，两个弹层会
 * **同时开着并互相遮挡**（实测抓到的：`dialogs=2`）。用户没法判断
 * 哪个是刚点的。
 *
 * 所以 `pinned` 上提到 `CitationList`，同一组引用里**最多只有一个固定**。
 * 悬停仍是各卡片自己的（多卡片同时悬停物理上不可能）。
 */

import { useRef, useState } from "react";
import { FileText, ExternalLink } from "lucide-react";
import type { Citation } from "@contracts";

export interface CitationCardProps {
  citation: Citation;
  /**
   * 当前固定（点击）展开的是不是这一条。
   *
   * 由 `CitationList` 持有——同一组引用里最多一个固定。
   */
  isPinned: boolean;
  /** 请求切换固定项（传 `null` 表示全部收起）。 */
  onPin: (index: number | null) => void;
  /** 点击"定位原文"时的回调（Phase B4 接原文查看器）。 */
  onLocate?: (citation: Citation) => void;
}

/**
 * 是否可用于定位（有页码或字符区间）。
 *
 * 后端 `Citation.has_position` 是 Pydantic 的 `@property`，
 * **不会出现在生成的 TS 类型里**（生成器只导出字段）。
 * 所以前端自己判断——这也更明确："什么叫可定位"由前端定义。
 */
function hasPosition(citation: Citation): boolean {
  return citation.page_no != null || (citation.char_start != null && citation.char_end != null);
}

/**
 * 把 chunk 原文里的 markdown 结构标记去掉，只用于**显示**。
 *
 * ## 为什么不能直接渲染 markdown
 *
 * 引用片段最长 300 字，且可能是 PDF/Markdown 任意来源。用 Markdown 组件
 * 渲染会在弹层里塞进标题/列表的块级样式，300 字的片段被 `h2` 撑大后
 * 很难扫读。这里要的是"原样呈现文字"，不是"重新排版一篇文档"。
 *
 * ## 为什么要去掉 `##`
 *
 * 结构化分块会把标题路径带进 chunk 正文（`## 交通标准`），这是**
 * 检索侧的上下文增强**，对 embedding 有益；但展示给用户时
 * `##` 是纯噪音——用户看到的应该是"交通标准"这四个字。
 *
 * 只处理行首的标题符号与列表符号，不动行内内容：
 * 误伤（比如正文里的 `a * b`）的代价比多显示几个符号更大。
 */
function plainQuote(raw: string): string {
  return raw
    .split("\n")
    .map((line) => line.replace(/^#{1,6}\s+/, "")) // 标题：`## 交通标准` → `交通标准`
    .join("\n")
    .replace(/\n{3,}/g, "\n\n") // 连续空行压成一个，避免弹层里大片空白
    .trim();
}

export function CitationCard({ citation, isPinned, onPin, onLocate }: CitationCardProps) {
  const [hovered, setHovered] = useState(false);
  const [dropDown, setDropDown] = useState(false);
  const anchorRef = useRef<HTMLSpanElement>(null);
  const open = hovered || isPinned;

  /**
   * 点击时决定弹层朝上还是朝下。
   *
   * 初版固定 `bottom-full`（朝上）。在消息列表里引用位于回答末尾，
   * 朝上弹会**盖住上一条消息**——用户点开引用后反而看不见当前答案了。
   *
   * 判据：徽章下方空间 >= 弹层高度（约 220px）就朝下，否则朝上。
   * 这是个近似值，但比"永远朝上"或"永远朝下"都稳。
   */
  const toggle = () => {
    if (!isPinned) {
      const rect = anchorRef.current?.getBoundingClientRect();
      const spaceBelow = rect ? window.innerHeight - rect.bottom : 0;
      setDropDown(spaceBelow >= 220);
    }
    onPin(isPinned ? null : citation.index);
  };

  return (
    <span
      ref={anchorRef}
      className="relative inline-block"
      // 整组（徽章 + 卡片）用同一对 enter/leave：
      // 从徽章移到卡片上时 count 不会归零，弹层不会闪
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        className="inline-flex h-[18px] min-w-[18px] items-center justify-center
                   rounded-sm border border-ink-200 bg-white px-1
                   align-super text-[10px] font-medium tabular-nums text-ink-600
                   hover:border-accent-500 hover:text-accent-600
                   transition-colors duration-100"
        onClick={toggle}
        aria-label={`引用 ${citation.index}：${citation.document_title}`}
        aria-expanded={open}
      >
        {citation.index}
      </button>

      {open && (
        <span
          // 宽度用 px 而不是 `w-72`：`html{font-size:14px}` 让 rem 缩小，
          // `w-72` 实际只有 252px，引用原文被挤成 5~6 字一行。
          className={`absolute left-0 z-20 block w-[340px]
                      rounded-md border border-ink-200 bg-white p-3 text-left
                      shadow-[0_2px_8px_rgba(0,0,0,0.08)]
                      ${dropDown ? "top-full mt-1.5" : "bottom-full mb-1.5"}`}
          role="dialog"
          aria-label="引用详情"
        >
          {/* 文档名 + 页码：让用户立刻知道"这是哪份文件的哪一页" */}
          <span className="flex items-start gap-1.5">
            <FileText size={13} className="mt-0.5 shrink-0 text-ink-400" aria-hidden />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[12px] font-medium text-ink-900">
                {citation.document_title}
              </span>
              <span className="meta">
                {citation.page_no != null ? `第 ${citation.page_no} 页` : "无页码"}
                {citation.section_path && citation.section_path.length > 0 && (
                  <> · {citation.section_path.join(" › ")}</>
                )}
              </span>
            </span>
          </span>

          {/*
           * 原文片段：引用最核心的信息。
           *
           * `whitespace-pre-wrap` 保留 chunk 里的换行——原文是多行条文时，
           * 压成一行会让"第 1 条 / 第 2 条"粘在一起难以阅读。
           */}
          <span className="mt-2 block max-h-32 overflow-y-auto border-l-2 border-ink-200
                           whitespace-pre-wrap break-words pl-2 text-[12px]
                           leading-relaxed text-ink-700">
            {plainQuote(citation.quote)}
          </span>

          {onLocate && hasPosition(citation) && (
            <button
              type="button"
              className="mt-2 inline-flex items-center gap-1 text-[12px] text-accent-600
                         hover:underline underline-offset-2"
              onClick={() => onLocate(citation)}
            >
              <ExternalLink size={11} aria-hidden />
              定位原文
            </button>
          )}
        </span>
      )}
    </span>
  );
}

/**
 * 一组引用徽章（回答末尾的"来源"行）。
 *
 * 持有 `pinnedIndex`：**点击固定只能有一个**。理由见文件头的说明。
 * 悬停状态各卡片自持，不需要上提。
 */
export function CitationList({
  citations,
  onLocate,
}: {
  citations: Citation[];
  onLocate?: (citation: Citation) => void;
}) {
  const [pinnedIndex, setPinnedIndex] = useState<number | null>(null);

  return (
    <div className="mt-3 flex flex-wrap items-center gap-1.5">
      <span className="meta">来源</span>
      {citations.map((citation) => (
        <CitationCard
          key={citation.index}
          citation={citation}
          isPinned={pinnedIndex === citation.index}
          onPin={setPinnedIndex}
          {...(onLocate ? { onLocate } : {})}
        />
      ))}
    </div>
  );
}
