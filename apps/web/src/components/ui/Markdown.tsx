/**
 * Markdown 渲染（回答正文）。
 *
 * ## 安全：必须 sanitize
 *
 * 回答内容来自 LLM，而 LLM 的输入包含**用户文档与用户问题**——
 * 这意味着攻击者可以通过上传含恶意 markdown 的文档，让模型复述出来，
 * 最终在别人浏览器里执行脚本（存储型 XSS）。
 *
 * 所以：
 * 1. `rehype-sanitize` 用默认 schema（禁 script/on* / javascript: URL）；
 * 2. **不用 `dangerouslySetInnerHTML`**，react-markdown 产出的是 React 元素。
 *
 * ## 样式：为什么手写而不用 @tailwindcss/typography
 *
 * typography 插件给的是"博客文章"风格（大标题、宽行距、居中图片），
 * 在对话气泡里会显得过于松垮。这里的密度要更紧凑：
 * 段落间距 0.75em（而非 1.5em），标题只比正文大 1~2px。
 */

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";

export interface MarkdownProps {
  content: string;
  /** 流式生成中：末尾加闪烁光标。 */
  streaming?: boolean;
}

export function Markdown({ content, streaming = false }: MarkdownProps) {
  return (
    <div
      className={[
        "text-[14px] leading-[1.65] text-ink-900",
        // 紧凑的排版：对话里段落间距不能像文章那样大
        "[&_p]:my-0 [&_p+p]:mt-3",
        "[&_ul]:my-2 [&_ul]:pl-5 [&_ol]:my-2 [&_ol]:pl-5 [&_li]:my-0.5",
        "[&_h1]:text-[15px] [&_h1]:font-semibold [&_h1]:mt-4 [&_h1]:mb-2",
        "[&_h2]:text-[14px] [&_h2]:font-semibold [&_h2]:mt-3 [&_h2]:mb-1.5",
        "[&_h3]:text-[14px] [&_h3]:font-medium [&_h3]:mt-3 [&_h3]:mb-1",
        // 行内代码：淡底 + 细边框，不用彩色
        "[&_code]:rounded-sm [&_code]:bg-ink-100 [&_code]:px-1 [&_code]:py-0.5",
        "[&_code]:text-[13px] [&_code]:font-mono",
        // 代码块：深色底，与正文明确区分
        "[&_pre]:my-3 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-ink-950",
        "[&_pre]:p-3 [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-ink-100",
        // 表格：细边框，表头淡底
        "[&_table]:my-3 [&_table]:w-full [&_table]:border-collapse [&_table]:text-[13px]",
        "[&_th]:border [&_th]:border-ink-200 [&_th]:bg-ink-50 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_th]:font-medium",
        "[&_td]:border [&_td]:border-ink-200 [&_td]:px-2 [&_td]:py-1",
        "[&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-ink-300 [&_blockquote]:pl-3 [&_blockquote]:text-ink-700",
        "[&_a]:text-accent-600 [&_a]:underline [&_a]:underline-offset-2",
        "[&_hr]:my-4 [&_hr]:border-ink-200",
        streaming ? "streaming-caret" : "",
      ].join(" ")}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
