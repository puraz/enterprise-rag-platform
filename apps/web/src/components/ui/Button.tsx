/**
 * 按钮：三种视觉层级 + 加载态。
 *
 * ## 为什么不用 shadcn/CLI 生成的版本
 *
 * shadcn 生成的是"可复制的一坨代码"，包含 cva、Radix Slot、asChild 等
 * 一套抽象。对本项目：
 * - 只有 4 种变体，不需要 cva 的组合能力；
 * - 不需要 `asChild`（我们不做"按钮式链接"，直接用 Link 包按钮）；
 * - 生成的代码有大量 `focus-visible:ring-2 ring-offset-2` + 多重阴影，
 *   正是"AI 味"的来源之一。
 *
 * 手写 40 行，样式集中在 `.btn` 类（见 styles.css），改一处全局生效。
 */

import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Loader2 } from "lucide-react";

type Variant = "primary" | "secondary" | "ghost" | "danger";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  /** 加载中：显示 spinner 并禁用（**不改变宽度**，避免布局跳动）。 */
  loading?: boolean;
  icon?: ReactNode;
}

/**
 * 每个变体是**自足的单个类**（形状 + 配色都在里面）。
 *
 * 原因见 styles.css 的说明：Tailwind v4 的 `@apply` 不能引用自定义类，
 * 所以用选择器分组把形状写进每个变体，而不是让调用方记得挂两个 class。
 */
const VARIANT_CLASS: Record<Variant, string> = {
  primary: "btn-primary",
  secondary: "btn-secondary",
  ghost: "btn-ghost",
  danger: "btn-danger",
};

export function Button({
  variant = "secondary",
  loading = false,
  icon,
  children,
  className = "",
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      type="button"
      className={`${VARIANT_CLASS[variant]} ${className}`}
      disabled={disabled || loading}
      // aria-busy 让屏幕阅读器知道"正在处理"，而不是只有一个无标签的 spinner
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? (
        <Loader2 size={14} className="animate-spin" aria-hidden />
      ) : (
        icon
      )}
      {children}
    </button>
  );
}
