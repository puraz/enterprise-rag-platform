/**
 * 应用骨架：左侧栏 + 主内容区。
 *
 * ## 布局判断
 *
 * 侧栏固定 240px（不是可拖拽宽度）：
 * - 企业工具的侧栏内容（导航 + 会话列表）宽度需求稳定；
 * - 可拖拽宽度需要持久化 + 边界处理 + 移动端退化，收益很低。
 *
 * 移动端（<768px）：侧栏收起为抽屉，由顶栏按钮触发。
 *
 * ## 为什么侧栏用深色而不是白色
 *
 * 灰色侧栏 + 白色内容区是常见做法，但**深色侧栏**在这里更好：
 * - 对话区会长时间注视，深色侧栏能降低周边亮度，减少视觉疲劳；
 * - 会话列表是"导航"性质，深色天然弱化它的视觉权重。
 *
 * 但不用纯黑（#000）：`oklch(0.24 ...)` 的墨色比纯黑柔和，
 * 长时间看不会觉得刺眼。
 */

import { useState, type ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  MessageSquare,
  Library,
  User,
  Plus,
  Menu,
  X,
  LogOut,
} from "lucide-react";
import { useLogout } from "@/features/auth/useAuth";

export interface AppShellProps {
  children: ReactNode;
  sidebar: ReactNode;
  userEmail?: string | undefined;
}

export function AppShell({ children, sidebar, userEmail }: AppShellProps) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const logout = useLogout();
  const navigate = useNavigate();

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      {/* 移动端遮罩 */}
      {drawerOpen && (
        <button
          type="button"
          aria-label="关闭菜单"
          className="fixed inset-0 z-30 bg-ink-950/40 md:hidden"
          onClick={() => setDrawerOpen(false)}
        />
      )}

      {/* 侧栏 */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-60 shrink-0 flex-col
                    bg-ink-950 text-ink-200 transition-transform duration-200
                    md:static md:translate-x-0
                    ${drawerOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="flex h-12 items-center justify-between px-3">
          <span className="text-[13px] font-medium tracking-wide text-white">
            知识助手
          </span>
          <button
            type="button"
            aria-label="关闭菜单"
            className="text-ink-400 hover:text-white md:hidden"
            onClick={() => setDrawerOpen(false)}
          >
            <X size={16} />
          </button>
        </div>

        {/* 主导航 */}
        <nav className="px-2">
          <SidebarLink to="/chat" icon={<MessageSquare size={15} />} label="对话" />
          <SidebarLink to="/knowledge" icon={<Library size={15} />} label="知识库" />
          <SidebarLink to="/profile" icon={<User size={15} />} label="我的" />
        </nav>

        <div className="mx-3 my-2 border-t border-ink-800" />

        {/* 会话/文档列表（由页面注入） */}
        <div className="min-h-0 flex-1 overflow-y-auto px-2">{sidebar}</div>

        {/* 底部：新建 + 用户 */}
        <div className="border-t border-ink-800 p-2">
          <button
            type="button"
            className="flex w-full items-center gap-2 rounded-md px-2 h-8
                       text-[13px] text-ink-300 hover:bg-ink-900 hover:text-white
                       transition-colors duration-100"
            onClick={() => {
              navigate("/chat");
              setDrawerOpen(false);
            }}
          >
            <Plus size={15} aria-hidden />
            新建对话
          </button>

          <div className="mt-1 flex items-center gap-2 rounded-md px-2 py-1.5">
            <span className="min-w-0 flex-1 truncate text-[12px] text-ink-400">
              {userEmail ?? "—"}
            </span>
            <button
              type="button"
              aria-label="退出登录"
              title="退出登录"
              className="text-ink-500 hover:text-white transition-colors"
              onClick={() => {
                logout();
                navigate("/login");
              }}
            >
              <LogOut size={14} />
            </button>
          </div>
        </div>
      </aside>

      {/* 主内容 */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* 移动端顶栏 */}
        <header className="flex h-12 items-center gap-2 border-b border-ink-200 px-3 md:hidden">
          <button
            type="button"
            aria-label="打开菜单"
            className="text-ink-700"
            onClick={() => setDrawerOpen(true)}
          >
            <Menu size={18} />
          </button>
          <span className="text-[13px] font-medium">知识助手</span>
        </header>

        <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
      </div>
    </div>
  );
}

function SidebarLink({
  to,
  icon,
  label,
}: {
  to: string;
  icon: ReactNode;
  label: string;
}) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-2 rounded-md px-2 h-8 text-[13px]
         transition-colors duration-100
         ${
           isActive
             ? "bg-ink-900 text-white"
             : "text-ink-300 hover:bg-ink-900 hover:text-white"
         }`
      }
    >
      {icon}
      {label}
    </NavLink>
  );
}
