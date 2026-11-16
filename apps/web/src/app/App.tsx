/**
 * 路由表。
 *
 * ## 为什么用 React Router 的 data mode（`createBrowserRouter`）
 *
 * 而不是 `<BrowserRouter>` + `<Routes>`：
 * - 路由级代码分割只需要在 route 定义里写 `lazy`；
 * - 认证守卫可以用 `loader`（在渲染前决定跳转，避免"闪一下未登录页面"）；
 * - 与 TanStack Query 配合自然（loader 里可以 `ensureQueryData`）。
 *
 * ## 权限分两层
 *
 * - **路由级**：`RequireAuth` 组件（未登录跳 /login）；
 * - **按钮级**：`can()` 工具（见 `lib/permissions.ts`，Phase B5 补）。
 */

import { createBrowserRouter, Navigate, Outlet } from "react-router-dom";
import { useAuthStore } from "@/features/auth/useAuth";
import { LoginPage } from "@/features/auth/LoginPage";
import { ChatPage } from "@/features/chat/ChatPage";

function RequireAuth() {
  const hasToken = useAuthStore((state) => state.hasToken);
  if (!hasToken) return <Navigate to="/login" replace />;
  return <Outlet />;
}

export const router = createBrowserRouter([
  { path: "/login", element: <LoginPage /> },
  {
    element: <RequireAuth />,
    children: [
      { index: true, element: <Navigate to="/chat" replace /> },
      { path: "/chat", element: <ChatPage /> },
      { path: "/knowledge", element: <Placeholder title="知识库" /> },
      { path: "/profile", element: <Placeholder title="我的" /> },
    ],
  },
  { path: "*", element: <Navigate to="/chat" replace /> },
]);

function Placeholder({ title }: { title: string }) {
  return (
    <div className="flex h-full items-center justify-center">
      <p className="text-[13px] text-ink-400">{title}（Phase B 后续实现）</p>
    </div>
  );
}
