import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "react-router-dom";
import { router } from "./app/App";
import "./app/styles.css";

/**
 * TanStack Query 客户端。
 *
 * ## 默认值的三个判断
 *
 * - `staleTime: 30s`：内部工具数据变化不快，30 秒内不重复拉取。
 *   设 0（每次都拉）在切页时会有明显的加载闪烁。
 * - `retry: 1`：只重试一次。默认 3 次会让"服务挂掉"时用户等 7 秒才看到错误。
 * - `refetchOnWindowFocus: false`：切回标签页不自动刷新。
 *   对话页有这个需求的话单独开（否则正在读的内容会被刷掉）。
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("找不到 #root 挂载点");

createRoot(rootElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
