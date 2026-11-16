import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

/**
 * Vite 配置。
 *
 * ## 为什么 dev 用 proxy 而不是直连 8000
 *
 * 前端跑在 5173，API 在 8000。直连会遇到两个问题：
 * 1. CORS 预检（尤其 SSE 的自定义头会触发 preflight）；
 * 2. cookie/同源策略在本地开发时不直观。
 *
 * proxy 让浏览器认为"前后端同源"，生产环境用 Nginx 做同样的事。
 *
 * ## 为什么 SSE 需要特殊配置
 *
 * `ws: true` 不适用于 SSE（那是 WebSocket）。SSE 需要的是：
 * - 不缓冲（`selfHandleResponse: false` 让 Vite 直接透传流）；
 * - 不设超时（Vite 默认无超时，但某些代理中间件会加）。
 * Vite 5+ 的 http-proxy 对 SSE 已经默认透传，无需额外配置。
 */
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      // 契约类型直接引用后端生成的产物（单一真相源）
      "@contracts": path.resolve(
        __dirname,
        "../../packages/contracts/generated/contracts.ts",
      ),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
