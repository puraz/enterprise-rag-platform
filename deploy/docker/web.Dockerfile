# 前端镜像：Vite 构建 → Nginx 托管静态文件 + 反代 API。
#
# ## 为什么同源反代而不是让前端直连 API 域名
#
# 1. 浏览器认为前后端同源 → 不需要 CORS 预检（SSE 自定义头会触发 preflight）；
# 2. refresh token 走 HttpOnly/SameSite cookie 时同源最简单；
# 3. 生产可以只暴露 80/443 一个端口。
#
# ## SSE 的关键配置
#
# Nginx 默认 `proxy_buffering on`，会把 token 攒成一批再发——
# 表现为“用户等 5 秒，然后整段答案一次性出现”。
# 因此必须 `proxy_buffering off` + 关闭 cache + 长 read timeout。
# 应用侧也返回了 `X-Accel-Buffering: no`，两者是双保险。

# syntax=docker/dockerfile:1

FROM node:22-bookworm-slim AS build

# Vite 配置里的 `@contracts` 指向 ../../packages/contracts/generated，
# 所以构建目录必须保留仓库的相对结构。
WORKDIR /repo

RUN corepack enable && corepack prepare pnpm@9.15.0 --activate

COPY apps/web/package.json apps/web/pnpm-lock.yaml ./apps/web/
RUN cd apps/web && pnpm install --frozen-lockfile

COPY apps/web ./apps/web
COPY packages/contracts/generated ./packages/contracts/generated

RUN cd apps/web && pnpm build

# ---------------------------------------------------------------------------

FROM nginx:1.27-alpine

COPY deploy/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /repo/apps/web/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
