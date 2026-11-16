/**
 * Vite 的环境类型声明。
 *
 * `vite/client` 提供 `import.meta.env` 的类型（`DEV` / `PROD` / `MODE`）。
 * 不引它时 `import.meta.env.DEV` 会报 "Property 'env' does not exist"。
 *
 * 这也是**为什么 dev 登录表单能在生产构建中被移除**：
 * Vite 在 build 时把 `import.meta.env.DEV` 替换成字面量 `false`，
 * 打包器（Rollup）随后会把整个 `if (false) { ... }` 分支 tree-shake 掉。
 * 所以生产 bundle 里**不存在**那段表单代码，不是靠运行时判断。
 */
/// <reference types="vite/client" />
