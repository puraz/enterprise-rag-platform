/**
 * 前端侧的契约补充类型。
 *
 * ## 为什么需要这个文件
 *
 * 后端契约里有 `Page[T]`（Pydantic 泛型），但**泛型无法直接导出成
 * TS 类型**（生成器只导出具体模型）。后端用 `Page[MessageItem]` 这类
 * 具体化形式作为响应模型，OpenAPI 里会展开成 `{items, meta}`。
 *
 * 所以前端自己声明一个泛型包装，与后端 `Page[T]` 的形状严格一致
 * （`items` + `meta`，见 AGENTS.md 4.2）。
 *
 * **这个类型必须与后端保持一致**：`tests/test_success_contract.py`
 * 检查后端形状，这里靠人工同步。若后端改了分页形状，需同步改这里。
 */

import type { PageMeta } from "@contracts";

export interface Page<T> {
  items: T[];
  meta: PageMeta;
}
