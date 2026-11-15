"""成功响应规范：三类形态 + 命名规则 + 状态码。

## 完整裁决（一次说清，不再反复）

| 问题 | 答案 | 理由 |
|---|---|---|
| 成功时有 `code` 吗 | **没有** | `code` 是"失败原因"。成功无原因。HTTP 2xx 就是成功信号。 |
| 成功要包 `data` 吗 | **不要** | 只在"成败都用 200"时才有意义；用状态码区分时是纯噪音 |
| 删除返回什么 | **204 + 空 body** | HTTP 语义的一部分。包 `{data: null}` 反而违反语义。 |
| 创建返回什么 | **201 + 对象 + Location 头** | 前端知道新资源地址，无需二次查询 |

## 三类出现形态（只有这三种）

### ① 单个资源对象

```json
GET /api/v1/documents/{id}  → 200
{"id": "...", "title": "...", "created_at": "..."}

POST /api/v1/documents  → 201 + Location: /api/v1/documents/{id}
{"id": "...", "title": "..."}
```

### ② 分页集合

```json
GET /api/v1/documents?limit=20&offset=0  → 200
{"items": [...], "meta": {"limit": 20, "offset": 0, "total": null, "has_more": true}}
```

见 `pagination.py`。

### ③ 无内容

```json
DELETE /api/v1/documents/{id}  → 204
(空 body)
```

## 命名规则（前端好对接的实际抓手）

规则比"结构"更重要。50 个接口形态一致但字段名各异，前端一样要写 50 套。

| 规则 | 正确 | 错误 |
|---|---|---|
| 字段一律 `snake_case` | `created_at` | `createdAt` / `Created-At` |
| 主键一律 `id` | `{"id": "..."}` | `{"document_id":...}` / `{"docId":...}` |
| 时间 `*_at` + ISO 8601 带时区 | `"2026-11-15T05:17:08Z"` | 时间戳毫秒 / 无时区 |
| 列表一律 `items` | `{"items": [...]}` | `data` / `list` / `results` |
| 布尔一律 `is_*` / `has_*` / `can_*` | `is_current`, `has_more` | `current`, `more` |
| 数量一律 `*_count`（整数） | `chunk_count: 42` | `chunks: 42`（与对象列表混淆） |
| 枚举值一律小写 snake_case | `"in_progress"` | `"InProgress"` / `"IN_PROGRESS"` |
| 分页一律 `Page[T]` | 见 pagination.py | 自造 `{total, page, list}` |

**为什么枚举值要小写**：前端大量用字符串比较与 `switch`，
大小写不一致导致的 bug（`"Active" !== "active"`）排查成本极高，
且 URL query 参数大小写敏感。

## 什么时候用 201 / 202 / 204

| 场景 | 状态码 | 理由 |
|---|---|---|
| POST 创建资源并立即返回 | 201 + `Location` | 前端可跳转到新资源 |
| POST 触发异步任务 | **202** | 资源尚未创建完成，202 表示"已接受" |
| DELETE 成功 | **204** | 无内容可返回 |
| PATCH/PUT 成功 | 200 + 更新后对象 | 前端无需再 GET 一次 |

**202 的一个重要细节**：异步接口必须返回 `task_id`，
前端用它轮询进度。没有 `task_id` 的 202 是死路——前端不知道等什么。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskAccepted(BaseModel):
    """异步任务已接受的响应（对应 HTTP 202）。

    用于"提交后立即返回、后台处理"的接口：文档解析、索引重建、批量操作。

    **必须带 `task_id`**：前端靠它轮询 `GET /api/v1/tasks/{id}` 拿进度。
    没有 task_id 的 202 等于告诉前端"等着吧"，但它不知道等什么。
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(description="任务 ID，用于轮询进度")
    status: str = Field(default="pending", description="任务初始状态")
    poll_url: str | None = Field(
        default=None,
        description="建议的轮询地址（绝对路径），前端可直接 GET",
    )


class CreatedResource(BaseModel):
    """创建成功的响应基类（对应 HTTP 201）。

    子类声明具体资源字段；本类只约束"必须能被定位"。

    用法：

        class DocumentCreated(CreatedResource):
            title: str
            format: str
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="新资源的 ID")


# ---------------------------------------------------------------- 校验


def validate_snake_case_field(name: str) -> bool:
    """检查字段名是否符合 snake_case。

    用于测试护栏：Pydantic 模型里出现 camelCase 字段时报警。
    后端如果输出 camelCase，前端 TS 类型也是 camelCase——
    看起来"能用"，但与 SQL 列名、日志字段全都不一致，跨层对照时会疯。
    """
    if not name:
        return False
    return name.islower() and all(char.isalnum() or char == "_" for char in name)


def validate_time_field_value(value: Any) -> bool:
    """检查时间字段是否为 ISO 8601 带时区字符串。

    不带时区的时间在跨时区场景下会错位（前端显示差 8 小时）。
    """
    if not isinstance(value, str):
        return False
    # 至少形如 2026-11-15T05:17:08Z 或 2026-11-15T05:17:08+08:00
    return "T" in value and (value.endswith("Z") or "+" in value[-6:] or "-" in value[-6:])
