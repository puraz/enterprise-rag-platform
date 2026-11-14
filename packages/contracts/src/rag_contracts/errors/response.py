"""统一错误响应体：四端共享的对外契约。

这是**唯一**允许出现在 HTTP 错误响应里的结构。
任何其它形状（FastAPI 默认的 `{"detail": ...}`、裸字符串、堆栈）都算契约违规。

字段设计取舍：
- `code` 给程序分支，`message` 给人看，两者故意分离 —— 同一 code 的
  文案可以随语言/场景变，但 code 不能变。
- `details` 只放**可公开**的结构化补充（哪个字段错了、哪个配额超了）。
  内部异常类型、SQL、文件路径一律不进这里。
- `request_id` 必须回给前端：用户报障时把它贴给客服，就能在日志里
  定位到那一次请求的全部链路。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rag_contracts.errors.codes import ErrorCode


class FieldError(BaseModel):
    """字段级校验错误。供前端把错误标到具体输入框上。"""

    model_config = ConfigDict(frozen=True)

    # loc 是路径片段，例如 ["body", "items", 0, "name"]
    location: list[str | int] = Field(default_factory=list, description="错误字段位置路径")
    message: str = Field(description="给用户看的中文提示")
    # 内部错误类型名（如 value_error / string_too_short），供前端做交互细化
    type: str = Field(description="校验器错误类型标识")


class ErrorResponse(BaseModel):
    """统一错误响应体。所有非 2xx 响应都必须是这个形状。"""

    model_config = ConfigDict(frozen=True)

    code: ErrorCode = Field(description="业务错误码，前端按此分支处理")
    message: str = Field(description="面向用户的中文错误信息")
    request_id: str = Field(description="请求追踪 ID，报障时提供此值")
    # 仅在校验失败时非空
    fields: list[FieldError] = Field(default_factory=list, description="字段级错误明细")
    # 可公开的额外上下文，例如 {"retry_after_s": 30, "limit": 100}
    details: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="可公开的结构化补充信息"
    )
