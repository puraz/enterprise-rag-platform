"""统一错误响应体：四端共享的对外契约。

这是**唯一**允许出现在 HTTP 错误响应里的结构。
任何其它形状（FastAPI 默认的 `{"detail": ...}`、裸字符串、堆栈）都算契约违规。

## 字段设计

| 字段 | 给谁用 |
|---|---|
| `code`（int） | 前端 `switch` 分支、监控告警规则、按号段批量判断 |
| `code_name`（str） | 日志/审计/排障（`10101` 不可读，`AUTH_TOKEN_EXPIRED` 一眼懂） |
| `message`（str） | 展示给用户的中文文案 |
| `request_id` | 用户报障时贴给客服，用来关联链路日志 |
| `fields` | 字段级校验错误，前端标到具体输入框 |
| `details` | 可公开的结构化补充（如 `retry_after_s`、配额上限） |

**`details` 只放可公开信息**。内部异常类型、SQL、文件路径一律不进这里——
它们会通过响应回到不可信的客户端。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rag_contracts.errors.codes import ErrorCode, code_name


class FieldError(BaseModel):
    """字段级校验错误。供前端把错误标到具体输入框上。"""

    model_config = ConfigDict(frozen=True)

    # loc 是路径片段，例如 ["body", "items", 0, "name"]
    location: list[str | int] = Field(default_factory=list, description="错误字段位置路径")
    message: str = Field(description="给用户看的中文提示")
    type: str = Field(description="校验器错误类型标识")


class ErrorResponse(BaseModel):
    """统一错误响应体。所有非 2xx 响应都必须是这个形状。"""

    model_config = ConfigDict(frozen=True)

    code: int = Field(
        description=(
            "业务错误码（数字）。按域号段编码：0xxxx 通用、1xxxx 认证、"
            "2xxxx 授权、3xxxx 文档、4xxxx 检索、5xxxx 上传、"
            "6xxxx 任务、7xxxx 配额、8xxxx 供应商、9xxxx 内部。"
            "前端按此分支处理，**一旦发布不可变更语义**"
        ),
        examples=[10101],
    )
    code_name: str = Field(
        description="错误码的语义名，便于日志检索与排障（如 AUTH_TOKEN_EXPIRED）",
        examples=["AUTH_TOKEN_EXPIRED"],
    )
    message: str = Field(description="面向用户的中文错误信息")
    request_id: str = Field(description="请求追踪 ID，报障时提供此值")
    # 仅在校验失败时非空
    fields: list[FieldError] = Field(default_factory=list, description="字段级错误明细")
    # 可公开的额外上下文，例如 {"retry_after_s": 30, "limit": 100}
    details: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="可公开的结构化补充信息"
    )

    @classmethod
    def build(
        cls,
        *,
        code: ErrorCode,
        message: str,
        request_id: str,
        fields: list[FieldError] | None = None,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> ErrorResponse:
        """构造响应，自动填充 `code_name`。

        用这个工厂而不是直接 `ErrorResponse(...)`：避免某个调用点
        忘填 `code_name`（那样日志里就只有数字码，排障困难）。
        """
        return cls(
            code=int(code),
            code_name=code_name(int(code)),
            message=message,
            request_id=request_id,
            fields=fields or [],
            details=details or {},
        )
