"""追踪配置（Langfuse）。

## 为什么配置里必须有"启用"开关，而不是"装了 SDK 就发数据"

追踪是**旁路**：它会产生网络请求、存储成本、并可能把业务数据带出进程。
默认关闭、显式开启，才符合"数据不能因为某个库的默认行为而外流"这条底线。

## 为什么 host 默认不是 3000

血泪教训：初版默认 `http://localhost:3000`，而本地 Grafana 正好占 3000。
于是 span 被 POST 到 Grafana 的 `/api/public/otel/v1/traces`——
**Grafana 返回 404，SDK 后台重试后丢弃，业务完全无感**，
表现就是"Langfuse 里没有数据，但也没人报错"。

所以：
- 默认值改成 Langfuse 自托管的约定端口 3030；
- `host` 必须显式写协议（`http://` 或 `https://`），启动时校验；
- 启动时用 `auth_check` 主动探测一次，把"发错地方"变成一条明确的 warning。

## 采样率

`sample_rate < 1` 时 SDK 只导出部分 trace。**被丢弃的 trace 仍然有 trace_id**
（只是不上报），所以 `model_calls.trace_id` 与日志关联不会断——
`null` 表示"没有追踪"，不等于"采样掉了"，两者不要混用。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

#: Langfuse 自托管的默认端口。
#:
#: 刻意**不用** 3000：那是 Grafana 的默认端口，撞了以后追踪静默失效。
DEFAULT_LANGFUSE_HOST = "http://localhost:3030"


class TracingSettings(BaseModel):
    """追踪（Langfuse）配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    enabled: bool = False
    """是否导出 trace。

    关闭时仍然构造 `LocalTracer`（内存记录），
    这样"链路上有哪些 span、父子关系对不对"在本地也能验证。
    """

    provider: Literal["langfuse"] = "langfuse"
    """追踪后端。目前只有 Langfuse；保留字段是为了将来换 Jaeger/Tempo 时不改配置名。"""

    public_key: SecretStr | None = None
    """Langfuse 项目公钥（`pk-lf-...`）。"""

    secret_key: SecretStr | None = None
    """Langfuse 项目私钥（`sk-lf-...`）。**只走环境变量，不进日志/库**。"""

    host: str = DEFAULT_LANGFUSE_HOST
    """Langfuse 服务地址，必须带协议。"""

    environment: str | None = None
    """环境标签（`local` / `staging` / `prod`），Langfuse 里可按它过滤。

    默认 `None` 时由装配方传入 `settings.env`，避免两处各写一遍。
    """

    release: str | None = None
    """版本标签，便于回答"这次回归是从哪个版本开始的"。"""

    sample_rate: float = Field(default=1.0, gt=0.0, le=1.0)
    """采样率。本地/预发 1.0，高 QPS 生产按成本要求下调。"""

    flush_at: int = Field(default=64, ge=1, le=10_000)
    """攒够多少 span 批量上报一次（减少 HTTP 往返）。"""

    flush_interval_s: float = Field(default=2.0, gt=0.0, le=60.0)
    """最长多久上报一次（保证低流量下也能及时看到数据）。"""

    timeout_s: int = Field(default=10, ge=1, le=120)
    """单次上报的 HTTP 超时。"""

    @field_validator("host")
    @classmethod
    def _validate_host(cls, value: str) -> str:
        """host 必须是带协议的绝对地址。

        `localhost:3030` 这种漏写协议的写法，httpx 会当成相对路径，
        表现为"每次都连接失败但不影响业务"——又一种静默失效。
        """
        host = value.strip().rstrip("/")
        if not host.startswith(("http://", "https://")):
            raise ValueError(f"tracing.host 必须带协议（http:// 或 https://），当前是 {value!r}")
        return host

    @property
    def configured(self) -> bool:
        """是否具备真正导出 trace 的条件。"""
        return self.enabled and bool(self.public_key) and bool(self.secret_key)

    @property
    def redacted_summary(self) -> dict[str, object]:
        """可安全打印的配置摘要（绝不包含密钥原文）。"""
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "host": self.host,
            "environment": self.environment,
            "release": self.release,
            "sample_rate": self.sample_rate,
            "has_public_key": bool(self.public_key),
            "has_secret_key": bool(self.secret_key),
        }


__all__ = ["DEFAULT_LANGFUSE_HOST", "TracingSettings"]
