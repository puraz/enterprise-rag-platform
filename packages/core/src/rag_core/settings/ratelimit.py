"""限流配置。

## 为什么限额要可配置

问答一次要调 3~4 次 LLM，成本是普通接口的几十倍。不同部署的
模型单价、并发规模、租户数量差别很大，把 20 QPS / 2 QPS 写死在代码里，
要么小规模部署被限得过宽松（保护不住后端），要么大规模部署被限得
过死（正常用户被误伤）。

## 为什么租户限额与用户限额分开

- 只限租户：一个用户能把整个租户的额度吃光（其它同事全被限）；
- 只限用户：大量小号注册能把全平台打爆。

## 为什么默认 fail-open

Redis 不可用时，拒绝所有请求会让全站不可用（比被刷更糟）。
`enforce_chat_ratelimit` 会记 warning 并放行，保证服务可用性。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatRateLimitSettings(BaseModel):
    """问答接口的双层限流配额。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    enabled: bool = Field(
        default=True,
        description="是否启用问答限流。关闭仅用于本机调试或压测基线",
    )
    tenant_rate: float = Field(
        default=20.0,
        gt=0,
        description="租户级长期平均速率（请求/秒）",
    )
    tenant_burst: int = Field(
        default=40,
        ge=1,
        description="租户级瞬时突发量（桶容量）",
    )
    user_rate: float = Field(
        default=2.0,
        gt=0,
        description="用户级长期平均速率（请求/秒）",
    )
    user_burst: int = Field(
        default=5,
        ge=1,
        description="用户级瞬时突发量（桶容量）",
    )


class RateLimitSettings(BaseModel):
    """限流域配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    chat: ChatRateLimitSettings = Field(default_factory=ChatRateLimitSettings)


__all__ = ["ChatRateLimitSettings", "RateLimitSettings"]
