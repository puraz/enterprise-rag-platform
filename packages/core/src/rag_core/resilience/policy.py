"""韧性策略：超时、重试、熔断的参数与默认值。

所有参数都有安全默认值——**不传参数也必须不会造成事故**。
这一点很重要：如果默认 max_attempts=1，大家会嫌"没重试"而随手改成 10；
默认值应该"够用且安全"，调整才是有意识的行为。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetryPolicy(BaseModel):
    """重试策略。

    退避算法选 **full jitter**（AWS 推荐）：
        delay = random(0, min(cap, base * 2**attempt))

    为什么不用固定间隔：假设 1000 个客户端同时遇到供应商 5xx，固定 1s
    后它们会**同时**重试，第二波又把供应商打挂——这叫重试风暴。
    full jitter 把重试时间摊平到整个区间，峰值压力降到 1/N。

    为什么不用纯指数退避（无抖动）：多个实例启动时间接近时，
    退避时刻也会同步，同样形成尖峰。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 总尝试次数（含首次）。3 次是很常见的取值：能扛住偶发抖动，
    # 又不会在真故障时把请求拖到用户失去耐心。
    max_attempts: int = Field(default=3, ge=1, le=10)

    base_delay_s: float = Field(default=0.2, gt=0)
    max_delay_s: float = Field(default=5.0, gt=0)
    # 指数退避的倍数
    multiplier: float = Field(default=2.0, ge=1.0)
    jitter: bool = True

    @model_validator(mode="after")
    def _check_bounds(self) -> RetryPolicy:
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("max_delay_s 不能小于 base_delay_s")
        return self


class BreakerPolicy(BaseModel):
    """熔断器策略。

    三态模型（与 Resilience4j / Hystrix 一致）：

        CLOSED ──失败率超阈值──▶ OPEN
          ▲                       │
          │                    冷却结束
       成功                    ▼
          └──── HALF_OPEN ◀──────┘

    - CLOSED：正常放行，统计失败率；
    - OPEN：直接拒绝，**不发起真实调用**（这才是熔断的价值：给下游喘息）；
    - HALF_OPEN：放少量试探请求，成功则恢复，失败则继续熔断。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 统计窗口：只看最近这段时间的成败，不累积历史
    window_s: float = Field(default=30.0, gt=0)
    # 窗口内至少这么多次调用才评估失败率。
    # 没有下限的话，1 次失败就是 100% 失败率，会把偶发抖动误判成故障。
    minimum_calls: int = Field(default=10, ge=1)
    # 失败率达到该值就打开熔断
    failure_rate_threshold: float = Field(default=0.5, gt=0, le=1)
    # 打开后多久进入半开
    open_timeout_s: float = Field(default=30.0, gt=0)
    # 半开状态最多放行几个试探请求
    half_open_max_calls: int = Field(default=3, ge=1)


class ResiliencePolicy(BaseModel):
    """一次调用的完整韧性策略。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 单次尝试的超时
    attempt_timeout_s: float = Field(default=30.0, gt=0)
    # 整个调用（含所有重试与退避等待）的总预算。
    # 没有这个，3 次 × 30s 超时 = 90s，用户早就关页面了，
    # 而且连接和信号量被占住 90s，放大故障。
    total_timeout_s: float = Field(default=60.0, gt=0)
    # 每个依赖同时允许的在途请求数
    max_concurrency: int = Field(default=10, ge=1)

    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    breaker: BreakerPolicy = Field(default_factory=BreakerPolicy)

    @model_validator(mode="after")
    def _total_covers_attempt(self) -> ResiliencePolicy:
        if self.total_timeout_s < self.attempt_timeout_s:
            raise ValueError("total_timeout_s 不能小于 attempt_timeout_s")
        return self
