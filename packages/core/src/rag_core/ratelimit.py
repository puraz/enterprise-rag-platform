"""限流器：Redis + Lua 原子令牌桶。

## 为什么必须用 Lua 而不是 Python 逻辑

朴素实现（读-判断-写）在并发下必然失控：

    remaining = redis.get(key)          # 两个请求都读到 10
    if remaining > 0:                   # 两个都判断通过
        redis.decr(key)                 # 实际只该放行 1 个

Redis 单命令是原子的，但"读+判断+写"三步不是。用 `WATCH/MULTI`
能解决，但重试风暴会让高并发下成功率骤降（每次冲突都要重试）。

Lua 脚本在 Redis 内部**整体原子执行**（单线程模型，脚本期间不插其他命令），
且没有网络往返开销。这是限流的正确做法。

## 为什么是令牌桶而不是固定窗口

固定窗口有"边界双倍"问题：窗口末尾和下一窗口开头各放行 N 个，
实际瞬时速率可达 2N。

    窗口 1: [.......|XXXXXXXX]  末尾 1 秒放行 100 个
    窗口 2: [XXXXXXXX|.......]  开头 1 秒又放行 100 个
    → 跨窗口的 2 秒内放行了 200 个

令牌桶按"平滑速率"补充令牌，任意时刻的突发量受桶容量约束，
不存在边界问题。

## 为什么桶容量（burst）与速率分开

- `rate`：长期平均速率（如 10 QPS）；
- `burst`：允许的瞬时突发量（如 20，允许短时 2 倍）。

两者相乘才是"最坏情况"。只配 rate 会让"每秒 1 个"的接口完全无法
应对"用户点了两次"这种正常场景。

## 三层限流的设计

| 维度 | key | 目的 |
|---|---|---|
| 租户 | `rl:t:{tenant_id}` | 防止单个租户打爆全平台 |
| 用户 | `rl:u:{tenant_id}:{user_id}` | 防止单个用户打爆本租户 |
| IP | `rl:ip:{ip}` | 防未认证请求（登录接口） |

**三层都要有**：只限租户时，一个用户能把整个租户的额度吃光；
只限用户时，大量小号注册能把全平台打爆。

## RLS 与 Redis 的关系

Redis 没有 RLS。所以 **key 里必须显式带 tenant_id**——
漏了就是跨租户共享配额（A 租户用光后 B 租户也被限流）。
这是"缓存/限流层的 tenant 隔离"与数据库 RLS 同等重要的一条。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from rag_core.logging import get_logger

log = get_logger(__name__)

# Lua 脚本：原子令牌桶。
#
# KEYS[1] = 桶的 key
# ARGV    = [rate, burst, now_ms, cost, ttl_seconds]
#
# 返回 {allowed(0/1), remaining, retry_after_ms}
#
# ## 脚本里的每一行都有理由
#
# - 用 hash 存 tokens + last_refill：两个值必须一起原子读写。
#   分开两个 key 会让中间状态可见（读到新 tokens 但旧 last_refill）。
# - 时间由**客户端传入**（`now_ms`）而不是脚本里 `redis.call('TIME')`：
#   Redis 的 TIME 在集群各节点可能不一致，且脚本里调用 TIME 会触发
#   复制相关的特殊处理。客户端传入让多实例行为可控（代价是要求
#   客户端时钟大致同步——对限流来说几秒偏差可接受）。
# - 桶满时不累积（`math.min(burst, ...)`）：否则长时间空闲后可以
#   瞬间放出远超 burst 的请求。
# - 返回 retry_after_ms 而不是让调用方猜：精确告诉客户端多久后可重试，
#   比"过一会儿再试"有用得多。
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(data[1])
local last_refill = tonumber(data[2])

if tokens == nil then
  -- 首次访问：满桶
  tokens = burst
  last_refill = now_ms
end

-- 按流逝时间补充令牌
local elapsed = math.max(0, now_ms - last_refill)
local refilled = tokens + (elapsed / 1000.0) * rate
if refilled > burst then
  refilled = burst
end
tokens = refilled

local allowed = 0
local retry_after_ms = 0

if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  -- 需要多少毫秒才能补足差额
  local deficit = cost - tokens
  retry_after_ms = math.ceil((deficit / rate) * 1000)
end

redis.call('HSET', key, 'tokens', tokens, 'last_refill', now_ms)
redis.call('EXPIRE', key, ttl)

return {allowed, math.floor(tokens), retry_after_ms}
"""


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    """一次限流判定的结果。"""

    allowed: bool
    remaining: int
    """剩余令牌数（整数近似）。"""

    retry_after_ms: int = 0
    """被拒时建议的重试间隔（毫秒）。allowed=True 时为 0。"""

    limit_name: str = ""

    @property
    def retry_after_s(self) -> float:
        return self.retry_after_ms / 1000.0


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """一条限流规则。"""

    name: str
    rate: float
    """长期平均速率（每秒允许的请求数）。"""

    burst: int
    """桶容量（允许的瞬时突发量）。"""

    @property
    def ttl_s(self) -> int:
        """桶 key 的存活时间。

        桶在空闲后会"自然回满"，所以不需要永久保存。
        ttl 取"从空到满所需时间 × 2"：够长以保持状态连续，
        够短以免 Redis 被大量不活跃用户的 key 撑满。

        **最少 60 秒**：太短的 ttl 会让桶频繁重建（每次重建都从满桶开始），
        等于凭空给用户加额度。
        """
        refill_time = self.burst / self.rate if self.rate > 0 else 60
        return max(60, int(refill_time * 2))


class RateLimiter:
    """基于 Redis + Lua 的限流器。

    用法：

        limiter = RateLimiter(redis.client)
        result = await limiter.check(
            policy=RateLimitPolicy("tenant_chat", rate=10, burst=20),
            key="t:550e8400-...",
        )
        if not result.allowed:
            raise RateLimitedError(retry_after_s=result.retry_after_s)
    """

    def __init__(self, client: object) -> None:
        """
        Args:
            client: `redis.asyncio.Redis` 实例（`RedisClient.client`）。
                用 object 类型是为了不 import redis（保持 core 的依赖面）。
                运行时需要支持 `eval` 方法与 `hgetall`。
        """
        self._client = client
        # 注册脚本（redis-py 会自动用 EVALSHA + 回退到 EVAL）
        self._script = client.register_script(_TOKEN_BUCKET_LUA)  # type: ignore[attr-defined]

    async def check(
        self,
        *,
        policy: RateLimitPolicy,
        key: str,
        cost: float = 1.0,
    ) -> RateLimitResult:
        """判定是否放行。

        Args:
            policy: 限流规则。
            key: 桶标识（**必须包含 tenant_id**，见模块 docstring）。
            cost: 本次消耗多少令牌。默认 1（一次请求）。
                大请求（如批量 embedding）可以设更高值。

        Returns:
            `RateLimitResult`。`allowed=False` 时 `retry_after_ms` 给出建议等待。
        """
        import time

        now_ms = int(time.time() * 1000)

        raw = await self._script(
            keys=[key],
            args=[
                policy.rate,
                policy.burst,
                now_ms,
                cost,
                policy.ttl_s,
            ],
        )

        # Lua 返回 table 时 redis-py 给 list
        allowed = bool(int(raw[0]))
        remaining = int(raw[1])
        retry_after_ms = int(raw[2])

        if not allowed:
            log.warning(
                "rate_limited",
                policy=policy.name,
                key=key,
                retry_after_ms=retry_after_ms,
                note="请求被限流",
            )

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            retry_after_ms=retry_after_ms,
            limit_name=policy.name,
        )

    async def peek(self, *, policy: RateLimitPolicy, key: str) -> int:
        """只读查询剩余令牌（不消耗）。

        用于管理端展示"当前配额余量"。**不修改状态**——
        用 `check(cost=0)` 也能实现，但那仍会写回 key（更新 last_refill），
        对一个纯查询接口来说是不必要的写放大。
        """
        data = await self._client.hgetall(key)  # type: ignore[attr-defined]
        if not data:
            return policy.burst

        tokens = float(data.get(b"tokens") or data.get("tokens") or policy.burst)
        last_refill_raw = data.get(b"last_refill") or data.get("last_refill")
        if last_refill_raw is None:
            return int(tokens)

        import time

        now_ms = int(time.time() * 1000)
        elapsed = max(0, now_ms - int(last_refill_raw))
        refilled = min(policy.burst, tokens + (elapsed / 1000.0) * policy.rate)
        return int(refilled)

    async def reset(self, *, key: str) -> None:
        """清除桶状态（管理端"重置配额"用）。

        为什么需要：用户被限流后即使等待也需要时间恢复。
        客服处理投诉时能"立即恢复"比"请等 60 秒"体验好得多。
        **必须审计**（这是特权操作）。
        """
        await self._client.delete(key)  # type: ignore[attr-defined]
        log.warning("rate_limit_reset", key=key, note="配额被手动重置，应记入审计")


# ---------------------------------------------------------------- key 构造


def tenant_key(tenant_id: uuid.UUID) -> str:
    """租户级限流 key。

    **必须带 tenant_id**：Redis 没有 RLS，漏了就是跨租户共享配额。
    """
    return f"rl:t:{tenant_id}"


def user_key(tenant_id: uuid.UUID, user_id: uuid.UUID) -> str:
    """用户级限流 key（含租户，避免同一用户在多个租户下共享配额）。"""
    return f"rl:u:{tenant_id}:{user_id}"


def ip_key(ip: str) -> str:
    """IP 级限流 key（未认证请求用，无租户信息）。"""
    return f"rl:ip:{ip}"


__all__ = [
    "RateLimitPolicy",
    "RateLimitResult",
    "RateLimiter",
    "ip_key",
    "tenant_key",
    "user_key",
]
