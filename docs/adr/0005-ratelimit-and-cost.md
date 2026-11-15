# A13 限流与成本：并发正确性靠 Redis，账平靠三条等式

> 日期：2026-11-15　环境：Valkey（Redis 协议）+ PostgreSQL 18
> 复现：`make test-int`（`tests/test_cost_and_ratelimit.py`，31 项）

## 1. 限流器：为什么必须用 Lua

朴素实现（Python 读-判断-写）在并发下必然超发：

```python
remaining = redis.get(key)     # 两个请求都读到 10
if remaining > 0:              # 两个都判断通过
    redis.decr(key)            # 实际只该放行 1 个
```

实测验证（`test_concurrent_requests_never_exceed_burst`）：

| 实现 | burst=10，50 个并发请求 | 放行数 |
|---|---|---|
| Lua 令牌桶（当前） | | **精确 10** |
| 读-判断-写（若用 Python） | | 会远大于 10 |

跨实例测试（`test_concurrent_across_multiple_instances`）用两个独立的
`RateLimiter` 实例（模拟多副本）共享同一桶，40 个并发仍精确放行 10 个
——证明原子性来自 Redis，不是进程内巧合。

### 令牌桶 vs 固定窗口

固定窗口有"边界双倍"问题：窗口末尾放行 N 个、下一窗口开头又放行 N 个，
跨窗口的 2 秒内实际放行 2N。令牌桶按平滑速率补充，不存在边界问题。

### 桶容量封顶

`test_bucket_does_not_overflow_burst` 验证：空闲 1 秒后（rate=100/s
理论上能补 100 个）放行数仍精确等于 burst=3。不封顶的话，
长时间空闲的用户可以瞬间发出远超配额的请求。

### Key 必须带 tenant_id

Redis 没有 RLS。key 漏了 tenant_id 就是跨租户共享配额——
A 租户用光后 B 租户也被限流。`tenant_key()` / `user_key()` 强制包含。

## 2. 缓存：三个必须处理的问题

| 问题 | 解法 | 验证 |
|---|---|---|
| 击穿（热点 key 过期瞬间并发 miss） | 进程内单飞（按 key 的 `asyncio.Event`） | loader 10 并发只跑 1 次 |
| 穿透（查询不存在的数据） | 空值缓存（短 TTL 60s） | loader 3 次只跑 1 次 |
| 一致性（文档更新后旧答案） | 按 tenant+namespace 用 SCAN 批量失效 | 失效后立刻不可读 |

### 一个被实测打回的设计：用 None 表示"未命中"

`get()` 最初返回 `None` 表示未命中，但**空值缓存也存 None**，
两者无法区分：

```
写入空值缓存 → get() 返回 None → get_or_load 认为未命中 → 又调 loader
→ 空值缓存永远失效，防穿透形同虚设
```

实测：3 次请求，loader 跑了 3 次。

修复：新增 `lookup()` 返回 `(found, value)` 二元组。
`get()` 保留为简化接口（不关心区别的场景用）。

**这是"哨兵值与被表示的数据类型重叠"的经典陷阱**——
用 None 同时表示"没有"和"值是 None"时信息必然丢失。

### TTL 分级

| 内容 | TTL | 理由 |
|---|---|---|
| 问答结果 | 1h | 文档可能更新 |
| Embedding | 24h | 只依赖文本内容 |
| 检索结果 | 5min | 文档更新要较快反映 |
| 权限/成员 | **60s** | 权限变更必须快速生效 |

权限缓存必须最短：用户被移出知识库后仍能查到内容是权限绕过。

## 3. 成本对账：三条等式

"账平"不是"总额看起来合理"，而是三条可验证的等式：

1. **记录的成本 == 按单价快照重算的成本**
   `cost_usd == (input-cached)/1e6*price_in + cached/1e6*price_cached + output/1e6*price_out`
2. **明细求和 == 聚合分项之和**（聚合 SQL 没漏维度）
3. **token 守恒**：`cached_tokens <= input_tokens`

### 为什么只对总额不够

两类错误会互相抵消：输入单价算高了 + 输出单价算低了 → 总额正常；
某次调用重复记录 + 另一次漏记 → 总额正常。逐条等式能定位到具体记录。

### 两个必须开的例外

- **我方缓存命中（`cache_hit=True`）**：成本被强制置 0（不计费），
  但用量字段仍有值（统计命中率用），所以按单价重算必然 ≠ 0。
  不开例外会让每条缓存命中都成"对账问题"。
- **单价全为空**：A3 的设计是"算不出成本但 Token 仍记录"，
  等补上定价后重算历史。这类记录跳过。

### 抽样必须标注

`max_detail_rows` 默认 5000。超过时报告里插入一条 `partial_check`
问题项，明确写"仅核对 N 条，结论不代表全量通过"。
**不标注就会让人误以为全量对账通过**——这比不做对账更危险。

## 4. 供应商降级

`ModelGateway.complete()` 新增 `fallback_provider` 参数：

- 主 provider 失败 → 自动尝试备用；
- **PERMANENT 类错误不降级**（参数错换 provider 也一样错，
  降级只是浪费时间且掩盖真实问题）；
- 返回值 `CallCost.is_fallback=True` + `provider=备用名`。

### 一个漏了 provider 的 bug

第一版只设置了 `is_fallback=True` 却没设 `provider`，导致降级调用的
provider 字段是空串。后果：按 provider 聚合成本时降级调用落到"未知"组
——而"降级用了哪家"正是最该被看见的信息。

测试 `test_gateway_marks_fallback_calls` 断言 `cost.provider == "backup"`
才抓到这个。

## 5. 未做与遗留

1. **限流未接入 HTTP 层**：`RateLimiter` 与 key 构造器就绪，
   但还没有 FastAPI 中间件/依赖去调用它。接入点在 A13 的"用户面接口"
   或 Phase B 的问答接口。
2. **配额（quota）未实现**：`7xxxx` 错误码段已留（`QUOTA_*`），
   但"租户月度限额、超额拒绝"需要配额表与周期重置逻辑，本步骤未做。
3. **成本日聚合表未建**：当前对账与聚合都直接查 `model_calls`
   （`ix_model_calls_unaggregated` 索引已就绪）。数据量上百万后
   需要物化日汇总表，否则看板查询会扫全表。
4. **`ModelCallRepository.record()` 没有调用方**：写入路径就绪，
   但问答/检索链路还没有把每次 LLM 调用落库的接线。
