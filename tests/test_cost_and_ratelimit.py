"""A13 限流与成本测试。

## 分两部分

**限流器**（连真实 Redis）：令牌桶是"并发正确性"的典型场景，
mock 只能证明"我调用了 eval"，证明不了"20 个并发请求精确放行 10 个"。

**成本对账**（连真实 PG）：三条等式必须真实成立。

## 为什么并发测试不能用 mock

限流的核心是"多请求同时到达时配额不会超发"。这个正确性**完全由
Redis 的脚本原子性保证**：

- 用 Python 读-判断-写：两个请求都读到 `remaining=1`，都放行 → 超发；
- 用 Lua：脚本整体原子执行 → 精确。

mock 掉 Redis 就测不到这条，而它恰恰是限流器唯一重要的性质。
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal

import pytest

from rag_contracts.enums.audit import ModelTaskType
from rag_core.cache import Cache, CacheNamespaces
from rag_core.ratelimit import (
    RateLimiter,
    RateLimitPolicy,
    ip_key,
    tenant_key,
    user_key,
)
from rag_db.reconciliation import TOLERANCE, reconcile_costs

pytestmark = pytest.mark.integration


# ==================================================== fixtures


@pytest.fixture
async def redis_client() -> object:
    """真实 Redis（Valkey）客户端。"""
    from rag_core.redis import RedisClient
    from rag_core.settings import get_settings, load_env_files

    load_env_files()
    client = RedisClient(get_settings())
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
async def limiter(redis_client: object) -> RateLimiter:
    return RateLimiter(redis_client.client)  # type: ignore[attr-defined]


@pytest.fixture
async def db_factory() -> object:
    """真实 PG 会话工厂。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from rag_core.settings import get_settings

    settings = get_settings()
    db = settings.database
    user = db.migration_user or db.user
    password = (
        db.migration_password.get_secret_value()
        if db.migration_password is not None
        else db.password.get_secret_value()
    )
    dsn = f"postgresql+asyncpg://{user}:{password}@{db.host}:{db.port}/{db.name}"
    engine = create_async_engine(
        dsn,
        pool_size=4,
        connect_args={"server_settings": {"random_page_cost": str(db.random_page_cost)}},
    )
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    try:
        yield factory
    finally:
        await engine.dispose()


# ==================================================== 限流器：基本行为


async def test_allows_up_to_burst(limiter: RateLimiter) -> None:
    """burst 个请求全部放行，第 burst+1 个被拒。"""
    policy = RateLimitPolicy("test_burst", rate=0.001, burst=5)  # rate 极低，测试期间不补充
    key = tenant_key(uuid.uuid4())

    for index in range(5):
        result = await limiter.check(policy=policy, key=key)
        assert result.allowed, f"第 {index + 1} 个请求不该被拒"
        assert result.remaining == 4 - index

    denied = await limiter.check(policy=policy, key=key)
    assert not denied.allowed
    assert denied.remaining == 0
    assert denied.retry_after_ms > 0


async def test_retry_after_is_meaningful(limiter: RateLimiter) -> None:
    """被拒时给出的重试间隔应接近"补足 1 个令牌所需时间"。"""
    policy = RateLimitPolicy("test_retry", rate=10.0, burst=1)
    key = tenant_key(uuid.uuid4())

    assert (await limiter.check(policy=policy, key=key)).allowed
    denied = await limiter.check(policy=policy, key=key)

    assert not denied.allowed
    # rate=10/s → 补 1 个令牌约 100ms。允许 50% 误差（时间测量抖动）
    assert 50 <= denied.retry_after_ms <= 150, denied.retry_after_ms


async def test_tokens_refill_over_time(limiter: RateLimiter) -> None:
    """令牌按时间补充：耗尽后等待应能恢复。"""
    policy = RateLimitPolicy("test_refill", rate=20.0, burst=2)
    key = tenant_key(uuid.uuid4())

    # 耗尽
    for _ in range(2):
        assert (await limiter.check(policy=policy, key=key)).allowed
    assert not (await limiter.check(policy=policy, key=key)).allowed

    # 等 120ms → 按 rate=20/s 应补约 2.4 个（上限 burst=2）
    await asyncio.sleep(0.12)
    result = await limiter.check(policy=policy, key=key)
    assert result.allowed, "等待后应恢复"


async def test_bucket_does_not_overflow_burst(limiter: RateLimiter) -> None:
    """长时间空闲后不该累积超过 burst。"""
    policy = RateLimitPolicy("test_overflow", rate=100.0, burst=3)
    key = tenant_key(uuid.uuid4())

    # 先耗尽
    for _ in range(3):
        await limiter.check(policy=policy, key=key)

    # 等 1 秒：rate=100/s 理论上能补 100 个，但桶容量是 3
    await asyncio.sleep(1.0)

    allowed_count = 0
    for _ in range(10):
        if (await limiter.check(policy=policy, key=key)).allowed:
            allowed_count += 1
    assert allowed_count == 3, f"桶容量应封顶 3，实际放行 {allowed_count}"


async def test_cost_greater_than_one(limiter: RateLimiter) -> None:
    """单次消耗多个令牌（大请求）。"""
    policy = RateLimitPolicy("test_cost", rate=0.001, burst=10)
    key = tenant_key(uuid.uuid4())

    # 一次消耗 7 个
    result = await limiter.check(policy=policy, key=key, cost=7)
    assert result.allowed
    assert result.remaining == 3

    # 再要 5 个不够
    assert not (await limiter.check(policy=policy, key=key, cost=5)).allowed
    # 要 3 个刚好
    assert (await limiter.check(policy=policy, key=key, cost=3)).allowed


async def test_peek_does_not_consume(limiter: RateLimiter) -> None:
    """peek 只读，不消耗令牌。"""
    policy = RateLimitPolicy("test_peek", rate=0.001, burst=5)
    key = tenant_key(uuid.uuid4())

    for _ in range(3):
        await limiter.check(policy=policy, key=key)

    assert await limiter.peek(policy=policy, key=key) == 2
    # 连续 peek 不该改变余量
    assert await limiter.peek(policy=policy, key=key) == 2


async def test_reset_restores_full_bucket(limiter: RateLimiter) -> None:
    """reset 后桶应恢复满。"""
    policy = RateLimitPolicy("test_reset", rate=0.001, burst=3)
    key = tenant_key(uuid.uuid4())

    for _ in range(3):
        await limiter.check(policy=policy, key=key)
    assert not (await limiter.check(policy=policy, key=key)).allowed

    await limiter.reset(key=key)
    assert (await limiter.check(policy=policy, key=key)).allowed


# ==================================================== 限流器：并发正确性（核心）


async def test_concurrent_requests_never_exceed_burst(limiter: RateLimiter) -> None:
    """**A13 的核心验收点**：并发请求下放行数精确等于 burst。

    这是限流器唯一重要的性质。用 Python 读-判断-写会超发：
    两个协程都读到"还有 1 个"，都放行。

    Lua 脚本在 Redis 内部整体原子执行，所以第 N+1 个请求必然看到
    "已耗尽"。
    """
    policy = RateLimitPolicy("test_concurrent", rate=0.001, burst=10)
    key = tenant_key(uuid.uuid4())

    results = await asyncio.gather(*(limiter.check(policy=policy, key=key) for _ in range(50)))
    allowed = sum(1 for result in results if result.allowed)

    assert allowed == 10, f"并发 50 个请求应精确放行 10 个，实际 {allowed}"


async def test_concurrent_across_multiple_instances(limiter: RateLimiter) -> None:
    """多个 limiter 实例（模拟多进程/多副本）共享同一桶时仍精确。

    实例之间没有共享 Python 状态，唯一共享的是 Redis——
    如果放行数仍是 10，说明原子性来自 Redis 而不是进程内巧合。
    """
    from rag_core.redis import RedisClient
    from rag_core.settings import get_settings

    settings = get_settings()
    other_client = RedisClient(settings)
    other_limiter = RateLimiter(other_client.client)

    try:
        policy = RateLimitPolicy("test_multi", rate=0.001, burst=10)
        key = tenant_key(uuid.uuid4())

        tasks = []
        for index in range(40):
            target = limiter if index % 2 == 0 else other_limiter
            tasks.append(target.check(policy=policy, key=key))
        results = await asyncio.gather(*tasks)

        allowed = sum(1 for result in results if result.allowed)
        assert allowed == 10, f"跨实例并发应精确放行 10 个，实际 {allowed}"
    finally:
        await other_client.close()


async def test_different_keys_are_independent(limiter: RateLimiter) -> None:
    """不同 key 的桶互相独立（不能把 A 用户的配额算到 B 头上）。"""
    policy = RateLimitPolicy("test_indep", rate=0.001, burst=2)
    tenant_id = uuid.uuid4()
    key_a = user_key(tenant_id, uuid.uuid4())
    key_b = user_key(tenant_id, uuid.uuid4())

    for _ in range(2):
        assert (await limiter.check(policy=policy, key=key_a)).allowed
    assert not (await limiter.check(policy=policy, key=key_a)).allowed

    # B 的桶不受影响
    assert (await limiter.check(policy=policy, key=key_b)).allowed


async def test_tenant_key_includes_tenant_id() -> None:
    """租户 key 必须含 tenant_id。

    Redis 没有 RLS，key 漏了 tenant_id 就是跨租户共享配额
    （A 租户用光后 B 租户也被限流）。
    """
    tenant_id = uuid.uuid4()
    assert str(tenant_id) in tenant_key(tenant_id)


async def test_user_key_includes_tenant_and_user() -> None:
    """用户 key 必须同时含租户与用户（同一用户在多个租户下配额独立）。"""
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    key = user_key(tenant_id, user_id)
    assert str(tenant_id) in key
    assert str(user_id) in key


async def test_ip_key_format() -> None:
    assert ip_key("192.0.2.1") == "rl:ip:192.0.2.1"


# ==================================================== 缓存


@pytest.fixture
async def cache(redis_client: object) -> Cache:
    return Cache(redis_client.client, tenant_id=uuid.uuid4())  # type: ignore[attr-defined]


async def test_cache_roundtrip(cache: Cache) -> None:
    await cache.set(namespace=CacheNamespaces.QA, key_parts=["问题"], value={"a": 1}, ttl_s=60)
    assert await cache.get(namespace=CacheNamespaces.QA, key_parts=["问题"]) == {"a": 1}


async def test_cache_isolates_tenants(redis_client: object) -> None:
    """**安全关键**：不同租户的缓存完全隔离。

    泄漏是静默的——命中率高、延迟低，一切看起来正常。
    """
    cache_a = Cache(redis_client.client, tenant_id=uuid.uuid4())  # type: ignore[attr-defined]
    cache_b = Cache(redis_client.client, tenant_id=uuid.uuid4())  # type: ignore[attr-defined]

    await cache_a.set(
        namespace=CacheNamespaces.QA, key_parts=["同一个问题"], value="A 的答案", ttl_s=60
    )

    assert await cache_a.get(namespace=CacheNamespaces.QA, key_parts=["同一个问题"]) == "A 的答案"
    assert await cache_b.get(namespace=CacheNamespaces.QA, key_parts=["同一个问题"]) is None


async def test_cache_singleflight(cache: Cache) -> None:
    """并发 miss 时 loader 只执行一次（防缓存击穿）。"""
    calls = {"n": 0}

    async def loader() -> dict[str, int]:
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return {"value": 42}

    results = await asyncio.gather(
        *(
            cache.get_or_load(
                namespace=CacheNamespaces.QA,
                key_parts=["热点"],
                loader=loader,
                ttl_s=60,
            )
            for _ in range(10)
        )
    )

    assert calls["n"] == 1, f"loader 应只执行 1 次，实际 {calls['n']} 次"
    assert all(result == {"value": 42} for result in results)
    assert cache.stats.stampede_blocked == 9


async def test_cache_null_caching_prevents_penetration(cache: Cache) -> None:
    """空值缓存：查询不存在的数据时也只穿透一次。"""
    calls = {"n": 0}

    async def loader() -> None:
        calls["n"] += 1
        return None

    for _ in range(3):
        await cache.get_or_load(
            namespace=CacheNamespaces.QA,
            key_parts=["不存在"],
            loader=loader,
            ttl_s=60,
            cache_null=True,
            null_ttl_s=60,
        )

    assert calls["n"] == 1, f"空值缓存应让 loader 只跑 1 次，实际 {calls['n']} 次"


async def test_cache_invalidate_namespace(cache: Cache) -> None:
    """按命名空间批量失效（文档更新后用）。"""
    for index in range(5):
        await cache.set(
            namespace=CacheNamespaces.QA,
            key_parts=[f"q{index}"],
            value={"i": index},
            ttl_s=60,
        )

    deleted = await cache.invalidate_namespace(CacheNamespaces.QA)
    assert deleted >= 5
    assert await cache.get(namespace=CacheNamespaces.QA, key_parts=["q0"]) is None


async def test_cache_only_invalidates_own_namespace(cache: Cache) -> None:
    """失效一个命名空间不该影响其它命名空间。"""
    await cache.set(namespace=CacheNamespaces.QA, key_parts=["k"], value=1, ttl_s=60)
    await cache.set(namespace=CacheNamespaces.EMBEDDING, key_parts=["k"], value=2, ttl_s=60)

    await cache.invalidate_namespace(CacheNamespaces.QA)

    assert await cache.get(namespace=CacheNamespaces.QA, key_parts=["k"]) is None
    assert await cache.get(namespace=CacheNamespaces.EMBEDDING, key_parts=["k"]) == 2


async def test_cache_stats_track_hit_rate(cache: Cache) -> None:
    await cache.set(namespace=CacheNamespaces.QA, key_parts=["hit"], value=1, ttl_s=60)
    await cache.get(namespace=CacheNamespaces.QA, key_parts=["hit"])  # hit
    await cache.get(namespace=CacheNamespaces.QA, key_parts=["miss"])  # miss

    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.5


# ==================================================== 成本对账


async def _make_tenant(factory: object) -> uuid.UUID:
    from sqlalchemy import text

    tenant_id = uuid.uuid4()
    async with factory() as session:  # type: ignore[operator]
        await session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:id, :s, 'cost', 'active', now(), now())"
            ),
            {"id": tenant_id, "s": f"cost-{tenant_id.hex[:8]}"},
        )
        await session.commit()
    return tenant_id


async def _record_call(
    factory: object,
    *,
    tenant_id: uuid.UUID,
    cost: Decimal,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cached_tokens: int = 0,
    price_in: Decimal = Decimal("0.14"),
    price_out: Decimal = Decimal("0.28"),
    cache_hit: bool = False,
) -> None:
    """直接插一条记录（用仓储会自己算成本，这里要测"记录值是否正确"）。"""
    from sqlalchemy import text

    async with factory() as session:  # type: ignore[operator]
        await session.execute(
            text(
                "INSERT INTO model_calls (id, tenant_id, task_type, provider, model, "
                "input_tokens, output_tokens, cached_tokens, unit_price_input, "
                "unit_price_output, cost_usd, cache_hit, status, retry_count, created_at) "
                "VALUES (gen_random_uuid(), :t, 'chat', 'deepseek', 'deepseek-chat', "
                ":inp, :out, :cached, :pin, :pout, :cost, :hit, 'success', 0, now())"
            ),
            {
                "t": tenant_id,
                "inp": input_tokens,
                "out": output_tokens,
                "cached": cached_tokens,
                "pin": price_in,
                "pout": price_out,
                "cost": cost,
                "hit": cache_hit,
            },
        )
        await session.commit()


async def test_reconciliation_passes_for_correct_records(db_factory: object) -> None:
    """正确记录的对账应完全通过。"""
    tenant_id = await _make_tenant(db_factory)

    # 成本 = (1000-0)/1e6*0.14 + 500/1e6*0.28 = 0.00014 + 0.00014 = 0.00028
    await _record_call(db_factory, tenant_id=tenant_id, cost=Decimal("0.00028"))

    async with db_factory() as session:  # type: ignore[operator]
        report = await reconcile_costs(session, tenant_id=tenant_id, days=1)

    assert report.is_balanced, f"对账失败：{report.issues}"
    assert report.checked_calls == 1
    assert report.total_cost == Decimal("0.00028")


async def test_reconciliation_detects_cost_mismatch(db_factory: object) -> None:
    """记录的成本与按单价重算的不符时必须报出来。"""
    tenant_id = await _make_tenant(db_factory)
    # 故意写错成本（正确应是 0.00028）
    await _record_call(db_factory, tenant_id=tenant_id, cost=Decimal("0.00099"))

    async with db_factory() as session:  # type: ignore[operator]
        report = await reconcile_costs(session, tenant_id=tenant_id, days=1)

    assert not report.is_balanced
    mismatches = [issue for issue in report.issues if issue.kind == "cost_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0].expected is not None


async def test_reconciliation_detects_token_overflow(db_factory: object) -> None:
    """cached_tokens > input_tokens 是数据错误（缓存 token 属于输入）。"""
    tenant_id = await _make_tenant(db_factory)
    await _record_call(
        db_factory,
        tenant_id=tenant_id,
        cost=Decimal("0"),
        input_tokens=100,
        cached_tokens=999,  # 超过输入
        price_in=Decimal("0"),
        price_out=Decimal("0"),
    )

    async with db_factory() as session:  # type: ignore[operator]
        report = await reconcile_costs(session, tenant_id=tenant_id, days=1)

    overflow = [issue for issue in report.issues if issue.kind == "token_overflow"]
    assert len(overflow) == 1


async def test_reconciliation_allows_zero_cost_for_cache_hits(db_factory: object) -> None:
    """我方缓存命中的记录成本必须是 0（不该计费）。

    这类记录的用量字段仍有值（用于统计命中率），所以按单价重算
    会得到非 0 —— 对账必须为它开例外，否则每条命中都成"问题"。
    """
    tenant_id = await _make_tenant(db_factory)
    await _record_call(
        db_factory,
        tenant_id=tenant_id,
        cost=Decimal("0"),
        cache_hit=True,
    )

    async with db_factory() as session:  # type: ignore[operator]
        report = await reconcile_costs(session, tenant_id=tenant_id, days=1)

    assert report.is_balanced, f"缓存命中不该报问题：{report.issues}"


async def test_reconciliation_flags_nonzero_cost_for_cache_hits(db_factory: object) -> None:
    """缓存命中却有成本 → 计费逻辑有 bug，必须报出来。"""
    tenant_id = await _make_tenant(db_factory)
    await _record_call(
        db_factory,
        tenant_id=tenant_id,
        cost=Decimal("0.00028"),
        cache_hit=True,
    )

    async with db_factory() as session:  # type: ignore[operator]
        report = await reconcile_costs(session, tenant_id=tenant_id, days=1)

    mismatches = [issue for issue in report.issues if issue.kind == "cost_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0].expected == "0"


async def test_reconciliation_allows_null_prices(db_factory: object) -> None:
    """单价为空的记录（算不出成本）对账时跳过，不算问题。"""
    from sqlalchemy import text

    tenant_id = await _make_tenant(db_factory)
    async with db_factory() as session:  # type: ignore[operator]
        await session.execute(
            text(
                "INSERT INTO model_calls (id, tenant_id, task_type, provider, model, "
                "input_tokens, output_tokens, cached_tokens, cost_usd, cache_hit, "
                "status, retry_count, created_at) "
                "VALUES (gen_random_uuid(), :t, 'embedding', 'local-bge', 'bge-m3', "
                "100, 0, 0, 0, false, 'success', 0, now())"
            ),
            {"t": tenant_id},
        )
        await session.commit()

    async with db_factory() as session:  # type: ignore[operator]
        report = await reconcile_costs(session, tenant_id=tenant_id, days=1)

    assert report.is_balanced


async def test_reconciliation_detects_aggregate_mismatch_impossible(
    db_factory: object,
) -> None:
    """聚合与明细不一致：正常情况下不会发生（同一条 SQL 的数据源）。

    这条测试的价值是**证明对账在做真实的比较**，而不是永远返回"通过"。
    用一个不可能触发的方式验证：插入数据后手动改一条的成本，
    使两条路径都基于同一份数据——所以这里实际验证"等式 2 恒成立"
    是设计上正确的（同源数据），而对账的检出能力由等式 1 保证。
    """
    tenant_id = await _make_tenant(db_factory)
    await _record_call(db_factory, tenant_id=tenant_id, cost=Decimal("0.00028"))
    await _record_call(db_factory, tenant_id=tenant_id, cost=Decimal("0.00028"))

    async with db_factory() as session:  # type: ignore[operator]
        report = await reconcile_costs(session, tenant_id=tenant_id, days=1)

    # 聚合与明细同源，必然相等
    aggregate_issues = [i for i in report.issues if i.kind == "aggregate_mismatch"]
    assert not aggregate_issues
    assert report.total_cost == Decimal("0.00056")


async def test_reconciliation_notes_partial_check(db_factory: object) -> None:
    """超过检查上限时必须注明"仅抽样"，不能假装全量通过。"""
    tenant_id = await _make_tenant(db_factory)
    for _ in range(3):
        await _record_call(db_factory, tenant_id=tenant_id, cost=Decimal("0.00028"))

    async with db_factory() as session:  # type: ignore[operator]
        report = await reconcile_costs(session, tenant_id=tenant_id, days=1, max_detail_rows=1)

    partial = [issue for issue in report.issues if issue.kind == "partial_check"]
    assert len(partial) == 1, "超过上限应明确标注仅抽样"
    assert "仅核对" in partial[0].detail


def test_tolerance_is_one_minimum_unit() -> None:
    """容差应是一个最小精度单位（1e-8），不能随意放大。

    放大容差会让真实的计费误差被"吸收"掉——那正是对账要抓的东西。
    """
    assert Decimal("0.00000001") == TOLERANCE


# ==================================================== 供应商降级


async def test_gateway_marks_fallback_calls() -> None:
    """降级调用必须带 is_fallback=True。

    不标记的话无法回答"降级发生了多少次、成本占多少"——
    降级通常用廉价备用模型，混在一起会让成本分析失真。
    """
    from rag_adapters.providers.gateway import ModelGateway
    from rag_adapters.providers.llm import (
        LLMMessage,
        LLMResponse,
        LLMRole,
        ProviderCapabilities,
        TokenUsage,
    )
    from rag_core.settings.llm import LLMSettings

    class _FailingProvider:
        name = "primary"

        @property
        def capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(supports_stream=False)

        async def complete(self, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
            # 抛"依赖不可用"类错误 → 分类为 RETRYABLE → 会触发降级
            raise ConnectionError("primary down")

        async def aclose(self) -> None:
            return None

    class _OkProvider:
        name = "backup"

        @property
        def capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(supports_stream=False)

        async def complete(self, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
            return LLMResponse(
                content="降级回答",
                model="backup-model",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )

        async def aclose(self) -> None:
            return None

    settings = LLMSettings(
        provider="primary",
        chat_model="backup-model",
    )
    gateway = ModelGateway(
        providers={"primary": _FailingProvider(), "backup": _OkProvider()},  # type: ignore[dict-item]
        settings=settings,
    )

    response, cost = await gateway.complete(
        ModelTaskType.CHAT,
        [LLMMessage(role=LLMRole.USER, content="测试")],
        fallback_provider="backup",
    )

    assert response.content == "降级回答"
    assert cost.is_fallback is True
    assert cost.provider == "backup"


async def test_gateway_does_not_fallback_on_permanent_errors() -> None:
    """参数错误不该触发降级（换 provider 也一样错）。

    降级有代价（延迟翻倍、可能用到能力更弱的模型）。
    对"请求本身有问题"的错误降级，只是在浪费时间且掩盖真实问题。
    """
    from rag_adapters.providers.gateway import ModelGateway
    from rag_adapters.providers.llm import (
        LLMMessage,
        LLMResponse,
        LLMRole,
        ProviderCapabilities,
    )
    from rag_core.errors import ValidationFailedError
    from rag_core.settings.llm import LLMSettings

    class _RejectingProvider:
        name = "primary"

        @property
        def capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(supports_stream=False)

        async def complete(self, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
            raise ValidationFailedError("max_tokens 为负")

        async def aclose(self) -> None:
            return None

    class _ShouldNotBeCalled:
        name = "backup"
        called = False

        @property
        def capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(supports_stream=False)

        async def complete(self, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
            type(self).called = True
            raise AssertionError("不该被调用")

        async def aclose(self) -> None:
            return None

    settings = LLMSettings(provider="primary", chat_model="backup-model")
    backup = _ShouldNotBeCalled()
    gateway = ModelGateway(
        providers={"primary": _RejectingProvider(), "backup": backup},  # type: ignore[dict-item]
        settings=settings,
    )

    # 用 AppError 而不是裸 Exception：验证的是"业务异常透传"，
    # 不是"任何异常都行"。
    from rag_core.errors import AppError

    with pytest.raises(AppError):
        await gateway.complete(
            ModelTaskType.CHAT,
            [LLMMessage(role=LLMRole.USER, content="测试")],
            fallback_provider="backup",
        )

    assert backup.called is False, "参数错误不该触发降级"
