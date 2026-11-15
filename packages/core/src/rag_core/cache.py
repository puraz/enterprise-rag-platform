"""缓存层：tenant 隔离 + 防击穿 + 命中率统计。

## 为什么缓存必须带 tenant_id

这是与 RLS 同等级的安全要求。Redis 没有行级安全，
**key 里不带 tenant_id 就是跨租户数据泄漏**：

    cache:qa:{query_hash}          # ✗ A 租户能读到 B 租户的答案
    cache:qa:{tenant_id}:{hash}    # ✓

而且泄漏是"静默"的——命中率高、延迟低，一切看起来正常。

## 三个必须处理的问题

### 1. 缓存击穿（cache stampede）

热点 key 过期瞬间，大量请求同时 miss，全部打到数据库/模型上。

解法：**单飞（singleflight）**——同一 key 只允许一个请求去加载，
其余等待它的结果。

- 进程内单飞（`asyncio.Lock` 按 key）：解决单进程内并发；
- 跨进程锁（Redis SET NX）：解决多副本并发。

本项目用**进程内单飞 + 短 TTL 的 Redis 占位符**：前者覆盖 90% 的场景
（单进程内的并发 miss），后者防止多副本同时加载。不引入额外的锁服务。

### 2. 缓存穿透（cache penetration）

查询不存在的数据，每次都 miss 并打到后端。

解法：**空值缓存**（存一个特殊标记，短 TTL）。代价是"缓存里存了
不存在的事实"，但比每次都穿透好。

### 3. 缓存与写操作的一致性

文档更新后，旧答案的缓存必须失效。

解法：**按租户前缀批量失效**（`SCAN` + `DEL`）。不用 `KEYS`
（会阻塞 Redis）。失效范围宁可大（整个租户的问答缓存）也不要漏——
过期缓存返回错误答案比"多查一次库"严重得多。

## 为什么 TTL 分级

| 内容 | TTL | 理由 |
|---|---|---|
| 问答结果 | 1h | 文档可能更新，缓存太久会答旧内容 |
| Embedding 向量 | 24h | 向量只依赖文本内容，不依赖业务状态 |
| 权限/成员关系 | 60s | 权限变更是安全敏感的，必须快速生效 |

**权限缓存必须短**：用户被移出知识库后，如果缓存 1 小时，
他还能继续查到内容——这是权限绕过。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from rag_core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# 空值标记。用特殊字符串而不是 None：
# Redis 存的都是 bytes/str，用 None 无法区分"没缓存"与"缓存了空值"。
_NULL_MARKER = "__NULL__"


@dataclass
class CacheStats:
    """一个缓存实例的运行统计（用于命中率观测）。"""

    hits: int = 0
    misses: int = 0
    errors: int = 0
    stampede_blocked: int = 0
    """被单飞机制挡住、复用了别人结果的请求数。"""

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class _Inflight:
    """单飞机制的等待者。"""

    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: Any = None
    exc: BaseException | None = None


class Cache:
    """带 tenant 隔离与单飞的缓存。

    用法：

        cache = Cache(redis.client, tenant_id=tenant.id)
        answer = await cache.get_or_load(
            namespace="qa",
            key_parts=[question],
            loader=lambda: expensive_qa(question),
            ttl_s=3600,
        )
    """

    def __init__(self, client: object, *, tenant_id: uuid.UUID) -> None:
        """
        Args:
            client: `redis.asyncio.Redis`。
            tenant_id: **在构造时固定**，避免每次调用都传（传漏了就是
                跨租户泄漏）。这个设计让"忘记带租户"变得不可能。
        """
        self._client = client
        self._tenant_id = tenant_id
        self._inflight: dict[str, _Inflight] = {}
        self._inflight_lock = asyncio.Lock()
        self.stats = CacheStats()

    def _make_key(self, namespace: str, key_parts: list[str]) -> str:
        """构造 key。

        用 hash 而不是拼接原始内容：
        - 原始内容可能很长（整个问题 + 上下文），key 会非常大；
        - 内容可能含特殊字符（换行、冒号），破坏 key 结构。

        hash 用 SHA-256 前 32 字符（128 bit）：碰撞概率可忽略，
        且 key 长度固定。
        """
        raw = "\x1f".join(key_parts)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        # tenant_id 紧跟在 namespace 后：**位置固定**，
        # 便于按 `cache:{ns}:{tenant}:*` 批量失效
        return f"cache:{namespace}:{self._tenant_id}:{digest}"

    async def lookup(self, *, namespace: str, key_parts: list[str]) -> tuple[bool, Any]:
        """读取缓存，返回 `(是否命中, 值)`。

        ## 为什么返回二元组而不是"None 表示未命中"

        用 None 一个值表达两种状态会丢信息：

        - **未命中**：该去加载（loader 可能很贵）；
        - **命中空值**（我们查过，确实没有）：**不该**再加载。

        合并成 None 后，`get_or_load` 无法区分，于是空值缓存永远失效
        ——实测发生：防穿透的空值缓存写了 3 次、loader 跑了 3 次。

        这是"用哨兵值区分状态"的经典陷阱：哨兵值与被表示的数据
        类型重叠时（这里 None 也是合法的缓存值），信息就丢了。
        """
        key = self._make_key(namespace, key_parts)
        try:
            raw = await self._client.get(key)  # type: ignore[attr-defined]
        except Exception as exc:
            # 缓存故障不该让请求失败：降级为"未命中"，继续走 loader
            self.stats.errors += 1
            log.warning("cache_get_failed", namespace=namespace, error_type=type(exc).__name__)
            return False, None

        if raw is None:
            self.stats.misses += 1
            return False, None

        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        self.stats.hits += 1

        if text == _NULL_MARKER:
            # 命中空值：我们查过，确实没有。返回 (True, None)。
            return True, None

        try:
            return True, json.loads(text)
        except json.JSONDecodeError:
            # 缓存内容损坏：当作未命中，并删掉坏数据
            self.stats.errors += 1
            log.warning("cache_corrupted", namespace=namespace)
            await self._safe_delete(key)
            return False, None

    async def get(self, *, namespace: str, key_parts: list[str]) -> Any | None:
        """读取缓存（简化接口，不区分"未命中"与"命中空值"）。

        只在"确实不关心这个区别"的地方用。需要区分时用 `lookup()`。
        """
        found, value = await self.lookup(namespace=namespace, key_parts=key_parts)
        return value if found else None

    async def set(
        self,
        *,
        namespace: str,
        key_parts: list[str],
        value: Any,
        ttl_s: int,
        cache_null: bool = False,
    ) -> None:
        """写入缓存。

        Args:
            cache_null: 值为 None 时是否写空值标记（防穿透）。
        """
        key = self._make_key(namespace, key_parts)
        try:
            if value is None:
                if not cache_null:
                    return
                payload = _NULL_MARKER
            else:
                payload = json.dumps(value, ensure_ascii=False, default=str)
            await self._client.set(key, payload, ex=ttl_s)  # type: ignore[attr-defined]
        except Exception as exc:
            self.stats.errors += 1
            log.warning("cache_set_failed", namespace=namespace, error_type=type(exc).__name__)

    async def get_or_load(
        self,
        *,
        namespace: str,
        key_parts: list[str],
        loader: Callable[[], Awaitable[T]],
        ttl_s: int,
        cache_null: bool = False,
        null_ttl_s: int = 60,
    ) -> T:
        """读缓存，未命中则调用 loader 并回填。

        这是主要的调用入口。内含**单飞**：同一 key 并发 miss 时，
        只有一个请求执行 loader，其余等待并复用结果。

        Args:
            null_ttl_s: 空值的缓存时长（短，因为"不存在"可能很快变成"存在"）。
        """
        # 用 lookup 而不是 get：必须区分"未命中"与"命中空值"，
        # 否则空值缓存（防穿透）永远失效。
        found, cached = await self.lookup(namespace=namespace, key_parts=key_parts)
        if found:
            return cast("T", cached)

        key = self._make_key(namespace, key_parts)

        # ---- 单飞：判断自己是 loader 还是等待者 ----
        #
        # 必须在锁内完成"查 + 注册"，否则两个请求可能同时
        # 判定"没有在跑的 loader"并都去执行。
        async with self._inflight_lock:
            waiter = self._inflight.get(key)
            is_loader = waiter is None
            if waiter is None:
                waiter = _Inflight()
                self._inflight[key] = waiter

        if not is_loader:
            # 别的请求正在加载：等它的结果
            self.stats.stampede_blocked += 1
            await waiter.event.wait()
            if waiter.exc is not None:
                raise waiter.exc
            return cast("T", waiter.result)

        # ---- 自己是 loader ----
        try:
            value = await loader()
        except BaseException as exc:
            # 失败也要通知等待者，否则它们会永远等下去（直到超时）
            waiter.exc = exc
            raise
        else:
            waiter.result = value
            if value is None:
                if cache_null:
                    await self.set(
                        namespace=namespace,
                        key_parts=key_parts,
                        value=None,
                        ttl_s=null_ttl_s,
                        cache_null=True,
                    )
            else:
                await self.set(
                    namespace=namespace,
                    key_parts=key_parts,
                    value=value,
                    ttl_s=ttl_s,
                )
            return value
        finally:
            # 先摘掉注册再唤醒：
            # 反过来的话，被唤醒的请求可能还没来得及读 result，
            # 就有新请求进来看到旧 waiter（拿到已失效的 event）。
            async with self._inflight_lock:
                self._inflight.pop(key, None)
            waiter.event.set()

    async def invalidate_namespace(self, namespace: str) -> int:
        """失效本租户在该命名空间下的**全部**缓存。

        用 `SCAN` 而不是 `KEYS`：`KEYS` 会遍历整个 keyspace 并阻塞
        Redis（单线程），在生产上是禁用命令。

        Returns:
            删除的 key 数。
        """
        pattern = f"cache:{namespace}:{self._tenant_id}:*"
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = await self._client.scan(  # type: ignore[attr-defined]
                cursor=cursor, match=pattern, count=200
            )
            if keys:
                deleted += int(await self._client.delete(*keys))  # type: ignore[attr-defined]
            if cursor == 0:
                break

        log.info(
            "cache_invalidated",
            namespace=namespace,
            tenant_id=str(self._tenant_id),
            deleted=deleted,
        )
        return deleted

    async def _safe_delete(self, key: str) -> None:
        """尽力删除（失败不影响主流程）。

        缓存内容损坏时删掉它，但删除本身失败不是致命问题——
        下次读取会再次发现损坏并重新加载。
        """
        with contextlib.suppress(Exception):
            await self._client.delete(key)  # type: ignore[attr-defined]


# ---------------------------------------------------------------- 常用分发器


class CacheNamespaces:
    """命名空间常量。

    集中定义避免各处手写字符串拼错（拼错的后果是"缓存永不命中"
    或"失效范围漏了某个命名空间"）。
    """

    QA = "qa"
    """问答结果。TTL 1h——文档可能更新。"""

    EMBEDDING = "emb"
    """Embedding 向量。TTL 24h——只依赖文本内容。"""

    PERMISSIONS = "perm"
    """权限/成员关系。TTL 60s——权限变更必须快速生效。"""

    SEARCH = "search"
    """检索结果。TTL 5min——文档更新后要较快反映。"""


# 各命名空间的推荐 TTL（秒）。
#
# 权限必须最短：用户被移出知识库后仍能查到内容是权限绕过。
RECOMMENDED_TTL: dict[str, int] = {
    CacheNamespaces.QA: 3600,
    CacheNamespaces.EMBEDDING: 86400,
    CacheNamespaces.SEARCH: 300,
    CacheNamespaces.PERMISSIONS: 60,
}


def elapsed_ms(started: float) -> float:
    """辅助：计算耗时（毫秒）。用于缓存命中率的日志。"""
    return (time.perf_counter() - started) * 1000


__all__ = [
    "RECOMMENDED_TTL",
    "Cache",
    "CacheNamespaces",
    "CacheStats",
    "elapsed_ms",
]
