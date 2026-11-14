"""启动自检：一次性验证 PG / Redis / pgvector 全部可用。

用途：
- 新人拉下代码后第一条命令，确认环境对；
- 部署后冒烟检查；
- CI 里作为集成测试的前置探测。

用法：uv run python scripts/check_deps.py
退出码 0 = 全通，1 = 有依赖不可用。
"""

from __future__ import annotations

import asyncio
import sys

from rag_core.db import Database
from rag_core.errors import AppError
from rag_core.logging import get_logger, setup_logging
from rag_core.redis import RedisClient
from rag_core.settings import get_settings

log = get_logger("check_deps")

# A3 建 chunks 表必需的扩展。缺了要在建表前装好，否则 ALTER 会失败。
REQUIRED_PG_EXTENSIONS = ("vector",)


async def check_postgres(settings: object) -> list[str]:
    """检查 PG 连通性与必需扩展。返回问题列表，空表示通过。"""
    from rag_core.settings import Settings

    assert isinstance(settings, Settings)
    problems: list[str] = []
    db = Database(settings)
    try:
        await db.ping()
        version = await db.server_version()
        extensions = await db.installed_extensions()
        print(f"  ✅ PostgreSQL {version}")
        print(f"     扩展: {', '.join(f'{k}@{v}' for k, v in extensions.items()) or '无'}")

        for required in REQUIRED_PG_EXTENSIONS:
            if required not in extensions:
                problems.append(
                    f"缺少扩展 {required}：sudo pacman -S pgvector 后执行 "
                    f"`CREATE EXTENSION {required};`"
                )
    except AppError as exc:
        problems.append(f"PostgreSQL: {exc}")
    finally:
        await db.close()
    return problems


async def check_redis(settings: object) -> list[str]:
    """检查 Redis/Valkey 连通性。"""
    from rag_core.settings import Settings

    assert isinstance(settings, Settings)
    problems: list[str] = []
    client = RedisClient(settings)
    try:
        await client.ping()
        info = await client.info_section("server")
        version = info.get("valkey_version") or info.get("redis_version") or "unknown"
        engine = "Valkey" if "valkey_version" in info else "Redis"
        print(f"  ✅ {engine} {version}")

        # 写入探测：不只看 PING，还要确认能真读写
        probe_key = "rag:healthcheck:probe"
        await client.client.set(probe_key, "1", ex=5)
        value = await client.client.get(probe_key)
        if value not in (b"1", "1"):
            problems.append(f"Redis 读写异常：GET 返回 {value!r}")
        await client.client.delete(probe_key)
        print("     读写: OK")
    except AppError as exc:
        problems.append(f"Redis: {exc}")
    finally:
        await client.close()
    return problems


async def main() -> int:
    settings = get_settings()
    setup_logging(settings)

    print("检查依赖连通性……")
    print(f"  PG:    {settings.database.redacted_dsn}")
    print(f"  Redis: {settings.redis.redacted_url}")

    problems: list[str] = []
    problems.extend(await check_postgres(settings))
    problems.extend(await check_redis(settings))

    print()
    if problems:
        print("❌ 发现问题：")
        for problem in problems:
            print(f"   - {problem}")
        return 1

    print("✅ 全部依赖就绪")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
