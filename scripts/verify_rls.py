"""RLS 有效性验证：超级用户 vs 普通角色 vs 不同 tenant_id。

这是整个数据安全模型的核心验证。五种情况必须分别确认：

1. 超级用户（puraz）：**能绕过 RLS**（这是我们要求应用不用它的原因）
2. 普通角色（rag_app）不设 tenant_id：**看到 0 行**（安全默认）
3. 普通角色设 tenant_id=A：**只看到 A 的数据**
4. 普通角色设 tenant_id=B：**只看到 B 的数据**
5. SET LOCAL 不跨事务泄漏（连接池复用的真实风险）

**脚本自己准备并清理数据**：不依赖库里已有数据，
否则上次清理后就跑不通（这本身就是一个"测试不该依赖环境状态"的教训）。

用法：uv run python scripts/verify_rls.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TENANT_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
TENANT_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
PROBE_USER = uuid.UUID("99999999-9999-9999-9999-999999999999")
PROBE_EMAIL = "rls-probe@test.local"
TENANT_A_SLUG = "rls-probe-a"
TENANT_B_SLUG = "rls-probe-b"


def _engine_for(user: str, password: str):
    return create_async_engine(
        f"postgresql+asyncpg://{user}:{password}@127.0.0.1:5432/rag",
        pool_size=2,
        max_overflow=0,
    )


async def _admin():
    return create_async_engine(
        "postgresql+asyncpg://puraz:123456@127.0.0.1:5432/rag",
        pool_size=2,
        max_overflow=0,
    )


async def _seed() -> None:
    """准备验证数据：2 个租户，各 1 个知识库。"""
    engine = await _admin()
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO users (id, email, display_name, status, platform_role, "
                    "tokens_valid_after, created_at, updated_at) "
                    "VALUES (:id, :e, 'probe', 'active', 'none', now(), now(), now()) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                {"id": PROBE_USER, "e": PROBE_EMAIL},
            )
            for tenant_id, slug, name, kb_id, kb_name in (
                (
                    TENANT_A,
                    TENANT_A_SLUG,
                    "probe-A",
                    uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                    "KB-A",
                ),
                (
                    TENANT_B,
                    TENANT_B_SLUG,
                    "probe-B",
                    uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                    "KB-B",
                ),
            ):
                await conn.execute(
                    text(
                        "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                        "VALUES (:id, :s, :n, 'active', now(), now()) ON CONFLICT (id) DO NOTHING"
                    ),
                    {"id": tenant_id, "s": slug, "n": name},
                )
                await conn.execute(
                    text(
                        "INSERT INTO knowledge_bases (id, tenant_id, name, status, visibility, "
                        "confidentiality, created_by, default_chunking_strategy, "
                        "default_chunk_size, default_chunk_overlap, document_count, "
                        "created_at, updated_at) VALUES (:id, :t, :n, 'active', 'tenant', "
                        "'internal', :u, 'recursive', 800, 100, 0, now(), now()) "
                        "ON CONFLICT (id) DO NOTHING"
                    ),
                    {"id": kb_id, "t": tenant_id, "n": kb_name, "u": PROBE_USER},
                )
    finally:
        await engine.dispose()


async def _cleanup() -> None:
    engine = await _admin()
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM knowledge_bases WHERE id IN (:a, :b)"),
                {
                    "a": uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                    "b": uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                },
            )
            await conn.execute(
                text("DELETE FROM tenants WHERE id IN (:a, :b)"), {"a": TENANT_A, "b": TENANT_B}
            )
            await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": PROBE_USER})
    finally:
        await engine.dispose()


async def _count_kb(session: AsyncSession) -> int:
    result = await session.execute(text("SELECT count(*) FROM knowledge_bases"))
    return int(result.scalar_one())


async def _check(
    user: str, password: str, tenant_id: uuid.UUID | None, label: str
) -> tuple[str, int]:
    engine = _engine_for(user, password)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s, s.begin():
            if tenant_id is not None:
                await s.execute(
                    text("SELECT set_config('app.tenant_id', :t, true)"),
                    {"t": str(tenant_id)},
                )
            return label, await _count_kb(s)
    finally:
        await engine.dispose()


async def _check_session_leak() -> tuple[int, int]:
    """同一连接连续两事务：A 之后不设 tenant，第二事务必须看不到数据。"""
    engine = _engine_for("rag_app", "123456")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s:
            async with s.begin():
                await s.execute(
                    text("SELECT set_config('app.tenant_id', :t, true)"),
                    {"t": str(TENANT_A)},
                )
                first = await _count_kb(s)
            async with s.begin():
                second = await _count_kb(s)
        return first, second
    finally:
        await engine.dispose()


async def main() -> None:
    os.environ.setdefault("RAG_LOG_LEVEL", "ERROR")

    print("=" * 68)
    print("RLS 有效性验证")
    print("=" * 68)

    await _seed()
    try:
        print(f"{'场景':<40}{'可见行数':>10}")
        print("-" * 68)

        results = [
            await _check("puraz", "123456", TENANT_A, "超级用户 + tenant=A（RLS 被绕过）"),
            await _check("rag_app", "123456", None, "rag_app 未设 tenant_id"),
            await _check("rag_app", "123456", TENANT_A, "rag_app + tenant=A"),
            await _check("rag_app", "123456", TENANT_B, "rag_app + tenant=B"),
        ]
        for label, count in results:
            print(f"{label:<40}{count:>10}")

        first, second = await _check_session_leak()
        print()
        print("会话泄漏检查")
        print(f"  事务1(tenant=A) 可见: {first}")
        print(
            f"  事务2(未设置)  可见: {second}   {'✅ 无泄漏' if second == 0 else '❌ 发生泄漏！'}"
        )

        print()
        print("-" * 68)
        checks = [
            ("超级用户绕过 RLS（证明必须用普通角色）", results[0][1] == 2),
            ("未设 tenant_id 时看不到任何数据", results[1][1] == 0),
            ("tenant=A 只看到自己的数据", results[2][1] == 1),
            ("tenant=B 只看到自己的数据", results[3][1] == 1),
            ("SET LOCAL 不跨事务泄漏", second == 0),
        ]
        for check_label, ok in checks:
            print(f"  {'✅' if ok else '❌'} {check_label}")

        if all(ok for _, ok in checks):
            print("\n全部通过")
        else:
            print("\n存在失败项，RLS 配置有问题")
            raise SystemExit(1)
    finally:
        await _cleanup()


if __name__ == "__main__":
    asyncio.run(main())
