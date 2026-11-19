"""RLS（行级安全）集成测试。

这批测试的价值在于：它们**必须用非超级用户运行**才有意义。
超级用户会绕过所有 RLS 策略，用它跑测试会"全绿"但什么都没验证。

因此这里显式创建 `rag_app` 连接（NOSUPERUSER NOBYPASSRLS），
并额外验证"超级用户确实绕过"——这条反向证明用来提醒：
**应用连接串不能用超级用户，否则 RLS 形同虚设**。

覆盖的路径（对应 docs/PROMPT.md 的要求）：
1. 跨租户读不到（SELECT）
2. 跨租户写不进（INSERT/UPDATE）
3. 未设 tenant 上下文时零行返回（安全默认，不是报错）
4. SET LOCAL 不跨事务泄漏（连接池复用的真实风险）
5. 超级用户绕过（反证：为什么必须用普通角色）
6. 并发下隔离仍成立（压测路径）
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_core.settings import get_settings

pytestmark = pytest.mark.integration

APP_ROLE = "rag_app"


def _app_engine():
    """创建以 rag_app（非超级用户）连接的引擎。

    ## 密码为什么必须从 Settings 读，不能硬编码

    这里原先写死了 `APP_PASSWORD = "123456"`（当时本机 PG 里 rag_app
    恰好就是这个密码）。换到 Docker compose 后容器里的 rag_app 密码来自
    `RAG_APP_PASSWORD`（默认 `rag_local_dev_only`），于是本文件所有用例
    全部报 `password authentication failed for user "rag_app"`——
    而错误信息完全指向不到"测试写死了密码"这个根因。

    AGENTS.md 的约定是「连接信息统一从 `Settings` 读」：同一份测试代码
    要在本机 PG（角色 puraz 建库）与 compose（rag_admin 建库）两种环境
    都能跑，不能有两套连接参数。
    """
    settings = get_settings()
    return create_async_engine(
        f"postgresql+asyncpg://{APP_ROLE}:{settings.database.password.get_secret_value()}"
        f"@{settings.database.host}:{settings.database.port}/{settings.database.name}",
        pool_size=5,
        max_overflow=0,
    )


async def _seed_tenant_with_kb(
    admin_session: AsyncSession, *, label: str
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """用超级用户建一个租户 + 用户 + 知识库（绕过 RLS 准备数据）。"""
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    await admin_session.execute(
        text(
            "INSERT INTO users (id, email, display_name, status, platform_role, "
            "tokens_valid_after, created_at, updated_at) "
            "VALUES (:id, :e, 'U', 'active', 'none', now(), now(), now())"
        ),
        {"id": user_id, "e": f"rls-{user_id.hex[:8]}@example.test"},
    )
    await admin_session.execute(
        text(
            "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
            "VALUES (:id, :s, :n, 'active', now(), now())"
        ),
        {"id": tenant_id, "s": f"rls-{tenant_id.hex[:10]}", "n": label},
    )
    await admin_session.execute(
        text(
            "INSERT INTO knowledge_bases (id, tenant_id, name, status, visibility, "
            "confidentiality, created_by, default_chunking_strategy, default_chunk_size, "
            "default_chunk_overlap, document_count, created_at, updated_at) "
            "VALUES (:id, :t, :n, 'active', 'tenant', 'internal', :u, 'recursive', "
            "800, 100, 0, now(), now())"
        ),
        {"id": kb_id, "t": tenant_id, "n": f"KB-{label}", "u": user_id},
    )
    return tenant_id, kb_id, user_id


async def _cleanup(admin_session: AsyncSession, tenant_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """清理测试数据。

    顺序很重要：`knowledge_bases.created_by` 是 `ON DELETE RESTRICT`，
    所以必须**先删知识库再删用户**。直接删用户会被外键拦住，
    而删租户虽然能级联删知识库，但如果租户本身已被删过，
    级联不会发生，用户就删不掉了。

    显式按依赖顺序删，比依赖级联更可靠。
    """
    await admin_session.execute(
        text("DELETE FROM knowledge_bases WHERE tenant_id = :t"), {"t": tenant_id}
    )
    await admin_session.execute(text("DELETE FROM tenants WHERE id = :t"), {"t": tenant_id})
    # 精确删本测试创建的那个用户（不能用邮箱前缀批量删：
    # 同一测试里可能建了多个用户，批量删会破坏其它租户的 FK）
    await admin_session.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})


# ==================================================== 读隔离


def test_app_role_cannot_see_other_tenant_rows() -> None:
    """**核心安全测试**：rag_app 设了 tenant=A，读不到 tenant=B 的数据。

    这是 RLS 存在的根本理由。没有它，一个漏写 WHERE tenant_id 的
    仓储方法就会让 A 租户看到 B 租户的全部文档。
    """

    async def scenario() -> tuple[int, int, int]:
        # 用超级用户准备数据
        from rag_core.db import Database

        admin = Database(get_settings())
        tenant_a = tenant_b = None
        try:
            async with admin.transaction() as s:
                tenant_a, kb_a, user_a = await _seed_tenant_with_kb(s, label="A")
                tenant_b, kb_b, user_b = await _seed_tenant_with_kb(s, label="B")

            app_engine = _app_engine()
            factory = async_sessionmaker(app_engine, expire_on_commit=False)
            try:
                # 设 tenant=A 查询
                async with factory() as s, s.begin():
                    await s.execute(
                        text("SELECT set_config('app.tenant_id', :t, true)"),
                        {"t": str(tenant_a)},
                    )
                    count_a = int(
                        (await s.execute(text("SELECT count(*) FROM knowledge_bases"))).scalar_one()
                    )
                    own = await s.execute(
                        text("SELECT id FROM knowledge_bases WHERE id = :i"), {"i": kb_a}
                    )
                    own_visible = own.scalar_one_or_none() is not None
                    other = await s.execute(
                        text("SELECT id FROM knowledge_bases WHERE id = :i"), {"i": kb_b}
                    )
                    other_visible = other.scalar_one_or_none() is not None
                return count_a, int(own_visible), int(other_visible)
            finally:
                await app_engine.dispose()
        finally:
            async with admin.transaction() as s:
                if tenant_a:
                    await _cleanup(s, tenant_a, user_a)
                if tenant_b:
                    await _cleanup(s, tenant_b, user_b)
            await admin.close()

    own_count, sees_own, sees_other = asyncio.run(scenario())
    assert sees_own == 1, "本租户数据应该可见"
    assert sees_other == 0, "**跨租户数据泄漏！**"
    # 只算自己那一条（可能库里有其它残留，所以断言 >= 1 且不含 B）
    assert own_count >= 1


def test_no_tenant_context_returns_zero_rows() -> None:
    """未设置 app.tenant_id 时应返回 0 行，而不是报错或返回全部。

    **这是实测踩到的坑**：`current_setting('app.tenant_id', true)` 未设置时
    返回空字符串 `''`，直接 `::uuid` 会报
    `invalid input syntax for type uuid: ""`。
    策略里必须用 `NULLIF(..., '')` 处理。

    不处理的话，任何漏设上下文的请求都会 500 —— 而且错误信息
    完全指不到"忘记设 tenant_id"这个根因。
    """

    async def scenario() -> int:
        from rag_core.db import Database

        admin = Database(get_settings())
        tenant_id = None
        try:
            async with admin.transaction() as s:
                tenant_id, _, seeded_user = await _seed_tenant_with_kb(s, label="ZeroCtx")

            app_engine = _app_engine()
            factory = async_sessionmaker(app_engine, expire_on_commit=False)
            try:
                async with factory() as s, s.begin():
                    result = await s.execute(text("SELECT count(*) FROM knowledge_bases"))
                    return int(result.scalar_one())
            finally:
                await app_engine.dispose()
        finally:
            async with admin.transaction() as s:
                if tenant_id:
                    await _cleanup(s, tenant_id, seeded_user)
            await admin.close()

    assert asyncio.run(scenario()) == 0


# ==================================================== 写隔离


def test_app_role_cannot_insert_other_tenant_row() -> None:
    """设了 tenant=A 时，插入 tenant=B 的记录必须被拒绝。

    **实测确认的 PostgreSQL 行为**：策略只写 `USING` 不写 `WITH CHECK` 时，
    PG 会用 `USING` 表达式**同时**检查新行，因此跨租户 INSERT 仍会被拒。

    所以这条测试单独存在时**不能证明 `WITH CHECK` 的必要性** ——
    去掉 `WITH CHECK` 它照样通过（我实测验证过）。
    `WITH CHECK` 的真正价值体现在 UPDATE 场景，见下一个测试。
    """

    async def scenario() -> bool:
        from rag_core.db import Database

        admin = Database(get_settings())
        tenant_a = tenant_b = user_a = user_b = None
        try:
            async with admin.transaction() as s:
                tenant_a, _, user_a = await _seed_tenant_with_kb(s, label="InsA")
                tenant_b, _, user_b = await _seed_tenant_with_kb(s, label="InsB")

            app_engine = _app_engine()
            factory = async_sessionmaker(app_engine, expire_on_commit=False)
            try:
                async with factory() as s, s.begin():
                    # 用 tenant=A 的上下文，尝试往 tenant=B 插知识库
                    await s.execute(
                        text("SELECT set_config('app.tenant_id', :t, true)"),
                        {"t": str(tenant_a)},
                    )
                    # 用 savepoint 捕获错误，避免整个事务失效
                    try:
                        async with s.begin_nested():
                            await s.execute(
                                text(
                                    "INSERT INTO knowledge_bases (id, tenant_id, name, "
                                    "status, visibility, confidentiality, created_by, "
                                    "default_chunking_strategy, default_chunk_size, "
                                    "default_chunk_overlap, document_count, "
                                    "created_at, updated_at) VALUES (:id, :t, :n, "
                                    "'active', 'tenant', 'internal', :u, 'recursive', "
                                    "800, 100, 0, now(), now())"
                                ),
                                {
                                    "id": uuid.uuid4(),
                                    "t": tenant_b,  # 别的租户
                                    "n": "越权插入",
                                    "u": user_a,
                                },
                            )
                        return False  # 没报错 = 越权成功
                    except Exception:
                        return True  # 被 RLS 拒绝
            finally:
                await app_engine.dispose()
        finally:
            async with admin.transaction() as s:
                if tenant_a:
                    await _cleanup(s, tenant_a, user_a)
                if tenant_b:
                    await _cleanup(s, tenant_b, user_b)
            await admin.close()

    assert asyncio.run(scenario()) is True, "**跨租户写入未被拒绝！**"


def test_app_role_cannot_move_row_to_other_tenant() -> None:
    """**UPDATE 攻击：把自己租户的行改成别的租户。**

    验证的是"改完之后的新行必须仍属于当前租户"。

    **实测澄清**（我最初写这段测试时的假设是错的）：
    PG 在只有 `USING` 时，会用同一表达式检查 UPDATE 之后的新行，
    因此**即使没有 `WITH CHECK`，跨租户 UPDATE 也会被拒**
    （报"新行违背了行级安全策略"）。

    所以这条测试证明的是"租户归属不可变更"这个事实，
    而不是"`WITH CHECK` 必需"。`WITH CHECK` 的价值在于
    把写校验与读可见性**显式分离**，防止未来放宽 `USING` 时连带放宽写入。
    """

    async def scenario() -> bool:
        from rag_core.db import Database

        admin = Database(get_settings())
        tenant_a = tenant_b = kb_a = user_a = user_b = None
        try:
            async with admin.transaction() as s:
                tenant_a, kb_a, user_a = await _seed_tenant_with_kb(s, label="MoveA")
                tenant_b, _, user_b = await _seed_tenant_with_kb(s, label="MoveB")

            app_engine = _app_engine()
            factory = async_sessionmaker(app_engine, expire_on_commit=False)
            try:
                async with factory() as s, s.begin():
                    await s.execute(
                        text("SELECT set_config('app.tenant_id', :t, true)"),
                        {"t": str(tenant_a)},
                    )
                    try:
                        async with s.begin_nested():
                            await s.execute(
                                text(
                                    "UPDATE knowledge_bases SET tenant_id = :new_t WHERE id = :kb"
                                ),
                                {"new_t": tenant_b, "kb": kb_a},
                            )
                        return False  # 没报错 = 搬家成功 = 越权
                    except Exception:
                        return True
            finally:
                await app_engine.dispose()
        finally:
            async with admin.transaction() as s:
                if tenant_a:
                    await _cleanup(s, tenant_a, user_a)
                if tenant_b:
                    await _cleanup(s, tenant_b, user_b)
            await admin.close()

    assert asyncio.run(scenario()) is True, "**数据可被搬到其它租户（WITH CHECK 缺失）！**"


# ==================================================== 会话泄漏（连接池风险）


def test_tenant_context_does_not_leak_across_transactions() -> None:
    """**SET LOCAL 不能跨事务泄漏。**

    这是连接池场景下的真实越权风险：
    请求 1 设了 tenant=A 并归还连接；请求 2 复用了这条连接但没设 tenant_id。
    如果用会话级 `SET`，请求 2 会读到 A 的数据。

    用 `set_config(..., true)`（事务级）后，事务结束自动清除。
    """

    async def scenario() -> tuple[int, int]:
        from rag_core.db import Database

        admin = Database(get_settings())
        tenant_id = None
        try:
            async with admin.transaction() as s:
                tenant_id, _, seeded_user = await _seed_tenant_with_kb(s, label="Leak")

            app_engine = _app_engine()
            factory = async_sessionmaker(app_engine, expire_on_commit=False)
            try:
                async with factory() as s:
                    # 事务 1：设 tenant 并查询
                    async with s.begin():
                        await s.execute(
                            text("SELECT set_config('app.tenant_id', :t, true)"),
                            {"t": str(tenant_id)},
                        )
                        first = int(
                            (
                                await s.execute(text("SELECT count(*) FROM knowledge_bases"))
                            ).scalar_one()
                        )
                    # 事务 2：不设 tenant，应看不到任何数据
                    async with s.begin():
                        second = int(
                            (
                                await s.execute(text("SELECT count(*) FROM knowledge_bases"))
                            ).scalar_one()
                        )
                return first, second
            finally:
                await app_engine.dispose()
        finally:
            async with admin.transaction() as s:
                if tenant_id:
                    await _cleanup(s, tenant_id, seeded_user)
            await admin.close()

    first, second = asyncio.run(scenario())
    assert first >= 1, "第一个事务应能看到自己的数据"
    assert second == 0, f"**租户上下文跨事务泄漏！第二个事务看到 {second} 行**"


# ==================================================== 反证：超级用户绕过


def test_superuser_bypasses_rls() -> None:
    """**反证测试**：超级用户不受 RLS 约束。

    这条测试不是"验证功能正确"，而是**记录一个危险的部署前提**：
    如果应用用超级用户连库，上面所有 RLS 测试都失去意义
    —— 策略还在，但完全不起作用且没有任何提示。

    这也是为什么本项目要求：
    - 迁移用 puraz（超级用户，需要 DDL 权限）
    - 运行时用 rag_app（NOSUPERUSER NOBYPASSRLS）
    """

    async def scenario() -> int:
        from rag_core.db import Database

        admin = Database(get_settings())
        tenant_a = tenant_b = None
        try:
            async with admin.transaction() as s:
                tenant_a, _, user_a = await _seed_tenant_with_kb(s, label="SuperA")
                tenant_b, _, user_b = await _seed_tenant_with_kb(s, label="SuperB")

            # 用超级用户，设 tenant=A，但应该能看到两个租户的库
            async with admin.transaction() as s:
                await s.execute(
                    text("SELECT set_config('app.tenant_id', :t, true)"),
                    {"t": str(tenant_a)},
                )
                result = await s.execute(
                    text("SELECT count(*) FROM knowledge_bases WHERE tenant_id IN (:a, :b)"),
                    {"a": tenant_a, "b": tenant_b},
                )
                return int(result.scalar_one())
        finally:
            async with admin.transaction() as s:
                if tenant_a:
                    await _cleanup(s, tenant_a, user_a)
                if tenant_b:
                    await _cleanup(s, tenant_b, user_b)
            await admin.close()

    # 超级用户能看到两个租户的数据 —— 证明 RLS 被绕过
    assert asyncio.run(scenario()) == 2, (
        "超级用户未能绕过 RLS，说明 FORCE 策略对 superuser 生效了？"
        "（PG 文档说 superuser 始终绕过，若这里失败说明环境行为异常）"
    )


# ==================================================== 并发下隔离仍成立


def test_concurrent_queries_keep_isolation() -> None:
    """**并发下隔离不能失效**（压测路径的简化版）。

    高并发时连接池会频繁复用连接。如果 tenant 上下文的设置
    与查询之间存在竞态（比如用会话级 SET），就可能读到别人的数据。

    这里用 20 个并发任务交替查两个租户，每个任务都必须只看到自己的数据。
    """

    async def scenario() -> list[str]:
        from rag_core.db import Database

        admin = Database(get_settings())
        tenant_a = tenant_b = None
        try:
            async with admin.transaction() as s:
                tenant_a, kb_a, user_a = await _seed_tenant_with_kb(s, label="ConcA")
                tenant_b, kb_b, user_b = await _seed_tenant_with_kb(s, label="ConcB")

            app_engine = _app_engine()
            factory = async_sessionmaker(app_engine, expire_on_commit=False)
            failures: list[str] = []

            async def query(
                tenant_id: uuid.UUID, allowed_kb: uuid.UUID, other_kb: uuid.UUID
            ) -> None:
                async with factory() as s, s.begin():
                    await s.execute(
                        text("SELECT set_config('app.tenant_id', :t, true)"),
                        {"t": str(tenant_id)},
                    )
                    rows = await s.execute(text("SELECT id FROM knowledge_bases"))
                    visible = {str(row[0]) for row in rows}
                    if str(other_kb) in visible:
                        failures.append(f"tenant {tenant_id} 看到了 {other_kb}")
                    if str(allowed_kb) not in visible:
                        failures.append(f"tenant {tenant_id} 没看到自己的 {allowed_kb}")

            try:
                tasks = []
                for _ in range(10):
                    tasks.append(query(tenant_a, kb_a, kb_b))
                    tasks.append(query(tenant_b, kb_b, kb_a))
                await asyncio.gather(*tasks)
            finally:
                await app_engine.dispose()
            return failures
        finally:
            async with admin.transaction() as s:
                if tenant_a:
                    await _cleanup(s, tenant_a, user_a)
                if tenant_b:
                    await _cleanup(s, tenant_b, user_b)
            await admin.close()

    failures = asyncio.run(scenario())
    assert not failures, f"并发下发生隔离失效: {failures[:3]}"
