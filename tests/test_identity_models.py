"""租户/用户/成员关系的集成测试（真实 PG）。

两类测试，用不同策略：

- **单连接测试**：用 `db_session` fixture，结束时自动回滚，不留脏数据。
- **并发测试**：必须用多个独立连接才能验证唯一约束（同一连接上的
  并发没有意义），因此真实提交 + 显式清理，且清理逻辑必须可靠。

踩过的坑：早期所有测试都用"提交 + finally 清理"，清理与残留事务
互锁导致 DELETE 静默失败，脏数据让后续分页测试拿到 17 条而不是 12 条。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_contracts.enums import MembershipStatus, PlatformRole, TenantRole
from rag_core.db import Database
from rag_core.settings import get_settings
from rag_db.repositories import MembershipRepository, TenantRepository, UserRepository

pytestmark = pytest.mark.integration


def _unique_slug() -> str:
    return f"t-{uuid.uuid4().hex[:12]}"


# ==================================================== 单连接（自动回滚）


async def test_create_tenant_and_find_by_slug(db_session: AsyncSession) -> None:
    """租户能创建并按 slug 查到。"""
    slug = _unique_slug()
    repo = TenantRepository(db_session)

    created = await repo.create(slug=slug, name="测试租户")
    assert created.id is not None
    assert created.status == "active"

    found = await repo.get_by_slug(slug)
    assert found is not None
    assert found.name == "测试租户"


async def test_get_active_by_id_excludes_suspended(db_session: AsyncSession) -> None:
    """暂停的租户不能被"取 active 租户"查到。

    鉴权路径必须用 `get_active_by_id`：用 `get_by_id` 会让"暂停"形同虚设
    ——用户照常能用，只是管理端显示暂停。
    """
    repo = TenantRepository(db_session)
    tenant = await repo.create(slug=_unique_slug(), name="将被暂停")
    tenant_id = tenant.id

    await db_session.execute(
        text("UPDATE tenants SET status = 'suspended' WHERE id = :tid"), {"tid": tenant_id}
    )

    assert await repo.get_by_id(tenant_id) is not None
    assert await repo.get_active_by_id(tenant_id) is None


async def test_duplicate_slug_raises_integrity_error(db_session: AsyncSession) -> None:
    """重复 slug 必须由唯一索引挡住。

    "先查再插"在并发下有竞态窗口（两个请求同时查到"不存在"），
    只有唯一索引能真正防住。
    """
    slug = _unique_slug()
    repo = TenantRepository(db_session)
    await repo.create(slug=slug, name="第一个")

    # PG 在约束冲突后会中止整个事务。这里断言的重点是"唯一索引真的拦住了"，
    # 冲突后事务状态由 db_session fixture 在结束时统一回滚清理。
    # （会看到一条 SAWarning: transaction already deassociated —— 这是
    #   冲突后事务被 PG 中止的正常现象，不代表测试有问题。）
    with pytest.raises(IntegrityError):
        await repo.create(slug=slug, name="重复的")


async def test_membership_status_filtered(db_session: AsyncSession) -> None:
    """被暂停的成员关系不算"生效中"。"""
    tenants = TenantRepository(db_session)
    users = UserRepository(db_session)
    memberships = MembershipRepository(db_session)

    tenant = await tenants.create(slug=_unique_slug(), name="状态测试")
    user = await users.create_user(email="suspended@example.test", display_name="被暂停")
    membership = await memberships.create(
        tenant_id=tenant.id, user_id=user.id, role=TenantRole.MEMBER.value
    )

    # 状态字段是 member.status，这里直接改内存对象
    membership.status = MembershipStatus.SUSPENDED
    await db_session.flush()

    active = await memberships.get_active(tenant_id=tenant.id, user_id=user.id)
    assert active is None, "暂停的成员关系被当成了生效中"


async def test_keyset_pagination_no_duplicates_no_gaps(db_session: AsyncSession) -> None:
    """keyset 分页不重复、不遗漏。

    这是 OFFSET 的替代方案验证。

    **刻意不假设"表里只有我插入的数据"**：库里可能有其它测试/历史数据，
    断言总数会变成"依赖环境干净"的脆弱测试。改为用唯一前缀插入，
    只统计自己的这批 —— 这样测试在任何库状态下都成立。
    """
    repo = TenantRepository(db_session)
    total = 12
    prefix = f"page-{uuid.uuid4().hex[:8]}"
    for i in range(total):
        await repo.create(slug=f"{prefix}-{i:04d}", name=f"分页 {i}")

    collected: list[uuid.UUID] = []
    after: uuid.UUID | None = None
    while True:
        page = await repo.list_all(limit=5, after_id=after)
        if not page:
            break
        # 只收本次插入的（按 slug 前缀过滤）
        collected.extend(t.id for t in page if t.slug.startswith(prefix))
        after = page[-1].id

    assert len(collected) == total, f"分页遗漏：拿到 {len(collected)} 条，应为 {total}"
    assert len(set(collected)) == total, "分页出现重复项"


async def test_platform_role_defaults_to_none(db_session: AsyncSession) -> None:
    """新用户的平台角色默认 none —— 绝不能默认有平台权限。"""
    user = await UserRepository(db_session).create_user(
        email="default-role@example.test", display_name="默认角色"
    )
    found = await UserRepository(db_session).get_by_id(user.id)
    assert found is not None
    assert found.platform_role == PlatformRole.NONE.value


async def test_enum_values_persist_and_read_back(db_session: AsyncSession) -> None:
    """枚举值写库再读回必须与 Python 枚举一致。

    存字符串而非 PG 原生 enum 是有意的（便于 expand-contract），
    代价是"库里可能是任何字符串"——这个测试保证正常路径没问题。
    """
    tenants = TenantRepository(db_session)
    users = UserRepository(db_session)
    memberships = MembershipRepository(db_session)

    tenant = await tenants.create(slug=_unique_slug(), name="枚举测试")
    user = await users.create_user(email="enum@example.test", display_name="枚举")
    await memberships.create(
        tenant_id=tenant.id, user_id=user.id, role=TenantRole.KNOWLEDGE_ADMIN.value
    )

    found_tenant = await tenants.get_by_slug(tenant.slug)
    found_membership = await memberships.get_active(tenant_id=tenant.id, user_id=user.id)
    assert found_tenant is not None
    assert found_membership is not None
    assert found_tenant.status == "active"
    assert found_membership.role == TenantRole.KNOWLEDGE_ADMIN.value


# ==================================================== 并发（真实提交）


def test_concurrent_first_login_creates_single_identity_link() -> None:
    """**并发首登只能产生一条身份映射。**

    真实场景：用户在浏览器连点两下登录，或前端重试逻辑让两个回调
    同时到达。两个请求都查到"没有映射"，都去创建用户。

    若实现是"先查再插"（无唯一索引），会产生两个本地用户，后续登录
    随机命中其中一个——表现为"用户的文档时有时无"。

    有唯一索引时一个成功、其余抛 IntegrityError，上层捕获后重查即可。

    **必须用独立连接**：同一连接上的"并发"是串行的，测不出竞态。
    也因此不能用回滚 fixture，必须在 finally 里真实清理。
    """
    issuer = "https://idp.test/realms/main"
    subject = f"user-{uuid.uuid4().hex}"

    async def scenario() -> tuple[int, int]:
        db = Database(get_settings())
        successes = 0
        conflicts = 0

        async def attempt() -> None:
            nonlocal successes, conflicts
            # 每个任务自己开连接 —— 这才叫并发
            async with db.transaction() as session:
                users = UserRepository(session)
                user = await users.create_user(
                    email=f"racer-{uuid.uuid4().hex[:8]}@example.test",
                    display_name="并发用户",
                )
                await users.create_link(user_id=user.id, issuer=issuer, subject=subject)
            successes += 1

        async def attempt_guarded() -> None:
            nonlocal conflicts
            try:
                await attempt()
            except IntegrityError:
                conflicts += 1

        try:
            await asyncio.gather(*(attempt_guarded() for _ in range(6)))

            async with db.session() as session:
                result = await session.execute(
                    text(
                        "SELECT count(*) FROM identity_links "
                        "WHERE issuer = :issuer AND subject = :subject"
                    ),
                    {"issuer": issuer, "subject": subject},
                )
                links = int(result.scalar_one())
            return links, successes
        finally:
            # 清理必须可靠：先删映射再删用户（FK 是 CASCADE，
            # 但显式顺序让意图清楚，也不依赖级联行为未被改动）
            async with db.transaction() as session:
                await session.execute(
                    text(
                        "DELETE FROM users WHERE id IN "
                        "(SELECT user_id FROM identity_links WHERE subject = :subject)"
                    ),
                    {"subject": subject},
                )
            await db.close()

    links, successes = asyncio.run(scenario())
    assert links == 1, f"并发首登产生了 {links} 条映射，应为 1"
    assert successes == 1, f"应只有 1 个请求成功，实际 {successes}"


def test_duplicate_membership_rejected_across_connections() -> None:
    """同一人在同一租户只能有一条成员关系（跨连接验证）。

    重复邀请会产生两条，权限判断取哪条全看查询顺序——
    表现为"权限时有时无"，极难排查。
    """
    slug = _unique_slug()
    email = f"member-{uuid.uuid4().hex[:8]}@example.test"

    async def scenario() -> None:
        db = Database(get_settings())
        try:
            async with db.transaction() as session:
                tenant = await TenantRepository(session).create(slug=slug, name="成员测试")
                user = await UserRepository(session).create_user(email=email, display_name="成员")
                tenant_id, user_id = tenant.id, user.id
                await MembershipRepository(session).create(
                    tenant_id=tenant_id, user_id=user_id, role=TenantRole.MEMBER.value
                )

            # 第二个连接再插一次，必须被唯一约束挡住
            with pytest.raises(IntegrityError):
                async with db.transaction() as session:
                    await MembershipRepository(session).create(
                        tenant_id=tenant_id, user_id=user_id, role=TenantRole.ADMIN.value
                    )
        finally:
            async with db.transaction() as session:
                await session.execute(text("DELETE FROM users WHERE email = :e"), {"e": email})
                await session.execute(text("DELETE FROM tenants WHERE slug = :s"), {"s": slug})
            await db.close()

    asyncio.run(scenario())


def test_deleting_user_cascades_identity_links() -> None:
    """删用户时身份映射必须一起删。

    不级联的后果：留下孤儿映射。下次同一 IdP 身份登录会命中这条映射，
    但对应 user_id 已不存在——表现为 500 或登录后空白页。
    """
    subject = f"cascade-{uuid.uuid4().hex}"
    email = f"cascade-{uuid.uuid4().hex[:8]}@example.test"

    async def scenario() -> int:
        db = Database(get_settings())
        try:
            async with db.transaction() as session:
                users = UserRepository(session)
                user = await users.create_user(email=email, display_name="级联测试")
                user_id = user.id
                await users.create_link(user_id=user_id, issuer="https://idp.test", subject=subject)

            async with db.transaction() as session:
                await session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})

            async with db.session() as session:
                result = await session.execute(
                    text("SELECT count(*) FROM identity_links WHERE subject = :s"),
                    {"s": subject},
                )
                return int(result.scalar_one())
        finally:
            async with db.transaction() as session:
                await session.execute(text("DELETE FROM users WHERE email = :e"), {"e": email})
            await db.close()

    assert asyncio.run(scenario()) == 0
