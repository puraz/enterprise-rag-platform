"""集成测试共享 fixture：自动回滚的事务。

为什么不每个测试自己写清理：
- 清理逻辑要按 FK 依赖顺序写，漏一张表就留脏数据；
- 清理本身可能失败（锁冲突），而 `finally` 里抛异常会掩盖真实失败原因；
- 残留数据会让**后续测试**出现莫名其妙的失败（我实际踩到了：分页测试拿到 17 条）。

本项目踩到的具体坑：测试用 `db.transaction()` 提交后，清理又开一个
`db.transaction()`，但外层 `db.session()` 的连接仍持有未提交事务，
两个事务互相等待，DELETE 静默失败，数据留在库里。

方案：外层包一个事务，测试内部所有操作都在同一条连接上，
结束时统一 ROLLBACK。代价是测试无法验证"真的提交了"（那种场景单独写，
不用这个 fixture）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_core.db import Database
from rag_core.settings import get_settings


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """提供一个测试后自动回滚的会话。

    用法：
        async def test_xxx(db_session: AsyncSession) -> None:
            repo = TenantRepository(db_session)
            await repo.create(slug="t1", name="测试")
            # 测试结束自动回滚，库里不留数据

    注意：这里的 `flush()` 会写库，但不会提交；回滚后一切消失。
    需要"证明提交真的生效"的测试（如事务提交/回滚语义本身）
    不要用这个 fixture，自己管连接。
    """
    db = Database(get_settings())
    connection = await db.engine.connect()
    transaction = await connection.begin()
    session = AsyncSession(bind=connection, expire_on_commit=False, autoflush=False)
    try:
        yield session
    finally:
        await session.close()
        # 回滚整个外层事务，测试期间的所有写入（含 flush 的）一并消失
        await transaction.rollback()
        await connection.close()
        await db.close()
