"""对话链路测试（A16）。

## 三层

1. **纯单测**：历史裁剪、会话标题、usage 提取等纯函数；
2. **集成测（真实 PG）**：序号分配的并发安全、跨用户隔离、软删；
3. **端到端（真实 LLM）**：单轮问答、多轮独立化改写、拒答。

## 为什么历史裁剪要单独测

多轮对话最隐蔽的 bug 是"历史不平整"：丢掉一条 `user` 只留 `assistant`，
模型看到"凭空出现的回答"，会开始瞎猜或复述。

裁剪必须**成对丢弃**（2 条 = 1 轮），这个约束只能用测试钉死。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from rag_api.services.chat import (
    MAX_HISTORY_CHARS,
    _extract_usage,
    _make_title,
    _trim_history,
)
from rag_contracts.enums.conversations import (
    FeedbackReason,
    MessageRole,
    MessageStatus,
    RefusalReason,
)

# ==================================================== 纯单测：历史裁剪


def test_trim_history_keeps_recent_turns() -> None:
    """超预算时保留最近的轮次（从最旧开始丢）。"""
    # 构造 5 轮，每轮 ~1000 字符 → 总 10000，超过预算 4000
    history: list[tuple[str, str]] = []
    for index in range(5):
        history.append(("user", f"问题{index}" + "x" * 1000))
        history.append(("assistant", f"回答{index}" + "y" * 1000))

    trimmed = _trim_history(history)

    total = sum(len(content) for _role, content in trimmed)
    assert total <= MAX_HISTORY_CHARS
    # 保留的是**最近**的
    assert "问题4" in trimmed[-2][1]
    # 最早被丢掉
    assert all("问题0" not in content for _role, content in trimmed)


def test_trim_history_drops_in_pairs() -> None:
    """**必须成对丢弃**：留下孤立的 assistant 会让模型看到"凭空出现的回答"。

    20 轮前的"你好"占预算却无信息量；但只丢 user 而留 assistant
    会造成对话结构破损，模型会开始复述或瞎猜。
    """
    history: list[tuple[str, str]] = []
    for index in range(10):
        history.append(("user", f"q{index}" + "x" * 300))
        history.append(("assistant", f"a{index}" + "y" * 300))

    trimmed = _trim_history(history)

    # 条数必须是偶数（成对）
    assert len(trimmed) % 2 == 0, f"裁剪后条数为奇数：{len(trimmed)}"
    # 且必须以 user 开头、assistant 结尾（结构完整）
    if trimmed:
        assert trimmed[0][0] == "user"
        assert trimmed[-1][0] == "assistant"


def test_trim_history_keeps_short_history_intact() -> None:
    """预算内不裁剪。"""
    history = [("user", "短问题"), ("assistant", "短回答")]
    assert _trim_history(history) == history


def test_trim_history_handles_empty() -> None:
    assert _trim_history([]) == []


def test_make_title_truncates_and_normalizes() -> None:
    """标题取前 30 字，并把连续空白压成单空格。"""
    long_query = "  一线城市   出差住宿\n每晚最多能报销多少？还有别的规定吗  "
    title = _make_title(long_query)

    assert len(title) <= 30
    assert "\n" not in title
    assert "  " not in title


def test_make_title_falls_back() -> None:
    """全空白时用默认标题（不能返回空串）。"""
    assert _make_title("   ") == "新会话"


def test_extract_usage_missing_logs_and_zeros() -> None:
    """流式没有 usage 时返回 (0,0)（不能崩）。

    这是**已知缺陷的信号**：成本会记 0，对账会显示"有调用无成本"。
    函数本身不抛异常（问答已完成，不能因为记账失败而失败），
    但会打 warning（见实现）。
    """
    assert _extract_usage([]) == (0, 0)


def test_extract_usage_takes_last() -> None:
    """多个 usage 回调时取最后一个（LangChain 的累计语义）。"""

    class _Usage:
        def __init__(self, input_tokens: int, output_tokens: int) -> None:
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    holder = [_Usage(10, 5), _Usage(426, 33)]
    assert _extract_usage(holder) == (426, 33)


# ==================================================== 枚举契约


def test_message_status_covers_partial_generation() -> None:
    """消息状态必须能表达"生成到一半停了"。

    只有 success/failed 两态时，"用户看到了部分内容但没写完"
    无法记录——而那恰恰是最需要记录的情况（前端要显示"回答中断"）。
    """
    values = {status.value for status in MessageStatus}
    assert "interrupted" in values
    assert "streaming" in values
    assert "failed" in values
    assert "completed" in values


def test_refusal_requires_reason() -> None:
    """拒答必须有原因（用于统计"为什么拒答率上升"）。"""
    values = {reason.value for reason in RefusalReason}
    assert values == {"no_materials", "insufficient_evidence", "prompt_injection_suspected"}


def test_feedback_reason_is_enumerable() -> None:
    """点踩原因用固定枚举（自由文本无法聚合统计）。"""
    assert FeedbackReason.NOT_RELEVANT.value == "not_relevant"
    assert len(list(FeedbackReason)) >= 5


# ==================================================== 集成：仓储行为


@pytest.fixture
async def db_factory() -> object:
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


async def _seed_user(factory: object) -> tuple[uuid.UUID, uuid.UUID]:
    """建租户 + 用户，返回 (tenant_id, user_id)。"""
    from sqlalchemy import text

    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    async with factory() as session:  # type: ignore[operator]
        await session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "created_at, updated_at) VALUES (:id, :e, 'chat', 'active', 'none', "
                "now(), now())"
            ),
            {"id": user_id, "e": f"chat-{user_id.hex[:8]}@test.local"},
        )
        await session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:id, :s, 'chat', 'active', now(), now())"
            ),
            {"id": tenant_id, "s": f"chat-{tenant_id.hex[:8]}"},
        )
        await session.commit()
    return tenant_id, user_id


async def _cleanup(factory: object, tenant_id: uuid.UUID) -> None:
    from sqlalchemy import text

    async with factory() as session:  # type: ignore[operator]
        await session.execute(text("DELETE FROM messages WHERE tenant_id = :t"), {"t": tenant_id})
        await session.execute(
            text("DELETE FROM conversations WHERE tenant_id = :t"), {"t": tenant_id}
        )
        await session.execute(text("DELETE FROM tenants WHERE id = :t"), {"t": tenant_id})
        await session.execute(text("DELETE FROM users WHERE email LIKE 'chat-%@test.local'"))
        await session.commit()


@pytest.mark.integration
async def test_sequence_allocation_is_atomic(db_factory: object) -> None:
    """**并发分配序号必须无重复**。

    朴素实现（`max(sequence) + 1`）在并发下会两个请求拿到同一个号，
    撞唯一约束。本实现用 `UPDATE ... RETURNING message_count` 原子递增。
    """
    from sqlalchemy import text

    from rag_db.repositories import ConversationRepository

    tenant_id, user_id = await _seed_user(db_factory)
    try:
        async with db_factory() as session:  # type: ignore[operator]
            conv = await ConversationRepository(session).create(
                tenant_id=tenant_id, user_id=user_id
            )
            conv_id = conv.id
            await session.commit()

        async def allocate() -> int:
            async with db_factory() as session:  # type: ignore[operator]
                seq = await ConversationRepository(session).allocate_sequence(
                    conversation_id=conv_id, tenant_id=tenant_id
                )
                await session.commit()
                return seq

        # 20 个并发分配
        results = await asyncio.gather(*[allocate() for _ in range(20)])

        # 必须全不重复
        assert len(set(results)) == 20, f"序号重复：{sorted(results)}"
        # 且是 1..20
        assert sorted(results) == list(range(1, 21))

        # message_count 与实际一致
        async with db_factory() as session:  # type: ignore[operator]
            row = await session.execute(
                text("SELECT message_count FROM conversations WHERE id = :c"),
                {"c": conv_id},
            )
            assert int(row.scalar_one()) == 20
    finally:
        await _cleanup(db_factory, tenant_id)


@pytest.mark.integration
async def test_conversation_is_private_to_user(db_factory: object) -> None:
    """**安全关键**：会话必须按 user_id 隔离，不只是 tenant_id。

    RLS 只保证租户隔离。同一租户下 A 用户读到 B 用户的对话是
    隐私事故，必须由仓储层额外过滤。
    """
    from rag_db.repositories import ConversationRepository

    tenant_id, user_a = await _seed_user(db_factory)
    user_b = uuid.uuid4()
    try:
        from sqlalchemy import text

        async with db_factory() as session:  # type: ignore[operator]
            await session.execute(
                text(
                    "INSERT INTO users (id, email, display_name, status, platform_role, "
                    "created_at, updated_at) VALUES (:id, :e, 'b', 'active', 'none', "
                    "now(), now())"
                ),
                {"id": user_b, "e": f"chat-{user_b.hex[:8]}@test.local"},
            )
            conv = await ConversationRepository(session).create(
                tenant_id=tenant_id, user_id=user_a, title="A 的私密会话"
            )
            conv_id = conv.id
            await session.commit()

        # B 在同一租户下查不到 A 的会话
        async with db_factory() as session:  # type: ignore[operator]
            found = await ConversationRepository(session).get(
                tenant_id=tenant_id, conversation_id=conv_id, user_id=user_b
            )
            assert found is None, "跨用户读到了别人的会话"

        # A 自己能查到
        async with db_factory() as session:  # type: ignore[operator]
            found = await ConversationRepository(session).get(
                tenant_id=tenant_id, conversation_id=conv_id, user_id=user_a
            )
            assert found is not None
    finally:
        await _cleanup(db_factory, tenant_id)


@pytest.mark.integration
async def test_soft_delete_hides_conversation(db_factory: object) -> None:
    """软删后查不到，但数据仍在（审计需要）。"""
    from sqlalchemy import text

    from rag_db.repositories import ConversationRepository

    tenant_id, user_id = await _seed_user(db_factory)
    try:
        async with db_factory() as session:  # type: ignore[operator]
            repo = ConversationRepository(session)
            conv = await repo.create(tenant_id=tenant_id, user_id=user_id)
            conv_id = conv.id
            await session.commit()

        async with db_factory() as session:  # type: ignore[operator]
            repo = ConversationRepository(session)
            assert await repo.soft_delete(conversation_id=conv_id)
            await session.commit()

        async with db_factory() as session:  # type: ignore[operator]
            repo = ConversationRepository(session)
            # 默认查不到
            assert (
                await repo.get(tenant_id=tenant_id, conversation_id=conv_id, user_id=user_id)
                is None
            )
            # 但数据仍在
            row = await session.execute(
                text("SELECT deleted_at FROM conversations WHERE id = :c"),
                {"c": conv_id},
            )
            assert row.scalar_one() is not None
    finally:
        await _cleanup(db_factory, tenant_id)


@pytest.mark.integration
async def test_streaming_message_reset_by_cleanup(db_factory: object) -> None:
    """滞留的 streaming 消息会被清理任务标成 interrupted。

    SSE 断连/进程崩溃时没机会改状态，消息会永远停在 streaming
    ——历史里一条永远转圈的回答。
    """
    from sqlalchemy import text

    from rag_db.repositories import ConversationRepository, MessageRepository

    tenant_id, user_id = await _seed_user(db_factory)
    try:
        async with db_factory() as session:  # type: ignore[operator]
            conv = await ConversationRepository(session).create(
                tenant_id=tenant_id, user_id=user_id
            )
            conv_id = conv.id
            seq = await ConversationRepository(session).allocate_sequence(
                conversation_id=conv_id, tenant_id=tenant_id
            )
            msg = await MessageRepository(session).add(
                tenant_id=tenant_id,
                conversation_id=conv_id,
                sequence=seq,
                role=MessageRole.ASSISTANT,
                status=MessageStatus.STREAMING,
            )
            msg_id = msg.id
            # 把 created_at 推到很久以前
            await session.execute(
                text("UPDATE messages SET created_at = now() - interval '1 hour' WHERE id = :m"),
                {"m": msg_id},
            )
            await session.commit()

        async with db_factory() as session:  # type: ignore[operator]
            reset = await MessageRepository(session).reset_stale_streaming(older_than_seconds=60)
            await session.commit()
            assert reset == 1

        async with db_factory() as session:  # type: ignore[operator]
            row = await session.execute(
                text("SELECT status FROM messages WHERE id = :m"), {"m": msg_id}
            )
            assert row.scalar_one() == MessageStatus.INTERRUPTED.value
    finally:
        await _cleanup(db_factory, tenant_id)


@pytest.mark.integration
async def test_refusal_requires_reason_constraint(db_factory: object) -> None:
    """数据库层强制"拒答必须带原因"（不是仅靠应用层检查）。"""
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    from rag_db.repositories import ConversationRepository

    tenant_id, user_id = await _seed_user(db_factory)
    try:
        async with db_factory() as session:  # type: ignore[operator]
            conv = await ConversationRepository(session).create(
                tenant_id=tenant_id, user_id=user_id
            )
            await session.commit()
            conv_id = conv.id

        # 直接插一条 refused=true 但没 reason 的消息 → 应被 CHECK 拒绝
        async with db_factory() as session:  # type: ignore[operator]
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        "INSERT INTO messages (id, tenant_id, conversation_id, sequence, "
                        "role, content, citations, retrieval_meta, status, refused, "
                        "input_tokens, output_tokens, cost_usd, created_at, updated_at) "
                        "VALUES (gen_random_uuid(), :t, :c, 1, 'assistant', '', "
                        "'[]'::jsonb, '{}'::jsonb, 'completed', true, 0, 0, 0, "
                        "now(), now())"
                    ),
                    {"t": tenant_id, "c": conv_id},
                )
                await session.commit()
            await session.rollback()
    finally:
        await _cleanup(db_factory, tenant_id)


# ==================================================== 端到端（真实 LLM）


@pytest.mark.slow
@pytest.mark.integration
async def test_multi_turn_standalone_rewrite() -> None:
    """**多轮 RAG 的核心验证**：追问的指代必须被独立化。

    第 1 轮问"一线城市"，第 2 轮问"那二线呢？"——
    后者单独检索召回为零，必须靠历史改写。
    """
    from sqlalchemy import text

    from rag_adapters.providers.factory import (
        build_embedding_provider,
        build_llm_provider,
    )
    from rag_api.services.chat import ChatService
    from rag_core.db import Database
    from rag_core.settings import get_settings, load_env_files

    load_env_files()
    settings = get_settings()
    db = Database(settings)
    llm = build_llm_provider(settings)
    embedder = build_embedding_provider(settings)

    try:
        async with db.transaction_bypass_rls() as session:
            row = await session.execute(text("SELECT id FROM tenants WHERE slug = 'eval-tenant'"))
            tenant_id = row.scalar_one_or_none()
            if tenant_id is None:
                pytest.skip("评估库未建（先跑 tests/eval/run_eval.py --setup）")
            row = await session.execute(
                text("SELECT id FROM users WHERE email = 'eval@eval.local' LIMIT 1")
            )
            user_id = row.scalar_one_or_none()
            if user_id is None:
                pytest.skip("评估用户未建（先跑 tests/eval/run_eval.py --setup）")
            row = await session.execute(
                text("SELECT id FROM knowledge_bases WHERE tenant_id = :t LIMIT 1"),
                {"t": tenant_id},
            )
            kb_id = row.scalar_one_or_none()
            if kb_id is None:
                # **三个前置数据都要检查**，不能只查第一个。
                #
                # 初版只检查 tenant 存在就继续，于是"租户在但知识库被删了"
                # 会抛 `NoResultFound`——看起来像代码 bug，实际是环境不完整。
                # 环境依赖型测试必须把全部前置条件都断言到，缺任何一个都 skip。
                pytest.skip("评估知识库未建（先跑 tests/eval/run_eval.py --setup）")

        async def ask(query: str, conversation_id: uuid.UUID | None) -> tuple[uuid.UUID, str]:
            async with db.transaction(tenant_id=tenant_id) as session:
                service = ChatService(session, llm=llm, embedder=embedder, settings=settings)
                answer_parts: list[str] = []
                conv_id = conversation_id
                async for event in service.chat_stream(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    query=query,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ):
                    if event.event == "delta":
                        answer_parts.append(str(event.data.get("text", "")))
                    elif event.event == "done":
                        conv_id = uuid.UUID(str(event.data["conversation_id"]))
                assert conv_id is not None
                return conv_id, "".join(answer_parts)

        # 第 1 轮
        conv_id, answer1 = await ask("一线城市出差住宿每晚最多能报销多少？", None)
        assert "600" in answer1, f"第 1 轮应提到 600 元：{answer1[:100]}"

        # 第 2 轮（追问）
        _, answer2 = await ask("那二线呢？", conv_id)
        # 二线城市是 400 元 —— 能答出来说明独立化生效了
        assert "400" in answer2, f"追问未正确消解（应提到 400 元）：{answer2[:150]}"

        # 验证独立化结果落库
        async with db.transaction(tenant_id=tenant_id) as session:
            row = await session.execute(
                text(
                    "SELECT standalone_query FROM messages "
                    "WHERE conversation_id = :c AND role = 'user' "
                    "ORDER BY sequence DESC LIMIT 1"
                ),
                {"c": conv_id},
            )
            standalone = row.scalar_one()
            # 独立化后的问题应包含"二线"（原问题只有"那二线呢"）
            assert "二线" in standalone
            assert len(standalone) > len("那二线呢？")

        # 清理测试数据
        async with db.transaction_bypass_rls() as session:
            await session.execute(
                text("DELETE FROM messages WHERE conversation_id = :c"), {"c": conv_id}
            )
            await session.execute(text("DELETE FROM conversations WHERE id = :c"), {"c": conv_id})
    finally:
        await llm.aclose()
        await embedder.aclose()
        await db.close()


# ==================================================== 契约：反馈请求体


def test_feedback_request_parses_json_body() -> None:
    """反馈必须是请求体模型，不能是散装标量参数。

    ## 为什么这条测试值得单写

    初版把 `rating` / `reason` / `comment` 写成路由函数的三个标量参数。
    FastAPI 对**标量参数默认按 query 解析**，前端按 REST 惯例 POST JSON body：

        POST /api/v1/messages/{id}/feedback
        {"rating": "like"}

    服务端去 `?rating=` 里找 → 422 `{"location":["query","rating"]}`。

    这个 bug 在单元测试里不会暴露（TestClient 也能直接传 query），
    只有端到端点一次点赞才看得到：**用户点了没反应，图标永远不变**。

    所以这里钉两件事：
    1. `FeedbackRequest` 能从 JSON body 解析（字段名与前端一致）；
    2. `extra="forbid"`——前端多发字段要报错，而不是静默接受。
    """
    from pydantic import ValidationError

    from rag_contracts.api.chat import FeedbackRequest

    # 点赞：不需要 reason
    like = FeedbackRequest.model_validate({"rating": "like"})
    assert like.rating.value == "like"
    assert like.reason is None

    # 点踩：必须带 reason（业务校验在路由层做，这里只验形状可承载）
    dislike = FeedbackRequest.model_validate({"rating": "dislike", "reason": "not_relevant"})
    assert dislike.rating.value == "dislike"
    assert dislike.reason is not None

    # 未知字段必须报错：静默忽略会让前端拼错字段名后毫无反馈
    with pytest.raises(ValidationError):
        FeedbackRequest.model_validate({"rating": "like", "ratings": "typo"})

    # 超过 1000 字符的评论要在契约层拦住（不靠业务层记得检查）
    with pytest.raises(ValidationError):
        FeedbackRequest.model_validate({"rating": "like", "comment": "x" * 1001})
