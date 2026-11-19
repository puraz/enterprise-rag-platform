"""引用事件语义测试：**引用在生成后组装，不在检索后先发**。

## 这批测试钉住的是一个已修复的设计错误

初版在 LLM 生成**之前**就发 `retrieval` 事件，payload 是"全部检索命中"的
`Citation` 列表。前端于是立刻显示「来源 1 2 3 4 5」，而有三个问题：

1. **编号对不上**：事件里的 index 按材料顺序编（1..N），答案里的 `[n]` 是
   模型自己写的。模型先引用第 2 条材料时，答案写 `[1]`，引用卡片 1 却
   指向第 1 条材料——用户点开看到**错的原文**；
2. **数量对不上**：检索 5 条 ≠ 引用 5 条。答案只用 1 条时另外 4 条是噪音；
3. **时机不对**：生成前根本不知道模型会引用什么。

修法：`retrieval` 只发 `sources_found`（数量），引用由 `assemble_citations`
在生成结束后按答案里的 `[n]` 组装。

## 为什么这些测试必须存在

这个 bug **没有任何报错**：接口返回 200、流正常结束、引用也"有内容"。
只有把事件序列摊开、逐字段核对，或者两个角色对同一问题检索并对比，
才能发现。所以测试直接断言事件的**形状**，而不只是"能跑通"。

## 测试策略

用 fake LLM（返回可控文本）+ fake embedder（返回固定向量）+ 真实 PG。
不用真模型：需要控制"答案里出现哪个 `[n]`"，真模型做不到。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from sqlalchemy import text

from rag_adapters.providers.embedding import (
    EmbeddingCapabilities,
    EmbeddingProvider,
    EmbeddingResult,
    EmbeddingTask,
)
from rag_adapters.providers.llm import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ProviderCapabilities,
    TokenUsage,
)
from rag_adapters.retrieval.keyword import tokenize_for_index
from rag_adapters.retrieval.state import RetrievalConfig
from rag_api.services.chat import ChatService
from rag_contracts.enums import TenantRole

pytestmark = pytest.mark.integration

_DIM = 1024


# ---------------------------------------------------------------- fakes


class _FakeEmbedder(EmbeddingProvider):
    """固定向量。

    继承 `EmbeddingProvider` 而不是鸭子类型：`ChatService` 的构造函数
    类型标注是 `EmbeddingProvider`，鸭子类型会让 mypy 报 incompatible
    （真实世界里把类型注解当摆设，是为了"跑得起来"而牺牲了检查）。

    检索命中哪几条由数据决定，向量本身只需可用。
    """

    name = "fake-embed"

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(dimension=_DIM)

    async def embed_batch(
        self, texts: list[str], *, task: EmbeddingTask = EmbeddingTask.DOCUMENT
    ) -> EmbeddingResult:
        self.calls.extend(texts)
        vectors: list[list[float]] = []
        for _ in texts:
            vector = [0.0] * _DIM
            vector[0] = 1.0
            vectors.append(vector)
        return EmbeddingResult(vectors=vectors, dimension=_DIM, model=self.name)


class _FakeLLM(LLMProvider):
    """按预设文本逐块产出，用于精确控制答案里的 `[n]` 标记。"""

    name = "fake"
    model_name = "fake-model"

    def __init__(self, answer_chunks: list[str]) -> None:
        self._chunks = answer_chunks
        self.prompts: list[list[LLMMessage]] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_stream=True,
            supports_json_mode=False,
            max_context=8000,
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        timeout_s: float | None = None,
    ) -> LLMResponse:  # pragma: no cover - 本测试走 stream
        raise NotImplementedError

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
        usage_callback: Any = None,
    ) -> AsyncIterator[str]:
        self.prompts.append(messages)
        for chunk in self._chunks:
            yield chunk
        if usage_callback is not None:
            # `total_tokens` 是 property（= input + output），不能作为构造参数
            usage_callback(TokenUsage(input_tokens=10, output_tokens=5, cached_tokens=0))


class _FakeSettings:
    """ChatService 的 settings 替身。

    ## 为什么不能只手写一个 `llm`

    初版只有 `llm = _LLM()`（当时 ChatService 只用它查单价）。后来
    `_retrieve` 开始记录 embedding / reranker 的成本，于是需要
    `settings.embedding.provider`、`settings.embedding.model_name`、
    `settings.embedding.unit_price_input` 等字段——
    这些用例全部报 `'_FakeSettings' object has no attribute 'embedding'`，
    而栈顶指向业务代码，看不出是测试替身漏了字段。

    修法不是把它们一个个手写出来（下次再加一个字段又会坏），
    而是直接用**真实的子配置对象**：它们都是纯 Pydantic 模型，
    不连库不连网，构造零成本。这样 ChatService 用到任何字段都有值，
    且默认值与生产一致（成本记 0 是因为 unit_price 默认 None，
    正是这些测试想要的语义）。
    """

    from rag_core.settings import EmbeddingSettings, LLMSettings, RerankerSettings

    # llm 用替身：真实 LLMSettings 的 pricing_for 会查单价表，
    # 而这些测试要的是"查不到单价 → 成本记 0"的确定行为。
    class _LLM:
        @staticmethod
        def pricing_for(_model: str) -> None:
            return None

    llm = _LLM()
    # 这两个用真实配置对象：字段齐全，且 unit_price_* 默认 None（成本记 0）
    embedding = EmbeddingSettings()
    reranker = RerankerSettings()


class _FakeSelfCheckLLM(_FakeLLM):
    """支持 `complete` 的 fake：第一次回答 answerability，第二次回答 support。

    调用顺序是确定的：
        1. `_judge_answerability`（生成前）→ `{"can_answer": ...}`
        2. `_judge_support`（生成后）→ `{"supported": ...}`
    用调用次数区分，避免在 fake 里猜提示词内容。
    """

    def __init__(
        self,
        answer_chunks: list[str],
        *,
        can_answer: bool = True,
        supported: bool = False,
    ) -> None:
        super().__init__(answer_chunks)
        self._can_answer = can_answer
        self._supported = supported
        self.complete_calls = 0

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        self.complete_calls += 1
        if self.complete_calls == 1:
            payload = {"can_answer": self._can_answer, "reason": "fake"}
        else:
            payload = {"supported": self._supported, "reason": "fake"}
        return LLMResponse(
            content=json.dumps(payload),
            model="fake-model",
            usage=TokenUsage(input_tokens=5, output_tokens=2, cached_tokens=0),
        )


# ---------------------------------------------------------------- 数据准备


async def _seed(session: Any) -> dict[str, Any]:
    """建租户 + 知识库 + 3 个文档（各 1 chunk + 1 向量）。

    三个文档内容都包含答案里可能出现的关键词，这样"引用哪一条"
    完全由 fake LLM 的输出决定，不受检索排序干扰。
    """
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    index_version_id = uuid.uuid4()

    await session.execute(
        text(
            "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
            "VALUES (:i, :s, 'cite-test', 'active', now(), now())"
        ),
        {"i": tenant_id, "s": f"cite-{tenant_id.hex[:10]}"},
    )
    await session.execute(
        text(
            "INSERT INTO users (id, email, display_name, status, platform_role, "
            "tokens_valid_after, created_at, updated_at) "
            "VALUES (:i, :e, 'cite', 'active', 'none', now(), now(), now())"
        ),
        {"i": user_id, "e": f"cite-{user_id.hex[:8]}@example.test"},
    )
    await session.execute(
        text(
            "INSERT INTO knowledge_bases (id, tenant_id, name, status, visibility, "
            "confidentiality, created_by, default_chunking_strategy, "
            "default_chunk_size, default_chunk_overlap, document_count, "
            "created_at, updated_at) VALUES (:i, :t, 'cite-kb', 'active', 'tenant', "
            "'public', :u, 'recursive', 800, 100, 3, now(), now())"
        ),
        {"i": kb_id, "t": tenant_id, "u": user_id},
    )
    await session.execute(
        text(
            "INSERT INTO index_versions (id, tenant_id, knowledge_base_id, "
            "version_tag, status, embedding_model, embedding_dim, chunk_count, "
            "expected_chunk_count, activated_at, created_at, updated_at) VALUES "
            "(:i, :t, :k, 'v1', 'active', 'fake-embed', 1024, 3, 3, now(), now(), now())"
        ),
        {"i": index_version_id, "t": tenant_id, "k": kb_id},
    )

    documents: list[dict[str, Any]] = []
    for offset, (label, content) in enumerate(
        (
            ("甲", "一线城市出差住宿每晚最多可报销 600 元。"),
            ("乙", "二线城市出差住宿每晚最多可报销 400 元。"),
            ("丙", "出差申请需提前三个工作日提交。"),
        )
    ):
        doc_id = uuid.uuid4()
        version_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        blob_id = uuid.uuid4()

        await session.execute(
            text(
                "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, storage_key, "
                "storage_backend, content_type, status, ref_count, created_at, updated_at) "
                "VALUES (:i, :t, :h, 100, :k, 'local', 'text/plain', 'complete', 1, now(), now())"
            ),
            {
                "i": blob_id,
                "t": tenant_id,
                "h": f"{label}-{offset}".ljust(64, "0")[:64],
                "k": f"cite/{offset}.txt",
            },
        )
        await session.execute(
            text(
                "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, "
                "original_filename, format, source_type, confidentiality, status, "
                "current_version_id, current_version_no, latest_version_no, retry_count, "
                "chunk_count, size_bytes, created_by, version, created_at, updated_at) "
                "VALUES (:i, :t, :k, :title, 'x.txt', 'text', 'upload', 'internal', 'ready', "
                "NULL, 1, 1, 0, 1, 100, :u, 1, now(), now())"
            ),
            {
                "i": doc_id,
                "t": tenant_id,
                "k": kb_id,
                "title": f"文档{label}",
                "u": user_id,
            },
        )
        await session.execute(
            text(
                "INSERT INTO document_versions (id, tenant_id, document_id, version_no, "
                "blob_id, status, chunking_strategy, chunking_config, chunk_count, "
                "is_current, created_by, created_at, updated_at) VALUES "
                "(:i, :t, :d, 1, :b, 'ready', 'recursive', '{}'::jsonb, 1, true, "
                ":u, now(), now())"
            ),
            {"i": version_id, "t": tenant_id, "d": doc_id, "b": blob_id, "u": user_id},
        )
        await session.execute(
            text("UPDATE documents SET current_version_id = :v WHERE id = :d"),
            {"v": version_id, "d": doc_id},
        )
        await session.execute(
            text(
                "INSERT INTO chunks (id, tenant_id, knowledge_base_id, document_id, "
                "document_version_id, chunk_index, content, search_text, chunk_level, is_current, "
                "char_start, char_end, content_hash, token_count, chunk_type, "
                "chunk_metadata, created_at, updated_at) VALUES "
                "(:i, :t, :k, :d, :v, 0, :c, :st, 0, true, 0, :len, :h, 10, 'text', "
                "'{}'::jsonb, now(), now())"
            ),
            {
                "i": chunk_id,
                "t": tenant_id,
                "k": kb_id,
                "d": doc_id,
                "v": version_id,
                "c": content,
                "st": tokenize_for_index(content),
                "len": len(content),
                "h": f"c{offset}".ljust(64, "0")[:64],
            },
        )
        await session.execute(
            text(
                "INSERT INTO chunk_embeddings (id, tenant_id, chunk_id, index_version_id, "
                "embedding, embedding_model, token_count, created_at, updated_at) "
                "VALUES (:i, :t, :c, :v, CAST(:e AS vector), 'fake-embed', 1, now(), now())"
            ),
            {
                "i": uuid.uuid4(),
                "t": tenant_id,
                "c": chunk_id,
                # 三条向量完全相同 → 检索必然全部命中，排序由距离并列决定。
                # 测试不依赖具体顺序：我们只断言"引用数量与标记一致"。
                "v": index_version_id,
                "e": str([1.0] + [0.0] * (_DIM - 1)),
            },
        )
        documents.append({"id": doc_id, "title": f"文档{label}", "content": content})

    return {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "kb_id": kb_id,
        "documents": documents,
    }


def _permission_ctx(seed: dict[str, Any]) -> Any:
    from rag_api.permissions.acl import PermissionContext

    return PermissionContext(
        user_id=seed["user_id"],
        tenant_id=seed["tenant_id"],
        roles=frozenset({TenantRole.OWNER}),
    )


async def _run_chat(
    session: Any,
    seed: dict[str, Any],
    *,
    answer_chunks: list[str],
    llm: LLMProvider | None = None,
    retrieval_config: RetrievalConfig | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """跑一次 chat_stream，收集 `(event, data)` 列表。"""
    service = ChatService(
        session,
        llm=llm or _FakeLLM(answer_chunks),
        embedder=_FakeEmbedder(),
        settings=_FakeSettings(),
        retrieval_config=retrieval_config,
    )
    events: list[tuple[str, dict[str, Any]]] = []
    async for event in service.chat_stream(
        tenant_id=seed["tenant_id"],
        user_id=seed["user_id"],
        query="一线城市住宿标准是多少？",
        knowledge_base_ids=[seed["kb_id"]],
        permission_ctx=_permission_ctx(seed),
    ):
        events.append((event.event, dict(event.data)))
    return events


# ---------------------------------------------------------------- 测试


@pytest.mark.integration
async def test_retrieval_event_carries_no_citations(db_session: Any) -> None:
    """**核心断言**：`retrieval` 事件里没有 citations，只有 sources_found。

    这条测试直接对应"生成过程中就显示来源 1,2,3,4,5"这个 bug。
    如果哪天有人把 citations 加回 retrieval 事件，这里会红。
    """
    async with db_session.begin_nested():
        seed = await _seed(db_session)
        events = await _run_chat(
            db_session, seed, answer_chunks=["一线城市住宿每晚不超过 600 元 [1]。"]
        )

    retrieval_events = [data for name, data in events if name == "retrieval"]
    assert len(retrieval_events) == 1, "应当只发一次 retrieval"

    payload = retrieval_events[0]
    assert "citations" not in payload, (
        "**retrieval 事件不能带 citations**——生成前不知道模型会引用哪几条，"
        "提前发会让编号与答案里的 [n] 对不上"
    )
    assert payload["sources_found"] == 3, "3 个 chunk 都应被检索到"
    assert payload["refused"] is False


@pytest.mark.integration
async def test_done_citations_only_include_referenced_material(db_session: Any) -> None:
    """答案只引用 `[1]` → `done` 只给 1 条引用（不是 3 条）。

    初版会给 3 条（全部检索命中），用户看到 3 个"来源"但答案其实只用了 1 个。
    """
    async with db_session.begin_nested():
        seed = await _seed(db_session)
        events = await _run_chat(
            db_session, seed, answer_chunks=["一线城市住宿每晚不超过 600 元 [1]。"]
        )

    done = next(data for name, data in events if name == "done")
    citations = done["citations"]

    assert len(citations) == 1, (
        f"答案只引用了 [1]，应当只有 1 条引用；实际 {len(citations)} 条"
        "（>1 说明还在按材料顺序全量给引用）"
    )
    assert citations[0]["index"] == 1


@pytest.mark.integration
async def test_done_citation_index_is_contiguous(db_session: Any) -> None:
    """引用 index 从 1 连续、不跳号。

    `AnswerWithCitations` 有这个校验，但 `ChatDonePayload.citations` 是
    裸 `list[Citation]`，**不会触发那个校验**。所以必须单独断言——
    跳号会让前端按 index 找引用时拿到错条目，且不报错。
    """
    async with db_session.begin_nested():
        seed = await _seed(db_session)
        # 答案里先引用 [2] 再引用 [1]：组装后应按首次出现顺序重新编号为 1,2
        events = await _run_chat(
            db_session,
            seed,
            answer_chunks=["二线城市 400 元 [2]。一线城市 600 元 [1]。"],
        )

    done = next(data for name, data in events if name == "done")
    indexes = [citation["index"] for citation in done["citations"]]

    assert indexes == [1, 2], f"index 必须从 1 连续，实际 {indexes}"


@pytest.mark.integration
async def test_out_of_range_marker_is_dropped(db_session: Any) -> None:
    """模型引用了不存在的编号 → 丢弃该引用，**保留答案**。

    抛异常会让"编号写错"变成"整个答案丢失"；静默保留会让前端拿到
    不存在的 index。正确做法是丢弃引用 + 计数供观测。
    """
    async with db_session.begin_nested():
        seed = await _seed(db_session)
        events = await _run_chat(
            db_session,
            seed,
            # [9] 不存在（只有 3 条材料）
            answer_chunks=["一线城市 600 元 [1]。另见 [9]。"],
        )

    done = next(data for name, data in events if name == "done")
    citations = done["citations"]

    assert len(citations) == 1, "越界的 [9] 应被丢弃，只保留 [1]"
    assert citations[0]["index"] == 1

    # 答案必须完整保留（含那个越界标记）
    deltas = "".join(data["text"] for name, data in events if name == "delta")
    assert "[9]" in deltas, "答案不能被改动——用户看到的就是模型原话"


@pytest.mark.integration
async def test_citation_quote_is_located_fragment(db_session: Any) -> None:
    """引用 quote 是**定位到的片段**，不是整块 chunk 的前 N 字。

    初版 `_build_citations` 用 `material.content[:300]`，于是引用卡片里
    显示的永远是 chunk 开头——如果答案引用的是 chunk 后半段的内容，
    用户点开看到的是完全不相干的文字。
    """
    async with db_session.begin_nested():
        seed = await _seed(db_session)
        events = await _run_chat(
            db_session,
            seed,
            answer_chunks=["一线城市出差住宿每晚最多可报销 600 元 [1]。"],
        )

    done = next(data for name, data in events if name == "done")
    citation = done["citations"][0]

    # 定位成功时 quote 应包含答案那句话的核心片段
    assert "600" in citation["quote"], f"quote 应包含答案引用的内容片段；实际 {citation['quote']!r}"

    # quote 必须是原文的子串（不能是模型改写后的句子）
    #
    # 注意 `citation["document_id"]` 是 JSON 序列化后的字符串，
    # 而 seed 里存的是 UUID —— 用 str() 比较。
    matched = next(
        (doc for doc in seed["documents"] if str(doc["id"]) == str(citation["document_id"])),
        None,
    )
    assert matched is not None, "引用的 document_id 必须来自检索材料"
    assert citation["quote"].strip() in matched["content"], (
        "quote 必须来自原文（不能是模型改写后的句子）"
    )


@pytest.mark.integration
async def test_invisible_document_citation_is_dropped(db_session: Any) -> None:
    """越权引用的第二道校验：不在可见集里的文档，引用必须被丢弃。

    ## 为什么 SQL 已过滤还要这一步

    这是 AGENTS.md 4.15 的硬要求。SQL 过滤挡的是"检索阶段"；
    但这些都可能引入未过滤内容：
    - 模型"记错"来源（把 A 文档的内容归到 B）；
    - 父块回填 / 上下文压缩带进别的 chunk；
    - 未来加多查询融合时某个子查询漏了条件。

    引用越界是**最隐蔽的泄漏形态**——它嵌在答案里，用户看不出异常。

    本测试直接构造一个不在 `visible_document_ids` 里的 Material，
    验证 `assemble_citations` 会丢弃它。
    """
    from rag_adapters.retrieval.citations_builder import Material, assemble_citations

    visible_doc = uuid.uuid4()
    invisible_doc = uuid.uuid4()

    materials = [
        Material(
            chunk_id=uuid.uuid4(),
            document_id=visible_doc,
            document_title="可见文档",
            chunk_index=0,
            content="一线城市出差住宿每晚最多可报销 600 元。",
            relevance_score=0.1,
        ),
        Material(
            chunk_id=uuid.uuid4(),
            document_id=invisible_doc,
            document_title="无权文档",
            chunk_index=0,
            content="绝密：高管薪酬方案。",
            relevance_score=0.2,
        ),
    ]

    result, stats = assemble_citations(
        answer="住宿 600 元 [1]。薪酬见 [2]。",
        materials=materials,
        visible_document_ids={visible_doc},  # invisible_doc 不在其中
    )

    assert len(result.citations) == 1
    assert result.citations[0].document_id == visible_doc
    # 答案保留（不能因为引用被丢弃就改答案）
    assert "[2]" in result.answer
    assert stats.out_of_range == 0, "这不是越界，是可见性拒绝，不该计入 out_of_range"


# ---------------------------------------------------------------- 混合检索与自省


@pytest.mark.integration
async def test_hybrid_retrieval_reaches_the_pipeline(db_session: Any) -> None:
    """开启关键词路后，检索图、RRF、材料回填都能跑通。

    之前的 `ChatService` 手写纯向量检索，`RetrievalGraph` 只在测试里用。
    这条测试确认生产编排真的走了图。
    """
    async with db_session.begin_nested():
        seed = await _seed(db_session)
        events = await _run_chat(
            db_session,
            seed,
            answer_chunks=["一线城市住宿每晚不超过 600 元 [1]。"],
            retrieval_config=RetrievalConfig(use_keyword=True, use_rewrite=False),
        )

    retrieval = next(data for name, data in events if name == "retrieval")
    assert retrieval["sources_found"] == 3
    assert retrieval["refused"] is False

    done = next(data for name, data in events if name == "done")
    assert done["refused"] is False
    assert len(done["citations"]) == 1


@pytest.mark.integration
async def test_self_check_flag_keeps_answer_but_marks_degraded(db_session: Any) -> None:
    """生成后自省不通过 + `flag`：保留答案，但 `done.degraded=true`。

    这是默认动作：先观察误报率，不因为一次判断就替用户丢掉答案。
    """
    async with db_session.begin_nested():
        seed = await _seed(db_session)
        llm = _FakeSelfCheckLLM(
            ["一线城市住宿每晚不超过 600 元 [1]。"],
            can_answer=True,
            supported=False,
        )
        events = await _run_chat(
            db_session,
            seed,
            answer_chunks=[],
            llm=llm,
            retrieval_config=RetrievalConfig(
                use_keyword=False,
                use_rewrite=False,
                use_self_check=True,
                self_check_refuse=False,
            ),
        )

    done = next(data for name, data in events if name == "done")
    assert done["refused"] is False, "flag 模式不能替用户丢答案"
    assert done["degraded"] is True, "必须标记答案缺乏材料支撑"
    assert len(done["citations"]) == 1
    # 答案原样保留
    deltas = "".join(data["text"] for name, data in events if name == "delta")
    assert "600" in deltas


@pytest.mark.integration
async def test_self_check_refuse_replaces_answer_and_citations(db_session: Any) -> None:
    """生成后自省不通过 + `refuse`：落库拒答、`done.refused=true`、无引用。

    流式已经发出的 token 收不回来，所以 `done` 是权威状态：
    前端必须用 done 覆盖已渲染的文本。
    """
    async with db_session.begin_nested():
        seed = await _seed(db_session)
        llm = _FakeSelfCheckLLM(
            ["一线城市住宿每晚不超过 600 元 [1]。"],
            can_answer=True,
            supported=False,
        )
        events = await _run_chat(
            db_session,
            seed,
            answer_chunks=[],
            llm=llm,
            retrieval_config=RetrievalConfig(
                use_keyword=False,
                use_rewrite=False,
                use_self_check=True,
                self_check_refuse=True,
            ),
        )

        done = next(data for name, data in events if name == "done")
        assert done["refused"] is True
        assert done["refusal_reason"] == "insufficient_evidence"
        assert done["citations"] == [], "拒答不能保留原答案的引用"
        assert done["degraded"] is False, "拒答是安全动作，不是降级"

        row = (
            await db_session.execute(
                text("SELECT content, refused, refusal_reason FROM messages WHERE id = :i"),
                {"i": uuid.UUID(done["message_id"])},
            )
        ).one()
        assert "根据现有资料无法回答" in str(row[0])
        assert row[1] is True
        assert row[2] == "insufficient_evidence"
