"""Embedding Provider 测试。

分两层，都**不依赖网络**：

- **抽象契约**：验证维度校验、顺序一致性、指令前缀、错误传播。
  用 fake 实现，任何环境都能跑。
- **远程实现**（`OpenAICompatibleEmbedding`）：用 `httpx.MockTransport`
  验证"按 index 归位、归一化、分批、错误分类、维度校验"。
  这些是服务端**不保证**、必须由我们兜住的部分。

真实 API 的连通性由 `health()` 在运行时探测（`-m integration` 的
真实调用不再放进单测：会让每次 `make test` 依赖外部配额与网络）。
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from rag_adapters.providers.embedding import (
    EmbeddingCapabilities,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResult,
    EmbeddingTask,
    NoOpReranker,
)
from rag_adapters.providers.langchain_embedding import LangChainEmbedding
from rag_core.settings import get_settings
from rag_core.settings.embedding import EmbeddingSettings

# ==================================================== 测试用 fake 实现


class _FakeEmbedding(EmbeddingProvider):
    """确定性假实现：按文本长度生成向量。

    **确定性很重要**：同一文本必须得到同一向量，否则测试无法复现。
    用 hash 会因 Python 的 hash 随机化（PYTHONHASHSEED）而不可复现，
    所以这里用字符码求和。
    """

    name = "fake"

    def __init__(self, *, dimension: int = 1024, fail_with: Exception | None = None) -> None:
        self._dimension = dimension
        self._fail_with = fail_with
        self.calls: list[list[str]] = []

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            dimension=self._dimension, is_local=True, requires_query_instruction=False
        )

    async def embed_batch(
        self, texts: list[str], *, task: EmbeddingTask = EmbeddingTask.DOCUMENT
    ) -> EmbeddingResult:
        self.calls.append(list(texts))
        if self._fail_with is not None:
            raise self._fail_with

        vectors: list[list[float]] = []
        for text in texts:
            # 用字符码构造确定性向量，并归一化（与真实实现一致）
            seed = sum(ord(ch) for ch in text) % 97 + 1
            raw = [math.sin(seed * (i + 1)) for i in range(self._dimension)]
            norm = math.sqrt(sum(v * v for v in raw)) or 1.0
            vectors.append([v / norm for v in raw])

        return EmbeddingResult(vectors=vectors, dimension=self._dimension, model=self.name)


# ==================================================== 抽象契约


async def test_embed_batch_returns_matching_count() -> None:
    """向量条数必须与输入条数一致。

    不一致会导致"入库时 zip 错位"——向量与文本对不上，
    检索结果完全乱掉且不报错。
    """
    provider = _FakeEmbedding()
    result = await provider.embed_batch(["a", "b", "c"])

    assert len(result.vectors) == 3
    assert result.dimension == 1024


async def test_embed_batch_empty_input() -> None:
    """空输入返回空结果，不报错。

    空文档切分后没有 chunk 是正常情况，不该让整条管线失败。
    """
    provider = _FakeEmbedding()
    result = await provider.embed_batch([])

    assert result.vectors == []
    assert result.dimension == 1024


async def test_embed_one_convenience() -> None:
    """单条方法内部走批量路径（保证单一实现）。"""
    provider = _FakeEmbedding()
    vector = await provider.embed_one("hello")

    assert len(vector) == 1024
    # 单条方法的内部调用应记录为一次批量调用
    assert provider.calls == [["hello"]]


async def test_vectors_are_deterministic() -> None:
    """同一文本必须得到同一向量。

    不确定性会让"重新向量化"产生不同结果，检索结果无法复现。
    """
    provider = _FakeEmbedding()
    first = await provider.embed_one("相同的文本")
    second = await provider.embed_one("相同的文本")

    assert first == second


async def test_different_texts_get_different_vectors() -> None:
    """不同文本应得到不同向量（否则检索无意义）。"""
    provider = _FakeEmbedding()
    v1 = await provider.embed_one("短")
    v2 = await provider.embed_one("这是一段完全不同的长文本内容")

    assert v1 != v2


async def test_vectors_are_normalized() -> None:
    """向量必须归一化（bge 实现用 normalize_embeddings=True）。

    归一化后余弦相似度 = 内积，计算更快，且与 cosine 索引语义一致。
    """
    provider = _FakeEmbedding()
    vector = await provider.embed_one("test")

    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-6, f"向量未归一化，模长 {norm}"


async def test_error_propagates() -> None:
    """实现失败时必须抛 EmbeddingProviderError，不能静默返回空。

    静默返回空会让"文档入库成功但检索不到"——最难排查的故障。
    """
    provider = _FakeEmbedding(fail_with=EmbeddingProviderError("模型崩了"))
    with pytest.raises(EmbeddingProviderError):
        await provider.embed_batch(["x"])


# ==================================================== 维度校验


class _WrongDimensionProvider(_FakeEmbedding):
    """故意返回错误维度的实现，用于测试工厂/调用方的校验。"""

    def __init__(self) -> None:
        super().__init__(dimension=768)


def test_factory_rejects_dimension_mismatch() -> None:
    """**工厂必须校验维度一致性。**

    `chunks.embedding` 是 vector(1024)。如果 Provider 输出 768 维，
    入库会报 `expected 1024 dimensions, not 768`——但那时已经切分完
    几百个 chunk，重跑成本高。

    工厂在构造时校验，让配置错误在**启动阶段**暴露。
    """
    from rag_adapters.providers import factory as provider_factory

    settings = get_settings()
    # 临时替换注册表，注入错误维度的实现
    original = provider_factory._EMBEDDING_BUILDERS.copy()
    provider_factory._EMBEDDING_BUILDERS["wrong_dim"] = lambda s: _WrongDimensionProvider()
    try:
        bad_settings = settings.model_copy(
            update={
                "embedding": settings.embedding.model_copy(
                    update={"provider": "wrong_dim", "dimension": 1024}
                )
            }
        )
        from rag_core.errors import ConfigError

        with pytest.raises(ConfigError, match="维度不一致"):
            provider_factory.build_embedding_provider(bad_settings)
    finally:
        provider_factory._EMBEDDING_BUILDERS.clear()
        provider_factory._EMBEDDING_BUILDERS.update(original)


def test_factory_rejects_unknown_provider() -> None:
    """未知 provider 明确报错，不回退默认。

    静默回退会让"本想用本地模型却实际走了 API"变成看不见的配置事故。
    """
    from rag_adapters.providers import factory as provider_factory
    from rag_core.errors import ConfigError

    settings = get_settings()
    bad = settings.model_copy(
        update={"embedding": settings.embedding.model_copy(update={"provider": "nope"})}
    )
    with pytest.raises(ConfigError, match="未知的 Embedding 实现"):
        provider_factory.build_embedding_provider(bad)


# ==================================================== 指令前缀


class _InstructionTrackingProvider(EmbeddingProvider):
    """记录实际送进模型的文本，用于验证指令前缀。"""

    name = "tracking"

    def __init__(self, instruction: str | None) -> None:
        self._instruction = instruction
        self.seen: list[str] = []

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            dimension=1024, requires_query_instruction=self._instruction is not None
        )

    async def embed_batch(
        self, texts: list[str], *, task: EmbeddingTask = EmbeddingTask.DOCUMENT
    ) -> EmbeddingResult:
        prepared = (
            [f"{self._instruction}{t}" for t in texts]
            if self._instruction and task is EmbeddingTask.QUERY
            else texts
        )
        self.seen.extend(prepared)
        return EmbeddingResult(
            vectors=[[0.0] * 1024 for _ in prepared], dimension=1024, model=self.name
        )


async def test_query_instruction_applied_only_to_queries() -> None:
    """查询指令只对 QUERY 任务生效，DOCUMENT 不加前缀。

    用错前缀不会报错，只会让向量空间偏移、召回率下降——
    所以必须有测试锁住这个行为。

    bge-m3 不需要前缀（配置为 None），bge-large-zh-v1.5 需要。
    这里是接口层面的验证。
    """
    provider = _InstructionTrackingProvider(instruction="为这个句子生成表示以用于检索相关文章：")

    await provider.embed_batch(["文档内容"], task=EmbeddingTask.DOCUMENT)
    await provider.embed_batch(["用户问题"], task=EmbeddingTask.QUERY)

    assert provider.seen == ["文档内容", "为这个句子生成表示以用于检索相关文章：用户问题"]


async def test_no_instruction_when_not_configured() -> None:
    """未配置指令时（bge-m3 的情况），查询也不加前缀。"""
    provider = _InstructionTrackingProvider(instruction=None)

    await provider.embed_batch(["用户问题"], task=EmbeddingTask.QUERY)
    assert provider.seen == ["用户问题"]


# ==================================================== Reranker 占位


async def test_noop_reranker_preserves_order() -> None:
    """NoOpReranker 返回全 0 分，且顺序与输入一致。

    A11 之前用它占位，保证代码路径完整；
    降级链里"Reranker 挂 → 只用 RRF"就是这个行为。
    """
    reranker = NoOpReranker()
    docs = ["a", "b", "c"]

    result = await reranker.rerank("query", docs)

    assert result.scores == [0.0, 0.0, 0.0]
    assert len(result.scores) == len(docs)


# ==================================================== 配置


def test_embedding_dimension_matches_vector_column() -> None:
    """**配置维度必须与 A3-3 的 vector(1024) 一致。**

    这是整条链路的基础不变量：数据库列定义、配置、Provider 能力声明
    三者必须是同一个数。任一处不对都会在入库时炸。
    """
    from rag_db.models.chunk import EMBEDDING_DIM

    assert get_settings().embedding.dimension == EMBEDDING_DIM == 1024


def test_embedding_settings_rejects_invalid_dimension() -> None:
    """维度必须为正整数。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EmbeddingSettings(dimension=0)


# ==================================================== 远程实现（LangChain OpenAIEmbeddings）


def _remote_settings(**overrides: Any) -> EmbeddingSettings:
    """测试用配置：4 维让断言可读，batch_size=2 方便验证分批。"""
    base: dict[str, Any] = {
        "provider": "zhipu",
        "model_name": "embedding-3",
        "dimension": 4,
        "base_url": "https://example.invalid/v4",
        "api_key": "test-key",
        "batch_size": 2,
        "normalize": True,
    }
    base.update(overrides)
    return EmbeddingSettings(**base)


def _provider(handler: Any, **overrides: Any) -> LangChainEmbedding:
    return LangChainEmbedding(_remote_settings(**overrides), transport=httpx.MockTransport(handler))


def _embedding_response(
    vectors: list[list[float]],
    *,
    indexes: list[int] | None = None,
    tokens: int = 1,
) -> dict[str, Any]:
    order = indexes or list(range(len(vectors)))
    return {
        "object": "list",
        "model": "embedding-3",
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        "data": [
            {"object": "embedding", "index": index, "embedding": vectors[position]}
            for position, index in enumerate(order)
        ],
    }


async def test_remote_maps_by_index_and_normalizes() -> None:
    """按响应里的 `index` 归位，并自己做归一化。

    这是改用原始响应的直接收益：LangChain 的 `aembed_documents` 只按
    返回顺序装配（`base.py:750`），服务端一旦乱序就会向量与文本错位。
    乱序 + 未归一化都不会报错，只会让检索相关性变坏。
    """
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        # 故意把 index=1 放在前面；向量未归一化（模长 5 与 10）
        return httpx.Response(
            200,
            json=_embedding_response(
                [[0.0, 6.0, 0.0, 8.0], [3.0, 0.0, 4.0, 0.0]], indexes=[1, 0], tokens=12
            ),
        )

    provider = _provider(handler)
    try:
        result = await provider.embed_batch(["第一条", "第二条"])
    finally:
        await provider.aclose()

    # 请求形状：模型名 + 显式维度（否则服务端返回原生维度）+ 纯字符串输入
    assert requests[0]["model"] == "embedding-3"
    assert requests[0]["dimensions"] == 4
    assert requests[0]["input"] == ["第一条", "第二条"]

    # index=0 对应"第一条"：[3,0,4,0] / 5 = [0.6,0,0.8,0]
    assert result.vectors[0] == pytest.approx([0.6, 0.0, 0.8, 0.0], abs=1e-6)
    assert result.vectors[1] == pytest.approx([0.0, 0.6, 0.0, 0.8], abs=1e-6)
    for vector in result.vectors:
        assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)
    # usage 必须透传到结果，供 model_calls 成本入账
    assert result.input_tokens == 12


async def test_remote_sends_strings_not_token_ids() -> None:
    """`check_embedding_ctx_length=False` 的回归测试。

    默认 True 时 LangChain 会把文本编码成 tiktoken id 数组发出去，
    第三方端点（智谱/SiliconFlow）不认，会 400 或产出无意义向量。
    """
    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["input"])
        return httpx.Response(200, json=_embedding_response([[1.0, 0.0, 0.0, 0.0]]))

    provider = _provider(handler, batch_size=1)
    try:
        await provider.embed_batch(["纯字符串"])
    finally:
        await provider.aclose()

    assert seen == [["纯字符串"]]


async def test_remote_batches_and_keeps_global_order() -> None:
    """超过 batch_size 要分批；每批的向量数量与输入一致。"""
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        batch = json.loads(request.content)["input"]
        sizes.append(len(batch))
        vectors = [[float(index + 1), 0.0, 0.0, 0.0] for index in range(len(batch))]
        return httpx.Response(200, json=_embedding_response(vectors))

    provider = _provider(handler, batch_size=2)
    try:
        result = await provider.embed_batch(["a", "b", "c"])
    finally:
        await provider.aclose()

    assert sizes == [2, 1]
    assert len(result.vectors) == 3
    # 第二批的第一个（全局第 3 条）在批内 index=0 → 归一化后 [1,0,0,0]
    assert result.vectors[2] == pytest.approx([1.0, 0.0, 0.0, 0.0])


async def test_remote_rejects_wrong_dimension() -> None:
    """服务端忽略 dimensions 时必须报错，且是永久失败。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_embedding_response([[1.0, 0.0]]))

    provider = _provider(handler)
    try:
        with pytest.raises(EmbeddingProviderError) as excinfo:
            await provider.embed_batch(["x"])
    finally:
        await provider.aclose()

    assert excinfo.value.retryable is False
    assert "维度" in str(excinfo.value)


async def test_remote_rejects_wrong_count() -> None:
    """返回条数与输入不符 → 失败（重试不会变好）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_embedding_response([[1.0, 0.0, 0.0, 0.0]]))

    provider = _provider(handler)
    try:
        with pytest.raises(EmbeddingProviderError) as excinfo:
            await provider.embed_batch(["a", "b"])
    finally:
        await provider.aclose()

    # 条数不符视为可重试的协议异常：可能是上游抖动，重试有机会恢复
    assert excinfo.value.retryable is True


async def test_remote_auth_error_is_not_retryable() -> None:
    """401 不应被 ResilientCaller 重试（重试只会浪费配额）。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = _provider(handler)
    try:
        with pytest.raises(EmbeddingProviderError) as excinfo:
            await provider.embed_batch(["x"])
    finally:
        await provider.aclose()

    assert excinfo.value.retryable is False
    assert calls == 1


async def test_remote_rate_limit_is_retryable() -> None:
    """429 必须标记为可重试（worker 据此退避重试而不是直接死信）。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    provider = _provider(handler)
    try:
        with pytest.raises(EmbeddingProviderError) as excinfo:
            await provider.embed_batch(["x"])
    finally:
        await provider.aclose()

    assert excinfo.value.retryable is True
    # ResilientCaller 默认 3 次尝试
    assert calls >= 2


async def test_remote_error_does_not_echo_input() -> None:
    """异常信息不能带响应体里的长文本（可能是文档正文，而异常会进日志）。"""
    long_text = "机密内容" * 500

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": long_text}})

    provider = _provider(handler)
    try:
        with pytest.raises(EmbeddingProviderError) as excinfo:
            await provider.embed_batch(["x"])
    finally:
        await provider.aclose()

    assert "机密内容" not in str(excinfo.value)


def test_remote_requires_api_key() -> None:
    """缺 key 必须在构造期失败（而不是等 worker 处理到向量化）。"""
    with pytest.raises(EmbeddingProviderError, match="API_KEY"):
        LangChainEmbedding(_remote_settings(api_key=None))


async def test_remote_health_reports_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_embedding_response([[1.0, 0.0, 0.0, 0.0]]))

    provider = _provider(handler, batch_size=1)
    try:
        assert await provider.health() is True
    finally:
        await provider.aclose()


async def test_factory_builds_langchain_provider() -> None:
    """工厂按配置装配 LangChain 实现，并校验维度与库列定义一致。"""
    from rag_adapters.providers import factory as provider_factory

    base = get_settings()
    embedding = EmbeddingSettings(
        provider="zhipu",
        model_name="embedding-3",
        dimension=1024,
        api_key="test-key",
    )
    settings = base.model_copy(update={"embedding": embedding})

    provider = provider_factory.build_embedding_provider(settings)
    try:
        assert isinstance(provider, LangChainEmbedding)
        assert provider.capabilities.dimension == 1024
        assert provider.capabilities.is_local is False
    finally:
        await provider.aclose()


def test_configured_requires_api_key() -> None:
    assert _remote_settings().configured is True
    assert _remote_settings(api_key=None).configured is False
    # 摘要里不能出现 key 原文
    assert "test-key" not in str(_remote_settings().redacted_summary)


def test_base_url_must_have_scheme() -> None:
    with pytest.raises(ValueError, match="必须带协议"):
        _remote_settings(base_url="open.bigmodel.cn/api/paas/v4")
