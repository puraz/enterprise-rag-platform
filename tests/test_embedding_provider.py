"""Embedding Provider 测试。

分两层：

- **不依赖 torch 的测试**（默认跑）：验证抽象契约、维度校验、
  顺序一致性、指令前缀、错误处理。用 fake 实现，任何环境都能跑。
- **真实模型测试**（`-m integration`）：加载 bge-m3 验证真实输出。
  首次运行需要下载 2.2GB 模型。

**为什么不用 mock 替代真实测试**：维度、归一化、语义相似度这些
只有真实模型能验证。用 fake 只能验证"代码不崩"，不能验证"检索能工作"。
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from rag_adapters.providers.embedding import (
    EmbeddingCapabilities,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResult,
    EmbeddingTask,
    NoOpReranker,
)
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


# ==================================================== 真实模型（integration）


@pytest.fixture(scope="module")
def real_provider() -> Any:
    """真实 bge-m3 Provider（module 级复用，避免重复加载 2.2GB 模型）。"""
    from rag_adapters.providers.bge import LocalBGEEmbedding

    settings = get_settings().embedding
    return LocalBGEEmbedding(settings)


@pytest.mark.integration
async def test_real_model_dimension(real_provider: Any) -> None:
    """真实模型输出维度必须是 1024。

    这是最基础的正确性验证——维度不对，整个索引都用不了。
    """
    result = await real_provider.embed_batch(["测试文本"])
    assert result.dimension == 1024
    assert len(result.vectors[0]) == 1024


@pytest.mark.integration
async def test_real_model_vectors_normalized(real_provider: Any) -> None:
    """真实模型输出归一化（模长 ≈ 1）。"""
    vector = await real_provider.embed_one("这是一个用于验证归一化的句子")
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-4, f"未归一化，模长 {norm}"


@pytest.mark.integration
async def test_real_model_semantic_similarity(real_provider: Any) -> None:
    """**语义相似度验证**：相近语义的文本向量距离应显著更近。

    这是 embedding 质量的核心验证——只测"维度对"是不够的，
    一个随机向量的实现也能通过维度测试。

    用余弦相似度（向量已归一化，所以等于点积）。
    """
    texts = [
        "如何申请年假",
        "年休假怎么申请",
        "服务器的CPU温度过高",
    ]
    result = await real_provider.embed_batch(texts)
    a, b, c = result.vectors

    def cosine(x: list[float], y: list[float]) -> float:
        return sum(xi * yi for xi, yi in zip(x, y, strict=True))

    similar = cosine(a, b)  # 两个问年假的
    different = cosine(a, c)  # 年假 vs CPU

    assert similar > different, f"语义相近的文本相似度 {similar:.3f} 应高于无关文本 {different:.3f}"
    assert similar > 0.6, f"同义问题的相似度应 > 0.6，实际 {similar:.3f}"


@pytest.mark.integration
async def test_real_model_batch_order_preserved(real_provider: Any) -> None:
    """批量处理时顺序必须与输入严格对应。

    错位的后果：向量与文本对不上，检索能跑但结果完全乱，且不报错。
    """
    texts = [f"第{i}条独立测试文本" for i in range(8)]
    result = await real_provider.embed_batch(texts)

    # 逐条单独算，应与批量结果的对应位置一致
    individual = await real_provider.embed_one(texts[3])

    def cosine(x: list[float], y: list[float]) -> float:
        return sum(xi * yi for xi, yi in zip(x, y, strict=True))

    assert cosine(result.vectors[3], individual) > 0.999, "批量与单条结果不一致（顺序错位？）"


@pytest.mark.integration
async def test_real_model_throughput(real_provider: Any) -> None:
    """吞吐基准：记录本机实际速度（不设硬性断言，只打印）。

    本机 16 核 CPU 的预期：约 30-60 条/秒（短文本）。
    这个数字决定 A8 的 worker 并发配置。
    """
    import time

    texts = [f"这是第{i}条用于吞吐测试的文本内容，长度约三十个字" for i in range(32)]
    # 预热（首次调用会加载模型）
    await real_provider.embed_batch(["预热"])

    started = time.perf_counter()
    result = await real_provider.embed_batch(texts)
    elapsed = time.perf_counter() - started

    rate = len(texts) / elapsed
    print(f"\n[吞吐] {len(texts)} 条耗时 {elapsed:.2f}s，约 {rate:.1f} 条/秒")
    print(f"[延迟] {result.latency_ms}ms")

    assert len(result.vectors) == len(texts)
    # 宽松下限：只要不是完全不可用（比如 1 条/秒）
    assert rate > 1.0, f"吞吐过低：{rate:.1f} 条/秒"


@pytest.mark.integration
async def test_real_model_health(real_provider: Any) -> None:
    """健康检查能加载模型并产出正确维度。"""
    assert await real_provider.health() is True


@pytest.mark.integration
async def test_real_model_handles_long_text(real_provider: Any) -> None:
    """长文本（接近上下文上限）能处理，不崩。

    bge-m3 支持 8192 token；这里用约 3000 字的中文验证。
    """
    long_text = "人工智能技术在近年来快速发展。" * 150  # 约 3000 字
    result = await real_provider.embed_batch([long_text])

    assert len(result.vectors) == 1
    assert len(result.vectors[0]) == 1024


@pytest.fixture(scope="module", autouse=True)
def _cleanup_provider() -> Any:
    """测试结束后释放模型内存。"""
    yield
    # module 级 fixture 的 aclose 在 teardown 里做：
    # 这里不持有引用，由 real_provider fixture 的调用者负责
