"""Reranker 契约、工厂与远程实现测试。

三个层次：

1. **NoOp**：降级链的终点，必须返回与输入等长、同序、全 0 的分数
   （图里的稳定排序据此保留 RRF 顺序）；
2. **工厂**：未知 provider 必须明确报错，不能静默降级；
3. **远程实现**（Cohere 风格 `/rerank`）：用 `httpx.MockTransport` 验证
   "按 index 归位、缺项报错、错误分类、分批、不回显原文"。

## 为什么远程实现的测试特别重要

rerank 响应是**按相关性降序**返回的，不是输入顺序。直接把
`results[i].score` 当第 i 个文档的分数，会让分数与内容错位——
表现是"重排后排序更乱"，没有任何异常。只能靠测试锁住。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from rag_adapters.providers import factory as provider_factory
from rag_adapters.providers.cohere_reranker import CohereStyleReranker
from rag_adapters.providers.embedding import (
    NoOpReranker,
    RerankerProviderError,
)
from rag_core.errors import ConfigError
from rag_core.settings import RerankerSettings, Settings

# ==================================================== NoOp


def test_noop_reranker_capabilities() -> None:
    provider = NoOpReranker()
    assert provider.capabilities.max_pairs >= 1


@pytest.mark.asyncio
async def test_noop_reranker_preserves_order_and_length() -> None:
    """全 0 分 + 与输入等长：图里的稳定排序据此保留融合顺序。"""
    provider = NoOpReranker()
    documents = ["a", "bb", "ccc"]

    result = await provider.rerank("问题", documents)

    assert result.scores == [0.0, 0.0, 0.0]
    assert len(result.scores) == len(documents)
    assert result.model == "none"


@pytest.mark.asyncio
async def test_noop_reranker_empty_input() -> None:
    result = await NoOpReranker().rerank("问题", [])
    assert result.scores == []


# ==================================================== 工厂


def test_factory_builds_noop() -> None:
    settings = Settings(reranker={"provider": "none"})
    provider = provider_factory.build_reranker(settings)
    assert isinstance(provider, NoOpReranker)


def test_factory_rejects_unknown_provider() -> None:
    """已移除的 `local_bge` 与任何拼错的键都必须明确报错。

    静默降级会让"开了重排但没跑"变成看不见的配置事故。
    """
    settings = Settings(reranker={"provider": "local_bge"})
    with pytest.raises(ConfigError, match="未知的 Reranker 实现"):
        provider_factory.build_reranker(settings)


def test_reranker_settings_defaults() -> None:
    settings = RerankerSettings()
    assert settings.provider == "none"
    assert settings.model_name == "Qwen/Qwen3-Reranker-4B"
    assert settings.base_url == "https://api.siliconflow.cn/v1"
    assert settings.configured is False


def test_reranker_settings_reject_legacy_local_fields() -> None:
    """旧 `.env` 里的本地模型字段必须报错，而不是被静默忽略。"""
    with pytest.raises(ValueError):
        RerankerSettings(provider="none", device="cuda")


def test_reranker_settings_requires_scheme_for_base_url() -> None:
    with pytest.raises(ValueError, match="必须带协议"):
        RerankerSettings(base_url="api.siliconflow.cn/v1")


def test_reranker_settings_summary_hides_key() -> None:
    summary = RerankerSettings(provider="siliconflow", api_key="sk-secret").redacted_summary
    assert "sk-secret" not in str(summary)
    assert summary["has_api_key"] is True
    assert RerankerSettings(provider="siliconflow", api_key="k").configured is True


# ==================================================== 远程实现


def _settings(**overrides: Any) -> RerankerSettings:
    base: dict[str, Any] = {
        "provider": "siliconflow",
        "model_name": "Qwen/Qwen3-Reranker-4B",
        "base_url": "https://example.invalid/v1",
        "api_key": "test-key",
        "batch_size": 2,
        "max_pairs": 100,
    }
    base.update(overrides)
    return RerankerSettings(**base)


def _provider(handler: Any, **overrides: Any) -> CohereStyleReranker:
    return CohereStyleReranker(_settings(**overrides), transport=httpx.MockTransport(handler))


async def test_remote_orders_scores_by_index() -> None:
    """响应按相关性降序返回，必须按 `index` 归位到输入顺序。"""
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        # 输入：["低", "高"]；服务端把高相关的排前面（index=1 在前）
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        )

    provider = _provider(handler)
    try:
        result = await provider.rerank("问题", ["低", "高"])
    finally:
        await provider.aclose()

    # 分数与输入顺序对应：第 0 条 0.1、第 1 条 0.9
    assert result.scores == [0.1, 0.9]
    assert result.model == "Qwen/Qwen3-Reranker-4B"

    payload = requests[0]
    assert payload["model"] == "Qwen/Qwen3-Reranker-4B"
    assert payload["query"] == "问题"
    assert payload["documents"] == ["低", "高"]
    # 不回显原文（少一个业务数据副本）
    assert payload["return_documents"] is False
    assert payload["top_n"] == 2


async def test_remote_batches_by_batch_size() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        docs = json.loads(request.content)["documents"]
        sizes.append(len(docs))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": index, "relevance_score": float(index + 1)}
                    for index in range(len(docs))
                ]
            },
        )

    provider = _provider(handler, batch_size=2)
    try:
        result = await provider.rerank("问题", ["a", "b", "c"])
    finally:
        await provider.aclose()

    assert sizes == [2, 1]
    # 第二批只有 1 条，批内 index=0 → 分数 1.0，接在全局第 3 个位置
    assert result.scores == [1.0, 2.0, 1.0]


async def test_remote_missing_scores_raise() -> None:
    """少一条分数必须报错：静默补 0 会让某段材料永远排最后。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    provider = _provider(handler)
    try:
        with pytest.raises(RerankerProviderError) as excinfo:
            await provider.rerank("问题", ["a", "b"])
    finally:
        await provider.aclose()

    assert excinfo.value.retryable is True
    assert "不完整" in str(excinfo.value)


async def test_remote_empty_documents_short_circuits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("空输入不应发请求")

    provider = _provider(handler)
    try:
        result = await provider.rerank("问题", [])
    finally:
        await provider.aclose()

    assert result.scores == []


async def test_remote_auth_error_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = _provider(handler)
    try:
        with pytest.raises(RerankerProviderError) as excinfo:
            await provider.rerank("问题", ["a"])
    finally:
        await provider.aclose()

    assert excinfo.value.retryable is False
    assert calls == 1


async def test_remote_rate_limit_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    provider = _provider(handler)
    try:
        with pytest.raises(RerankerProviderError) as excinfo:
            await provider.rerank("问题", ["a"])
    finally:
        await provider.aclose()

    assert excinfo.value.retryable is True


def test_remote_requires_api_key() -> None:
    with pytest.raises(RerankerProviderError, match="API_KEY"):
        CohereStyleReranker(_settings(api_key=None))


def test_remote_capabilities_are_not_local() -> None:
    provider = CohereStyleReranker(_settings())
    assert provider.capabilities.is_local is False
    assert provider.capabilities.max_pairs == 100


@pytest.mark.asyncio
async def test_factory_builds_remote_provider() -> None:
    settings = Settings(
        reranker={
            "provider": "siliconflow",
            "model_name": "Qwen/Qwen3-Reranker-4B",
            "api_key": "test-key",
        }
    )
    provider = provider_factory.build_reranker(settings)
    try:
        assert isinstance(provider, CohereStyleReranker)
    finally:
        await provider.aclose()
