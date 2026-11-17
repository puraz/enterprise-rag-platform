"""本地 reranker 测试（不下载模型）。

## 测什么

`LocalBGEReranker` 的价值不在"模型准不准"（那要用真实语料评估），
而在**接口契约**：

1. 返回分数的顺序必须与输入文档严格一致（错位不报错，只表现为排序乱）；
2. 空输入不加载模型、不报错；
3. 候选超过 `max_pairs` 时正确分批；
4. 分数数量不符必须报错——静默截断是最危险的失败形态。

用假的 `CrossEncoder` 替换 `_load_model`，避免测试下载 280MB 权重。
"""

from __future__ import annotations

import pytest

from rag_adapters.providers.bge_reranker import LocalBGEReranker
from rag_adapters.providers.embedding import RerankerProviderError
from rag_core.settings import RerankerSettings


class _FakeCrossEncoder:
    """按文档长度打分，并记录每次 predict 的批次。"""

    def __init__(self, *, fixed_scores: list[float] | None = None) -> None:
        self.batches: list[list[tuple[str, str]]] = []
        self._fixed = fixed_scores

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        self.batches.append(list(pairs))
        if self._fixed is not None:
            return list(self._fixed)
        return [float(len(document)) for _query, document in pairs]


def _provider(*, max_pairs: int = 10, batch_size: int = 8) -> LocalBGEReranker:
    return LocalBGEReranker(
        RerankerSettings(
            provider="local_bge",
            model_name="fake-reranker",
            max_pairs=max_pairs,
            batch_size=batch_size,
        )
    )


@pytest.mark.asyncio
async def test_scores_follow_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    fake = _FakeCrossEncoder()
    monkeypatch.setattr(provider, "_load_model", lambda: fake)

    result = await provider.rerank("问题", ["短", "中等长度", "很长很长"])

    # 长度：1 / 4 / 4 —— 顺序必须与输入一致
    assert result.scores == [float(len(doc)) for doc in ["短", "中等长度", "很长很长"]]
    assert result.model == "fake-reranker"


@pytest.mark.asyncio
async def test_empty_documents_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    fake = _FakeCrossEncoder()
    monkeypatch.setattr(provider, "_load_model", lambda: fake)

    result = await provider.rerank("问题", [])

    assert result.scores == []
    assert fake.batches == [], "空输入不应该加载/调用模型"


@pytest.mark.asyncio
async def test_batches_by_max_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """候选数超过 max_pairs 时分批，且总分数量仍与输入一致。"""
    provider = _provider(max_pairs=2)
    fake = _FakeCrossEncoder()
    monkeypatch.setattr(provider, "_load_model", lambda: fake)

    documents = ["a", "bb", "ccc", "dddd", "eeeee"]
    result = await provider.rerank("问题", documents)

    assert len(fake.batches) == 3, f"5 条 / max_pairs=2 应分批 3 次，实际 {len(fake.batches)}"
    assert result.scores == [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.asyncio
async def test_score_count_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """数量不符必须报错：静默截断会让分数与内容错位且不告警。"""
    provider = _provider()
    fake = _FakeCrossEncoder(fixed_scores=[1.0])
    monkeypatch.setattr(provider, "_load_model", lambda: fake)

    with pytest.raises(RerankerProviderError) as excinfo:
        await provider.rerank("问题", ["文档一", "文档二"])

    assert excinfo.value.retryable is False


def test_capabilities_reflect_settings() -> None:
    provider = _provider(max_pairs=50)
    caps = provider.capabilities

    assert caps.max_pairs == 50
    assert caps.is_local is True
