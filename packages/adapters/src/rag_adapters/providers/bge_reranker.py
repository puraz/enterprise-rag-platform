"""本地 bge-reranker（sentence-transformers CrossEncoder）。

## 为什么需要重排

双路召回（向量 + 关键词）经 RRF 融合后，排序只反映"两路各自的排名"，
不反映"这条材料与问题的**细粒度**相关性"。Cross-encoder 把
`(query, document)` 拼在一起做一次完整编码，能捕捉"同一个话题但
答非所问"的差异——这是 bi-encoder（embedding）做不到的。

代价是 **O(n) 次模型推理**：20 条候选在 CPU 上约数百毫秒。所以：

- 只重排融合后的前 `rerank_top_n` 条（尾部本来进不了 `final_limit`）；
- 尾部结果保留但不重排（丢弃会伤召回）；
- **默认关闭**（`RetrievalSettings.use_rerank=False`），先在评估里
  验证增益（`tests/eval/run_eval.py --rerank`）再开。

## 为什么用 CrossEncoder 而不是自己拼模型

`sentence-transformers` 已封装了 tokenizer 批处理、长度截断、
设备放置与推理循环。自己写只会重复这些，且更容易在
"padding / truncation / 顺序"上出错——**顺序错位不报错，
只表现为排序变差**，是检索里最难发现的一类 bug。

## 分数语义

返回的是模型原始分数（logits），**不做 sigmoid 转换**：
重排只关心**相对顺序**，而 sigmoid 是单调变换，不改变顺序。
保留原始分数让"分数分布是否漂移"仍可观测（转换后会压缩到 0~1，
反而看不出模型是否在退化的输入上给出低置信分数）。

不要把这些分数与向量距离比较——两者量纲与方向都不同。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rag_adapters.providers.embedding import (
    RerankerCapabilities,
    RerankerProvider,
    RerankerProviderError,
    RerankResult,
)
from rag_core.logging import get_logger
from rag_core.settings import RerankerSettings

log = get_logger(__name__)


class LocalBGEReranker(RerankerProvider):
    """基于 sentence-transformers `CrossEncoder` 的本地重排实现。"""

    def __init__(self, settings: RerankerSettings) -> None:
        self._settings = settings
        self.name = settings.model_name
        self._model: Any = None
        self._load_lock = asyncio.Lock()

    @property
    def capabilities(self) -> RerankerCapabilities:
        return RerankerCapabilities(
            max_pairs=self._settings.max_pairs,
            max_input_tokens=self._settings.max_length,
            is_local=True,
        )

    async def _ensure_loaded(self) -> Any:
        """延迟加载模型（只加载一次）。

        加载是 CPU 密集的同步操作（读权重 + 构建计算图），必须切线程，
        否则会阻塞整个事件循环——期间所有请求都卡住。
        """
        loaded: Any = self._model
        if loaded is not None:
            return loaded

        async with self._load_lock:
            loaded = self._model
            if loaded is not None:
                return loaded

            started = time.perf_counter()
            try:
                self._model = await asyncio.to_thread(self._load_model)
            except Exception as exc:
                raise RerankerProviderError(
                    f"加载 reranker {self._settings.model_name} 失败", retryable=False
                ) from exc

            log.info(
                "reranker_model_loaded",
                model=self._settings.model_name,
                device=self._settings.device,
                elapsed_s=round(time.perf_counter() - started, 1),
            )
            return self._model

    def _load_model(self) -> Any:
        """同步加载（在线程池里执行）。"""
        import os

        if self._settings.hf_endpoint:
            # 必须在 import sentence_transformers 之前设置：
            # 它读取该环境变量决定下载源（直连 huggingface.co 在国内会超时）。
            os.environ["HF_ENDPOINT"] = self._settings.hf_endpoint

        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {
            "device": self._settings.device,
            "max_length": self._settings.max_length,
        }
        if self._settings.cache_dir is not None:
            kwargs["cache_folder"] = self._settings.cache_dir
        return CrossEncoder(self._settings.model_name, **kwargs)

    async def rerank(self, query: str, documents: list[str]) -> RerankResult:
        """对候选文档打分。

        **返回顺序必须与 `documents` 严格一致**：调用方会用
        `zip(scores, documents, strict=True)` 排序，错位不会报错，
        只会让"重排后顺序更乱"——最隐蔽的失败形态。

        候选数超过 `max_pairs` 时分批推理，避免一次性吃满内存。
        """
        if not documents:
            return RerankResult(scores=[], model=self.name, latency_ms=0)

        model = await self._ensure_loaded()
        started = time.perf_counter()
        batch_cap = max(1, self._settings.max_pairs)
        scores: list[float] = []

        try:
            for start in range(0, len(documents), batch_cap):
                batch = documents[start : start + batch_cap]
                pairs = [(query, document) for document in batch]
                raw = await asyncio.to_thread(
                    model.predict,
                    pairs,
                    batch_size=self._settings.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                # numpy 数组有 tolist；某些版本可能返回 list/标量。
                values = raw.tolist() if hasattr(raw, "tolist") else list(raw)

                # 单条输入时 predict 可能返回标量而不是长度 1 的列表。
                if isinstance(values, (int, float)):
                    values = [values]

                scores.extend(float(value) for value in values)
        except Exception as exc:
            raise RerankerProviderError("reranker 推理失败", retryable=True) from exc

        if len(scores) != len(documents):
            # 长度不符说明实现/模型行为异常。**必须报错**：
            # 静默截断会让分数与内容错位，重排结果看起来"只是变差了"。
            raise RerankerProviderError(
                f"reranker 返回 {len(scores)} 个分数，但输入 {len(documents)} 条文档",
                retryable=False,
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "rerank_completed",
            model=self.name,
            pairs=len(documents),
            latency_ms=latency_ms,
        )
        return RerankResult(scores=scores, model=self.name, latency_ms=latency_ms)

    async def aclose(self) -> None:
        """释放模型内存。"""
        self._model = None
        import gc

        gc.collect()


__all__ = ["LocalBGEReranker"]
