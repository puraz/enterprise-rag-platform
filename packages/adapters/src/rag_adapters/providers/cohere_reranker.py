"""Cohere 风格 `/rerank` 协议适配（SiliconFlow / Cohere / Jina 等）。

## 为什么不直接用 LangChain 的 rerank 封装（已实测）

LangChain 确实有 rerank 实现，但都不是"通用远程 API"：

| 类 | 问题 |
|---|---|
| `InfinityRerank` | 硬编码 `http://localhost:7997`；需 `infinity_client` 包；需自己注入 client |
| `JinaRerank` | 硬编码 Jina 端点；无 base_url 参数 |
| `FlashrankRerank` | 本地模型（重） |
| `CohereRerank` | `langchain_community` 0.4.x 已移除；且绑定 cohere SDK |

另外它们都是 **sync 的 `Document` 压缩器**，默认 `top_n=3`
（会丢掉其余候选并打乱输入顺序）——而检索图需要"每个输入一条分数、
保留尾部"。强行套用会在适配层堆更多胶水，还不如 100 行直写协议。

**关键：rerank 不属于 OpenAI 协议**。embedding 能直接用
`OpenAIEmbeddings`，rerank 不能——这是协议事实，不是框架选型问题。

## 协议形状（SiliconFlow / Cohere / Jina 基本一致）

    POST {base_url}/rerank
    {"model": "Qwen/Qwen3-Reranker-4B", "query": "...",
     "documents": ["..."], "return_documents": false, "top_n": N}

    → {"results": [{"index": 0, "relevance_score": 0.93}, ...]}

所以"换供应商"只需要改 `base_url` + `model_name`：
工厂里 `siliconflow` / `cohere_compatible` 两个键指向同一个实现。

## 三个关键取舍

### 1. `return_documents=false`

响应不回显文档原文。文档正文属于业务数据，能少一个副本就少一个
（响应体积也从几十 KB 降到几百字节）。

### 2. 必须按 `index` 归位

`results` 是**按相关性降序**返回的，不是输入顺序。直接按位置取分数
会把分数与内容错位——表现为"重排后排序更乱"，且不报错。

### 3. 分数越大越相关

与向量距离方向相反。`RetrievalGraph._rerank` 已按降序排序，
这里只负责"每个文档对应一个分数"，不预先排序。
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from rag_adapters.providers.embedding import (
    RerankerCapabilities,
    RerankerProvider,
    RerankerProviderError,
    RerankResult,
)
from rag_core.logging import get_logger
from rag_core.resilience import ResiliencePolicy, ResilientCaller
from rag_core.settings.retrieval import RerankerSettings

log = get_logger(__name__)


class CohereStyleReranker(RerankerProvider):
    """远程 rerank（Cohere 风格 `/rerank` 协议）。

    适配 SiliconFlow、Cohere、Jina 等提供同形状接口的服务：
    `base_url` + `model_name` 是唯一需要改的配置。
    """

    def __init__(
        self,
        settings: RerankerSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if settings.api_key is None or not settings.api_key.get_secret_value():
            raise RerankerProviderError(
                "未配置 RAG_RERANKER__API_KEY（远程 rerank 必需）", retryable=False
            )

        self.name = settings.model_name
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=settings.timeout_s,
            transport=transport,
            headers={
                "Authorization": f"Bearer {settings.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )
        self._caller = ResilientCaller(
            f"reranker:{settings.provider}",
            ResiliencePolicy(max_concurrency=settings.max_concurrency),
        )

    @property
    def capabilities(self) -> RerankerCapabilities:
        return RerankerCapabilities(
            max_pairs=self._settings.max_pairs,
            is_local=False,
        )

    async def rerank(self, query: str, documents: list[str]) -> RerankResult:
        """对候选文档打分。

        返回分数与 `documents` **顺序严格对应**（越大越相关），
        调用方据此排序。
        """
        if not documents:
            return RerankResult(scores=[], model=self.name)

        started = time.perf_counter()
        scores: list[float] = []
        size = min(self._settings.batch_size, self._settings.max_pairs)
        for offset in range(0, len(documents), size):
            batch = documents[offset : offset + size]
            scores.extend(await self._rerank_batch(query, batch))

        log.info(
            "rerank_completed",
            document_count=len(documents),
            batches=(len(documents) + size - 1) // size,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return RerankResult(
            scores=scores,
            model=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    # ------------------------------------------------------------ 内部

    async def _rerank_batch(self, query: str, batch: list[str]) -> list[float]:
        payload: dict[str, Any] = {
            "model": self._settings.model_name,
            "query": query,
            "documents": batch,
            # 不回显原文：少一份业务数据副本，响应也更小
            "return_documents": False,
            # 显式要求全量：少一条就说明协议异常，我们要能发现
            "top_n": len(batch),
        }

        async def _post() -> dict[str, Any]:
            response = await self._client.post("/rerank", json=payload)
            if response.status_code >= 400:
                raise _error_from_response(response)
            try:
                body: dict[str, Any] = response.json()
            except ValueError as exc:
                raise RerankerProviderError("rerank 响应不是合法 JSON", retryable=True) from exc
            return body

        body, _stats = await self._caller.call(_post, operation="rerank")
        results = body.get("results")
        if not isinstance(results, list):
            raise RerankerProviderError("rerank 响应缺少 results", retryable=True)

        return _order_scores(results, expected=len(batch))

    async def health(self) -> bool:
        """探活：用两个极短文本验证鉴权与协议。"""
        try:
            result = await self.rerank("测试", ["相关", "无关"])
            return len(result.scores) == 2
        except Exception as exc:
            log.warning("reranker_health_failed", error_type=type(exc).__name__)
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


def _order_scores(results: list[Any], *, expected: int) -> list[float]:
    """把 `results` 按 `index` 归位成与输入等长的分数列表。

    `results` 是按**相关性降序**返回的，直接按位置取会错位。
    缺项/重复/越界都视为协议错误：静默补 0 会让"某段材料永远排最后"，
    在检索质量上表现为难以解释的波动。
    """
    scores: list[float | None] = [None] * expected
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or not isinstance(score, (int, float)):
            continue
        if 0 <= index < expected and scores[index] is None:
            scores[index] = float(score)

    missing = [position for position, value in enumerate(scores) if value is None]
    if missing:
        raise RerankerProviderError(
            f"rerank 返回的分数不完整（缺少 index {missing[:5]}），期望 {expected} 条",
            retryable=True,
        )
    return [float(value) for value in scores if value is not None]


def _error_from_response(response: httpx.Response) -> RerankerProviderError:
    """HTTP → 领域异常（不把响应体原文放进异常）。"""
    status = response.status_code
    if status in (401, 403):
        return RerankerProviderError(
            f"rerank 鉴权失败（HTTP {status}）：检查 RAG_RERANKER__API_KEY",
            retryable=False,
        )
    if status == 429:
        return RerankerProviderError("rerank 触发限流（HTTP 429）", retryable=True)
    if status >= 500:
        return RerankerProviderError(f"rerank 服务端错误（HTTP {status}）", retryable=True)
    if status in (400, 413, 422):
        return RerankerProviderError(
            f"rerank 请求被拒绝（HTTP {status}）：检查模型名 / 批量大小", retryable=False
        )
    return RerankerProviderError(f"rerank 调用失败（HTTP {status}）", retryable=False)


__all__ = ["CohereStyleReranker"]
