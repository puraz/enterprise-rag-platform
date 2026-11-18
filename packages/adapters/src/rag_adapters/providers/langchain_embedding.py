"""基于 LangChain `OpenAIEmbeddings` 的 Embedding 实现。

## 为什么用 LangChain 而不是自己发 HTTP

`/embeddings` 看着简单，但换供应商时要重写的东西比想象中多：

- 鉴权头、base_url、路径拼接的差异；
- 批量切分与并发（`chunk_size`）；
- 重试开关（`max_retries`，必须与我们的 ResilientCaller 互斥）；
- 维度参数（有的叫 `dimensions`，有的只认模型原生维度）；
- 响应解析与错误类型。

这些 LangChain 已经封装且被大量供应商验证过。自研第二套的结局是
"某个字段只在一家实现里修好"——这正是 AGENTS.md 4.4 要避免的。
换成 SiliconFlow / 百炼 / vLLM / TEI 只改 `base_url` + `model_name`。

## 为什么仍然要一层自己的 Adapter

项目的检索层只认 `EmbeddingProvider` 协议（维度、顺序、归一化、
`EmbeddingTask`）。直接把 LangChain 对象传进业务，会让
"换 embedding 框架"变成一次全链路改造。这层 Adapter 很薄，
只做四件 LangChain 不做/不该由它决定的事，见下面各节。

## 三个必须显式设置、否则会静默出错的参数

### 1. `check_embedding_ctx_length=False`（最关键）

默认 `True` 时，LangChain 会用 **tiktoken** 把文本编码成 token id 数组，
再以 `input: [[...], [...]]` 发给服务端——那是 OpenAI 的扩展字段。
智谱/SiliconFlow 等兼容端点只接受字符串，会直接 400 或（更糟）
把 token 数组当字符串处理，产出无意义向量。

### 2. `max_retries=0`

LangChain 默认重试 2 次。与 `ResilientCaller` 叠加会变成
"3 × 3 = 9 次"，且熔断器看不到真实的失败节奏。

### 3. 独立的 `http_async_client`

默认共享模块级 client；某个实例 `aclose()` 会让其它实例抛
`Event loop is closed`（本项目在 LLM 侧踩过同样的坑）。

## 三、一个必须自己解析响应的理由（已实测）

`aembed_documents` 只返回向量列表：**既没有 `usage`，也不读 `index`**
（`langchain_openai/embeddings/base.py:750` 按响应顺序装配）。而成本对账需要
`usage.prompt_tokens`，排序正确性需要 `index`。

所以这里用 LangChain 实例持有的 `async_client`（鉴权/base_url/超时/
连接池都由 LangChain 配好）发请求，只把“响应 → 领域对象”的十几个字段
自己解析。这不是重写 HTTP 客户端，而是补齐框架没暴露的两个字段。
顺带修掉了旧实现"依赖响应顺序"的隐含假设。

## 归一化必须自己做

LangChain 不管归一化，OpenAI 协议也不保证单位向量。本项目用
`vector_cosine_ops` 建索引，归一化后余弦 = 内积：查询更快，
且将来换内积索引不用重建数据。
"""

from __future__ import annotations

import math
import time
from typing import Any

import httpx

from rag_adapters.providers.embedding import (
    EmbeddingCapabilities,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResult,
    EmbeddingTask,
)
from rag_core.logging import get_logger
from rag_core.resilience import ResiliencePolicy, ResilientCaller
from rag_core.settings.embedding import EmbeddingSettings

log = get_logger(__name__)


class LangChainEmbedding(EmbeddingProvider):
    """OpenAI 兼容 embedding（LangChain `OpenAIEmbeddings` 适配）。"""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """构造客户端。

        Args:
            settings: embedding 配置。
            transport: 仅用于测试（注入 `httpx.MockTransport`）。
        """
        if settings.api_key is None or not settings.api_key.get_secret_value():
            # 构造期失败：配置缺失应该在装配/启动时报，
            # 而不是等 worker 处理到向量化（那时已烧掉解析成本）。
            raise EmbeddingProviderError(
                "未配置 RAG_EMBEDDING__API_KEY（远程 embedding 必需）",
                retryable=False,
            )

        from langchain_openai import OpenAIEmbeddings

        self.name = settings.model_name
        self._settings = settings
        self._expected_dimension = settings.dimension
        # 独立 client：不要与其它 provider 共享（关闭会互相影响）
        self._http = httpx.AsyncClient(
            timeout=settings.timeout_s,
            transport=transport,
        )
        self._embeddings = OpenAIEmbeddings(
            model=settings.model_name,
            # 收 SecretStr 而不是 str：这是 LangChain 声明的类型，
            # 且 SecretStr 不会被 repr 打出来。
            api_key=settings.api_key,
            base_url=settings.base_url,
            # 只有需要显式降维时才传；None 表示用模型原生维度
            dimensions=settings.dimension if settings.send_dimensions else None,
            # 每次 `aembed_documents` 只发一个 HTTP 请求（我们自己分批）
            chunk_size=settings.batch_size,
            max_retries=0,
            # 关键：否则会发 tiktoken token 数组，第三方端点不认
            check_embedding_ctx_length=False,
            http_async_client=self._http,
        )
        self._caller = ResilientCaller(
            f"embedding:{settings.provider}",
            ResiliencePolicy(max_concurrency=settings.max_concurrency),
        )

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            dimension=self._expected_dimension,
            max_input_tokens=self._settings.max_input_tokens,
            supports_batch=True,
            is_local=False,
            requires_query_instruction=self._settings.query_instruction is not None,
        )

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = EmbeddingTask.DOCUMENT,
    ) -> EmbeddingResult:
        """批量生成向量（顺序与输入严格对应）。"""
        if not texts:
            return EmbeddingResult(vectors=[], dimension=self._expected_dimension, model=self.name)

        prepared = self._apply_instruction(texts, task)
        started = time.perf_counter()
        vectors: list[list[float]] = []
        input_tokens = 0

        # 显式分批：失败只重试当前批，不会把整份文档的所有批再算一遍。
        size = self._settings.batch_size
        for offset in range(0, len(prepared), size):
            batch = prepared[offset : offset + size]
            batch_vectors, batch_tokens = await self._embed_batch(batch)
            vectors.extend(self._finalize(vector) for vector in batch_vectors)
            input_tokens += batch_tokens

        latency_ms = int((time.perf_counter() - started) * 1000)
        self._validate(vectors, expected_count=len(texts))

        log.info(
            "embedding_batch_completed",
            count=len(vectors),
            batches=math.ceil(len(texts) / size),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            task=str(task),
        )
        return EmbeddingResult(
            vectors=vectors,
            dimension=self._expected_dimension,
            model=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
        )

    # ------------------------------------------------------------ 内部

    async def _embed_batch(self, batch: list[str]) -> tuple[list[list[float]], int]:
        """发一次请求，返回 `(向量, 输入 token 数)`。

        **用 `async_client.create` 而不是 `aembed_documents`**：
        后者丢掉 `usage` 与 `index`（见模块 docstring 第三节）。
        """
        payload: dict[str, Any] = {"input": batch, "model": self._settings.model_name}
        if self._settings.send_dimensions:
            payload["dimensions"] = self._expected_dimension

        async def _call() -> Any:
            try:
                return await self._embeddings.async_client.create(**payload)
            except Exception as exc:
                raise _classify(exc) from exc

        response, _stats = await self._caller.call(_call, operation="embeddings")

        data = list(getattr(response, "data", None) or [])
        vectors = [_as_vector(item) for item in _order_by_index(data, expected=len(batch))]
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        return vectors, tokens

    def _finalize(self, vector: list[float]) -> list[float]:
        if not self._settings.normalize:
            return vector
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # 零向量无法归一化，也无法用于余弦检索；通常是空串输入，
            # 属于上游切分问题，重试无用。
            raise EmbeddingProviderError("embedding 返回零向量（输入可能为空）", retryable=False)
        return [value / norm for value in vector]

    def _validate(self, vectors: list[list[float]], *, expected_count: int) -> None:
        """返回前校验条数与维度。

        服务端忽略 `dimensions` 时，错误会在**批量入库**阶段才暴露，
        届时已切分完几百个 chunk。这里失败能直接指向配置。
        """
        if len(vectors) != expected_count:
            raise EmbeddingProviderError(
                f"返回向量数 {len(vectors)} 与输入条数 {expected_count} 不一致",
                retryable=False,
            )
        for vector in vectors:
            if len(vector) != self._expected_dimension:
                raise EmbeddingProviderError(
                    f"服务端返回维度 {len(vector)} 与配置的 "
                    f"{self._expected_dimension} 不一致；"
                    "请检查 RAG_EMBEDDING__DIMENSION 与 chunks.embedding 的列定义",
                    retryable=False,
                )

    def _apply_instruction(self, texts: list[str], task: EmbeddingTask) -> list[str]:
        """按任务类型加指令前缀（只有配置了才加）。"""
        instruction = self._settings.query_instruction
        if instruction is None or task is not EmbeddingTask.QUERY:
            return texts
        return [f"{instruction}{text}" for text in texts]

    async def health(self) -> bool:
        """探活：验证鉴权、维度与网络（成本可忽略）。"""
        try:
            result = await self.embed_batch(["健康检查"])
            return bool(result.vectors) and len(result.vectors[0]) == self._expected_dimension
        except Exception as exc:
            log.warning("embedding_health_failed", error_type=type(exc).__name__)
            return False

    async def aclose(self) -> None:
        """关闭连接池（worker 优雅退出时必须调用）。"""
        await self._http.aclose()


def _order_by_index(items: list[Any], *, expected: int) -> list[Any]:
    """按 `index` 把返回项归位到输入顺序。

    `index` 是请求内的 0-based 下标。若服务端没给（或不是 0..n-1 的排列），
    退化为按返回顺序使用并留 debug 日志：**静默接受一个可能乱序的结果，
    等于把不确定性埋进检索质量里**。
    """
    if len(items) != expected:
        raise EmbeddingProviderError(
            f"embedding 返回 {len(items)} 条，期望 {expected} 条", retryable=True
        )

    indices: list[int] = []
    for item in items:
        index = item.get("index") if isinstance(item, dict) else getattr(item, "index", None)
        if not isinstance(index, int):
            break
        indices.append(index)

    if len(indices) == expected and sorted(indices) == list(range(expected)):
        order = sorted(range(expected), key=lambda position: indices[position])
        return [items[position] for position in order]

    log.debug("embedding_index_missing_or_unexpected", indices=indices)
    return items


def _as_vector(item: Any) -> list[float]:
    embedding = (
        item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
    )
    if not isinstance(embedding, list) or not embedding:
        raise EmbeddingProviderError("embedding 响应缺少向量", retryable=True)
    return [float(value) for value in embedding]


def _classify(exc: Exception) -> EmbeddingProviderError:
    """把 SDK 异常映射成带 `retryable` 的领域异常。

    ## 为什么按异常类型而不是 HTTP 状态码

    LangChain/OpenAI SDK 已经把状态码分类成异常类型了，再自己解析
    `response.status_code` 反而会漏掉包装层。`retryable` 是 worker
    决定"重试还是死信"的唯一依据（AGENTS.md 4.11）。
    """
    try:
        import openai
    except ImportError:  # pragma: no cover - 由 langchain-openai 保证已安装
        return EmbeddingProviderError(f"embedding 调用失败: {type(exc).__name__}")

    # 配置类：重试不会变好
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return EmbeddingProviderError(
            "embedding 鉴权失败：检查 RAG_EMBEDDING__API_KEY 与模型权限", retryable=False
        )
    if isinstance(exc, openai.BadRequestError):
        return EmbeddingProviderError(
            "embedding 请求被拒绝：检查模型名 / 维度 / 批量大小", retryable=False
        )
    # 可恢复类
    if isinstance(exc, openai.RateLimitError):
        return EmbeddingProviderError("embedding 触发限流（429）", retryable=True)
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return EmbeddingProviderError("embedding 网络超时/连接失败", retryable=True)
    if isinstance(exc, openai.InternalServerError):
        return EmbeddingProviderError("embedding 服务端错误（5xx）", retryable=True)
    return EmbeddingProviderError(f"embedding 调用失败: {type(exc).__name__}", retryable=False)


__all__ = ["LangChainEmbedding"]
