"""Embedding 与 Reranker 供应商抽象。

## 与 LLM 抽象的区别

| | LLM | Embedding |
|---|---|---|
| 输出 | 文本 + token 用量 | 定长向量 |
| 维度 | 不关心 | **必须与数据库列定义一致** |
| 批处理 | 单次对话 | 一次几百条文本 |
| 成本 | 按 token | 本地免费 / API 按 token |
| 失败影响 | 本次回答失败 | 整批 chunk 无法入库 |

## 维度是本层最关键的不变量

`chunks.embedding` 是 `vector(1024)`（A3-3 已固定）。如果 Provider
返回别的维度，写入会报 "expected 1024 dimensions, not 768"——
但这个错误发生在**批量入库时**，届时已经解析/切分完几百个 chunk，
重跑成本高。

所以抽象层要求实现声明 `dimension`，并在启动时与配置校验：

    settings.embedding.dimension == provider.dimension

不匹配就启动失败（`INTERNAL_CONFIG_ERROR`），而不是等到入库才炸。

## 批处理与背压

Embedding 是典型的"一次能处理一批，但总量很大"的场景：
- 一份 300 页的 PDF 可能切出 2000 个 chunk；
- 逐条 embedding 会产生 2000 次模型前向，耗时是批处理的 10-50 倍。

所以接口从设计上就是 `embed_batch(texts) -> list[vector]`，
且实现内部再按 `batch_size` 分块（避免一次吃进 2000 条把内存打满）。

**顺序必须与输入严格对应**：返回的向量列表第 i 个必须对应输入第 i 条。
错位的后果是"检索能跑但相关性完全乱"，且极难发现（不报错）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class EmbeddingTask(StrEnum):
    """embedding 的任务类型。

    **bge 系列模型对查询与文档采用不同的指令前缀**：
    - 文档：`"为这个句子生成表示以用于检索相关文章：{text}"`（部分版本）
    - 查询：同样的前缀

    bge-m3 不需要指令前缀（官方说明），但接口保留这个区分：
    换成需要前缀的模型（如 bge-large-zh-v1.5）时，实现内部据此加前缀，
    调用方不用改代码。

    用错任务类型的后果：向量空间偏移，召回率下降但不报错。
    """

    DOCUMENT = "document"
    """为入库的文档块生成向量。"""

    QUERY = "query"
    """为用户查询生成向量。"""


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """一批 embedding 的结果。"""

    vectors: list[list[float]]
    """向量列表，**顺序与输入严格对应**。"""

    dimension: int
    """向量维度。调用方应校验它与数据库列定义一致。"""

    model: str
    """实际使用的模型标识（进 model_calls 记账）。"""

    latency_ms: int | None = None

    input_tokens: int = 0
    """本次调用消耗的输入 token 数（服务端 usage）。

    为 0 表示服务端没返回 usage（并非没消耗）。入账时据此判断是否写
    `model_calls`：0 的层不写，避免账本里堆满“0 token”的噪音。
    """


@dataclass(frozen=True, slots=True)
class EmbeddingCapabilities:
    """Embedding 模型的能力声明。"""

    dimension: int
    """输出向量维度。**必须与 `chunks.embedding` 的列定义一致。**"""

    max_input_tokens: int = 512
    """单条文本的最大 token 数。超长会被截断（静默损失语义）。"""

    supports_batch: bool = True
    """是否支持一次输入多条。本地模型都支持；部分 API 也支持。"""

    is_local: bool = True
    """本地模型：无 API 成本，但占内存与 CPU。"""

    requires_query_instruction: bool = False
    """是否需要对查询加指令前缀（bge-*-zh-v1.5 需要，bge-m3 不需要）。"""


class EmbeddingProviderError(Exception):
    """Embedding 调用失败。不继承 AppError（与 LLM Provider 同样的理由）。"""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class EmbeddingProvider(ABC):
    """Embedding 供应商接口。

    实现约定：
    - **不做重试**（由 ResilientCaller 负责）；
    - **不吞异常**；
    - **不记录文本内容**（文档正文不能进日志）；
    - 返回顺序必须与输入严格对应。
    """

    name: str = "unknown"

    @property
    @abstractmethod
    def capabilities(self) -> EmbeddingCapabilities:
        raise NotImplementedError

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = EmbeddingTask.DOCUMENT,
    ) -> EmbeddingResult:
        """为一批文本生成向量。

        Args:
            texts: 文本列表。允许为空（返回空结果，不报错）。
            task: 文档还是查询（影响是否加指令前缀）。

        Raises:
            EmbeddingProviderError: 调用失败。
        """
        raise NotImplementedError

    async def embed_one(
        self,
        text: str,
        *,
        task: EmbeddingTask = EmbeddingTask.DOCUMENT,
    ) -> list[float]:
        """单条便捷方法。内部走 `embed_batch`，保持单一实现路径。"""
        result = await self.embed_batch([text], task=task)
        if not result.vectors:
            raise EmbeddingProviderError("模型返回空向量", retryable=False)
        return result.vectors[0]

    async def health(self) -> bool:
        """是否可用（模型已加载）。默认 True。"""
        return True

    async def aclose(self) -> None:  # noqa: B027 - 可选钩子，非抽象方法
        """释放资源（如卸载模型）。

        **有意不做成 abstractmethod**：多数实现（API 型）无需释放资源，
        强制实现会让每个子类写一个空方法。这是模板方法模式里的
        "可选钩子"，与必须实现的 `embed_batch` 不同。
        """


# ---------------------------------------------------------------- Reranker


@dataclass(frozen=True, slots=True)
class RerankResult:
    """一组候选的重排结果。"""

    scores: list[float]
    """与输入候选**顺序对应**的相关性分数。"""

    model: str
    latency_ms: int | None = None

    input_tokens: int = 0
    """本次重排消耗的 token（query + 所有候选）。0 表示服务端未返回 usage。"""


class RerankerProviderError(Exception):
    """Reranker 失败。"""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class RerankerCapabilities:
    """Reranker 能力声明。"""

    max_pairs: int = 100
    """单次最多处理多少 (query, doc) 对。超过要分批。"""

    max_input_tokens: int = 512
    is_local: bool = True


class RerankerProvider(ABC):
    """Reranker 接口。

    **A5 暂不实现**（用户选择推到 A11）：rerank 的价值只能在一批
    真实检索结果上验证（rerank 前后 nDCG 对比），提前实现无法验证效果。

    这里先把接口定下来，保证 A11 加实现时不需要改调用方。
    """

    name: str = "unknown"

    @property
    @abstractmethod
    def capabilities(self) -> RerankerCapabilities:
        raise NotImplementedError

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> RerankResult:
        """对候选文档按与 query 的相关性打分。

        **返回分数列表的顺序必须与 documents 一致**——
        调用方要用 zip(scores, documents) 排序。
        """
        raise NotImplementedError


class NoOpReranker(RerankerProvider):
    """直通实现：不做重排，返回全 0 分。

    A11 之前用它占位，保证代码路径完整（`docs/PROMPT.md` 的降级链里
    "Reranker 挂 → 只用 RRF" 就是这个行为）。
    """

    name = "none"

    @property
    def capabilities(self) -> RerankerCapabilities:
        return RerankerCapabilities(is_local=True)

    async def rerank(self, query: str, documents: list[str]) -> RerankResult:
        return RerankResult(scores=[0.0] * len(documents), model=self.name)
