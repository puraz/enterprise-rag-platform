"""Provider 工厂：按配置构造 LLM / Embedding / Reranker 实例。

## 为什么用注册表而不是 if-else

`docs/PROMPT.md` 要求"见到 `if provider == "xxx"` 一律改成能力匹配或
策略注册"。注册表让新增实现不需要修改工厂逻辑本身。

## 为什么 Embedding 与 LLM 分开构造

两类能力的配置、生命周期、成本模型完全不同：
- LLM：远程 HTTP，无状态，可随时切换；
- Embedding：本地加载 2GB 模型，切换意味着**全部向量失效需重建索引**。

所以它们各自有独立的 provider 配置项与工厂函数。混在一起会让
"改 LLM 模型"误触发 embedding 重建。

## Reranker 的实现

`RAG_RERANKER__PROVIDER` 可选 `none`（直通）/ `siliconflow` / `cohere_compatible`
（后两者是同一个 Cohere 风格 `/rerank` 适配器，区别只在配置）。
是否参与检索由 `RAG_RETRIEVAL__USE_RERANK` 控制——**两个开关分开**：
换模型与"开不开这个节点"是两件事。
"Reranker 挂 → 只用 RRF"的降级由调用方（`ChatService`）实现，
而不是在工厂里静默换成 NoOp（那会让评估"看起来开了但没跑"）。

为什么没有用 LangChain 的 rerank 封装：它只有厂商专属实现
（Jina/Infinity/本地模型），没有通用的远程 rerank 抽象（rerank 不在
OpenAI 协议里）。详见 `cohere_reranker.py` 的模块 docstring。

## Embedding 为什么用 LangChain

`/embeddings` 是 OpenAI 协议的一部分，LangChain 的 `OpenAIEmbeddings`
已经封装了鉴权/批量/重试开关/维度参数；自己再写一套只会让
"新增一家供应商"变成改代码。本地 bge-m3 已移除（它把 torch 带进镜像，
连带 2GB+ CUDA wheel）。
"""

from __future__ import annotations

from collections.abc import Callable

from rag_adapters.providers.cohere_reranker import CohereStyleReranker
from rag_adapters.providers.embedding import (
    EmbeddingProvider,
    NoOpReranker,
    RerankerProvider,
)
from rag_adapters.providers.langchain_embedding import LangChainEmbedding
from rag_adapters.providers.langchain_llm import LangChainLLMProvider
from rag_adapters.providers.llm import LLMProvider
from rag_core.errors import ConfigError
from rag_core.settings import Settings

# ---------------------------------------------------------------- LLM

_LLM_BUILDERS: dict[str, Callable[[Settings], LLMProvider]] = {
    "deepseek": lambda settings: LangChainLLMProvider(settings.llm),
    # OpenAI 兼容端点（通义/智谱/Moonshot/vLLM/Ollama）：
    # 和 deepseek 是同一套实现，区别只在 base_url 与模型名
    "openai_compatible": lambda settings: LangChainLLMProvider(settings.llm),
    "openai": lambda settings: LangChainLLMProvider(settings.llm),
}


def build_llm_provider(settings: Settings) -> LLMProvider:
    """按配置构造 LLM Provider。

    未知 provider 抛 `ConfigError` 而不是回退默认：
    静默回退会让"本想用 A 却实际用 B"变成看不见的配置事故。
    """
    name = settings.llm.provider
    builder = _LLM_BUILDERS.get(name)
    if builder is None:
        known = ", ".join(sorted(_LLM_BUILDERS))
        raise ConfigError(f"未知的 LLM 供应商 {name!r}（可选：{known}）")
    return builder(settings)


# ---------------------------------------------------------------- Embedding

_EMBEDDING_BUILDERS: dict[str, Callable[[Settings], EmbeddingProvider]] = {
    # 智谱（BigModel）：OpenAI 兼容的 /embeddings
    "zhipu": lambda settings: LangChainEmbedding(settings.embedding),
    # SiliconFlow / 百炼兼容模式 / vLLM / TEI 等所有提供 /embeddings 的服务。
    #
    # 三个键指向同一个实现："用哪家"是配置（base_url + model_name），
    # 不是代码分支。保留多个键是为了让配置语义清晰、日志可读。
    "siliconflow": lambda settings: LangChainEmbedding(settings.embedding),
    "openai_compatible": lambda settings: LangChainEmbedding(settings.embedding),
    "openai": lambda settings: LangChainEmbedding(settings.embedding),
}


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """按配置构造 Embedding Provider。"""
    name = settings.embedding.provider
    builder = _EMBEDDING_BUILDERS.get(name)
    if builder is None:
        known = ", ".join(sorted(_EMBEDDING_BUILDERS))
        raise ConfigError(f"未知的 Embedding 实现 {name!r}（可选：{known}）")
    provider = builder(settings)

    # **维度校验放在工厂里**：这是唯一能保证"配置与数据库列定义一致"的位置。
    # 放到入库时才校验的话，错误会在切分完几百个 chunk 后才暴露。
    actual = provider.capabilities.dimension
    if actual != settings.embedding.dimension:
        raise ConfigError(
            f"Embedding 维度不一致：实现返回 {actual}，配置声明 "
            f"{settings.embedding.dimension}。请检查 RAG_EMBEDDING__DIMENSION"
        )
    return provider


# ---------------------------------------------------------------- Reranker

_RERANKER_BUILDERS: dict[str, Callable[[Settings], RerankerProvider]] = {
    # 直通：不做重排，按融合顺序返回。
    "none": lambda settings: NoOpReranker(),
    # 远程 /rerank（Cohere 风格协议）。两个键指向同一个实现：
    # "用哪家"是配置（base_url + model_name），不是代码分支。
    "siliconflow": lambda settings: CohereStyleReranker(settings.reranker),
    "cohere_compatible": lambda settings: CohereStyleReranker(settings.reranker),
}


def build_reranker(settings: Settings) -> RerankerProvider:
    """按配置构造 Reranker。

    未知实现抛 `ConfigError`（与 LLM 工厂一致）：静默回退到 `NoOp`
    会让"本想开重排却实际没跑"变成看不见的配置事故。
    """
    name = settings.reranker.provider
    builder = _RERANKER_BUILDERS.get(name)
    if builder is None:
        known = ", ".join(sorted(_RERANKER_BUILDERS))
        raise ConfigError(f"未知的 Reranker 实现 {name!r}（可选：{known}）")
    return builder(settings)


def available_llm_providers() -> tuple[str, ...]:
    return tuple(sorted(_LLM_BUILDERS))


def available_embedding_providers() -> tuple[str, ...]:
    return tuple(sorted(_EMBEDDING_BUILDERS))


def available_reranker_providers() -> tuple[str, ...]:
    return tuple(sorted(_RERANKER_BUILDERS))
