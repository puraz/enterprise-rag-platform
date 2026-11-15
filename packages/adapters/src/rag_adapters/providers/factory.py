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

## Reranker 的现状

A5 不实现真实 reranker（用户选择推到 A11）。工厂返回 `NoOpReranker`，
保证代码路径完整——`docs/PROMPT.md` 的降级链里
"Reranker 挂 → 只用 RRF"就是这个行为。
"""

from __future__ import annotations

from collections.abc import Callable

from rag_adapters.providers.bge import LocalBGEEmbedding
from rag_adapters.providers.embedding import (
    EmbeddingProvider,
    NoOpReranker,
    RerankerProvider,
)
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
    "local_bge": lambda settings: LocalBGEEmbedding(settings.embedding),
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
    # A11 会加：
    # "local_bge": lambda s: LocalBGEReranker(s.reranker),
    # "cohere": lambda s: CohereReranker(s.reranker),
    "none": lambda settings: NoOpReranker(),
}


def build_reranker(settings: Settings) -> RerankerProvider:
    """按配置构造 Reranker。

    A5 只有 `none`（直通）。A11 加真实实现时在这里注册即可，
    调用方代码不变。
    """
    name = getattr(settings, "reranker_provider", "none")
    builder = _RERANKER_BUILDERS.get(name, _RERANKER_BUILDERS["none"])
    return builder(settings)


def available_llm_providers() -> tuple[str, ...]:
    return tuple(sorted(_LLM_BUILDERS))


def available_embedding_providers() -> tuple[str, ...]:
    return tuple(sorted(_EMBEDDING_BUILDERS))
