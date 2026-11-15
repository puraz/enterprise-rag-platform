"""分块器注册表与工厂。

## 为什么用注册表而不是 if-else

`docs/PROMPT.md` 的硬约束："见到 `if provider == "xxx"` 一律改成能力匹配
或策略注册"。分块策略是典型场景：新增策略只加一个类 + 一行注册。

## 为什么策略名必须与枚举一致

`ChunkingStrategy`（contracts）是前端与 API 看到的名字，
`document_versions.chunking_strategy` 存的是它的值。
注册表的键必须与枚举**值**完全一致，否则：

    DB 里存 "recursive" → 注册表找不到 "recursive" → 报"未知策略"

而根因只是名字拼写不同（比如注册成 "RecursiveChunker"）。
`tests/test_chunkers.py` 会断言注册表覆盖枚举全部成员。
"""

from __future__ import annotations

from collections.abc import Callable

from rag_adapters.chunkers.base import BaseChunker, ChunkConfig
from rag_adapters.chunkers.fixed import FixedChunker
from rag_adapters.chunkers.parent_child import ParentChildChunker
from rag_adapters.chunkers.recursive import RecursiveChunker
from rag_adapters.chunkers.semantic import EmbedFn, SemanticChunker
from rag_adapters.chunkers.sentence_window import SentenceWindowChunker
from rag_adapters.chunkers.structured import StructuredChunker
from rag_contracts.enums.retrieval import ChunkingStrategy

# 策略名 → 构造器。
#
# 构造器接受 (config, embed_fn) 两个参数：语义策略需要 embedding，
# 其余策略忽略它。统一签名让工厂不需要按名字分支。
_ChunkerBuilder = Callable[[ChunkConfig, "EmbedFn | None"], BaseChunker]

_REGISTRY: dict[str, _ChunkerBuilder] = {
    ChunkingStrategy.FIXED.value: lambda config, _embed: FixedChunker(config),
    ChunkingStrategy.RECURSIVE.value: lambda config, _embed: RecursiveChunker(config),
    ChunkingStrategy.STRUCTURED.value: lambda config, _embed: StructuredChunker(config),
    ChunkingStrategy.SENTENCE_WINDOW.value: lambda config, _embed: SentenceWindowChunker(config),
    ChunkingStrategy.PARENT_CHILD.value: lambda config, _embed: ParentChildChunker(config),
    ChunkingStrategy.SEMANTIC.value: lambda config, embed_fn: SemanticChunker(
        config, embed_fn=embed_fn
    ),
}


def build_chunker(
    strategy: str,
    *,
    config: ChunkConfig | None = None,
    embed_fn: EmbedFn | None = None,
) -> BaseChunker:
    """按策略名构造 chunker。

    Args:
        strategy: `ChunkingStrategy` 的值。
        config: 切分参数（来自版快照或库默认值）。
        embed_fn: 仅语义策略使用。其余策略忽略。

    Raises:
        ValueError: 策略名未注册。

    **未知策略必须报错而不是回退默认**：静默回退会让人以为
    "配置生效了"，实际却在用另一个策略切分——这类问题只能靠
    对比 chunk 结果才能发现，排查成本极高。
    """
    builder = _REGISTRY.get(strategy)
    if builder is None:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"未知的切分策略 {strategy!r}（已注册：{known}）")
    return builder(config or ChunkConfig(), embed_fn)


def registered_strategies() -> tuple[str, ...]:
    """已注册的策略名（供启动日志与文档生成）。"""
    return tuple(sorted(_REGISTRY))


def strategies_requiring_embedding() -> tuple[str, ...]:
    """需要 embedding 才能发挥最佳效果的策略。

    handler 据此决定"要不要在切分前加载模型"——
    用 recursive 的文档不需要加载 2GB 模型。
    """
    return (ChunkingStrategy.SEMANTIC.value,)


__all__ = [
    "build_chunker",
    "registered_strategies",
    "strategies_requiring_embedding",
]
