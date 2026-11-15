"""分块层：把解析产物切成检索单元。

## 为什么分块在 adapters 而不是 worker

切分策略是**纯算法**：输入 `ParsedArtifact`，输出 `list[ChunkDraft]`。
它不认识数据库、不认识任务、不认识租户。这种纯度让它：

- 可以被脚本直接调用（`scripts/chunk_preview.py` 做策略对比）；
- 可以被单测覆盖（不需要起 PG/Redis）；
- 可以在未来被 API 端的"预览切分效果"接口复用（用户调参时实时看结果）。

如果放在 `rag_worker` 里，上面三件事都要绕道。

## 六种策略

| 策略 | 依据 | 适合 |
|---|---|---|
| `fixed` | 固定长度滑窗 | 对照基线 |
| `recursive` | 分隔符优先级递归 | 通用（默认） |
| `structured` | 标题层级 | 有清晰章节的文档 |
| `sentence_window` | 句子 + 动态窗口 | 连续散文、问答 |
| `parent_child` | 小块检索 + 父块回填 | 需要上下文完整性的场景 |
| `semantic` | 相邻向量相似度突变 | 无标题的长文本 |

## 契约

所有策略必须满足 `base.py` 里列的 6 条（不产生空块、不切受保护类型、
位置成对、顺序单调、可复现、不越界）。`verify_chunks()` 负责校验。
"""

from rag_adapters.chunkers.base import (
    PROTECTED_BLOCK_TYPES,
    BaseChunker,
    ChunkConfig,
    ChunkDraft,
    verify_chunks,
)
from rag_adapters.chunkers.fixed import FixedChunker
from rag_adapters.chunkers.parent_child import ParentChildChunker
from rag_adapters.chunkers.recursive import RecursiveChunker
from rag_adapters.chunkers.semantic import EmbedFn, SemanticChunker
from rag_adapters.chunkers.sentence_window import SentenceWindowChunker
from rag_adapters.chunkers.structured import StructuredChunker

__all__ = [
    "PROTECTED_BLOCK_TYPES",
    "BaseChunker",
    "ChunkConfig",
    "ChunkDraft",
    "EmbedFn",
    "FixedChunker",
    "ParentChildChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceWindowChunker",
    "StructuredChunker",
    "verify_chunks",
]
