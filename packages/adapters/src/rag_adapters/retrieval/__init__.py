"""检索层：融合策略 + 检索图 + 提示词构造。

## 为什么在 adapters 而不是 apps/api

这里的东西是**纯检索策略**：给定查询与检索器，产出排好序的材料。
它不认识路由、不认识请求上下文、不认识租户（ACL 由注入的检索器负责）。

放在 adapters 的好处：
- 检索图能被单测（注入内存假检索器，无需数据库）；
- 评估脚本（`tests/eval/run_eval.py`）能直接用它，
  于是"评估的实现"与"生产的实现"是同一份代码——否则评估结果
  不能代表线上行为；
- 未来 API 端与 worker 端都能复用。

## 模块

| 模块 | 内容 |
|---|---|
| `fusion.py` | RRF 融合（只用排名，免疫分数量纲差异） |
| `state.py` | 图状态与配置（每个节点一个开关） |
| `graph.py` | 检索图节点 + 提示词构造 |
"""

from rag_adapters.retrieval.fusion import (
    DEFAULT_RRF_K,
    FusedItem,
    fuse_two,
    reciprocal_rank_fusion,
)
from rag_adapters.retrieval.graph import (
    KeywordSearcher,
    QueryRewriter,
    Reranker,
    RetrievalGraph,
    SelfChecker,
    VectorSearcher,
    build_generation_prompt,
    build_self_check_prompt,
)
from rag_adapters.retrieval.state import (
    RetrievalConfig,
    RetrievalState,
    RetrievedChunk,
)

__all__ = [
    "DEFAULT_RRF_K",
    "FusedItem",
    "KeywordSearcher",
    "QueryRewriter",
    "Reranker",
    "RetrievalConfig",
    "RetrievalGraph",
    "RetrievalState",
    "RetrievedChunk",
    "SelfChecker",
    "VectorSearcher",
    "build_generation_prompt",
    "build_self_check_prompt",
    "fuse_two",
    "reciprocal_rank_fusion",
]
