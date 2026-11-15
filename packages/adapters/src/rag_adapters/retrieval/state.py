"""检索图：路由 → 混合检索 → 融合 → 重排 → 生成。

## 为什么用图而不是一串 if-else

检索链路有**条件分支与回退**：

    问题是否清晰？ ──否──▶ 改写 ──▶ 重新检索
         │是
         ▼
    是否该用关键词？──是──▶ 混合检索
         │否
         ▼
      纯向量检索
         │
         ▼
    结果够不够？──不够──▶ 扩大召回 / 拒答

写成嵌套 if-else 后，三个问题会立刻出现：
1. **路径不可见**：出问题时不知道走了哪条分支（日志要自己串）；
2. **不可中断/恢复**：长链路（含 LLM 调用）中途失败只能整体重跑；
3. **加节点要改主干**：每加一个"可选优化"都得小心不破坏已有分支。

LangGraph 把节点与边显式化，天然解决前两条；配合"每个节点可独立
开关"的配置，第三条也解决了——这正是 A11 "每加一个节点做对比"的基础。

## 节点清单与启用顺序（A11 的推进路线）

    base      : retrieve(vector) → generate          ← 基线，可跑通即验收
    +keyword  : retrieve(vector + keyword) → fuse    ← 混合检索
    +rewrite  : rewrite → retrieve → fuse → generate ← 查询改写
    +rerank   : ... → rerank → generate              ← 重排
    +selfcheck: ... → generate → 自省 → 可能重新检索 ← 自反思

**每个节点的开关是独立的**（`RetrievalConfig`），这样才能做到
"只加一个节点、其余不变"地对比。否则多个节点一起上，指标变化
无法归因到具体节点。

## 为什么融合与重排是两件事

- **融合（RRF）** 解决"多路结果怎么合并"，只用排名，不需要模型；
- **重排（rerank）** 用交叉编码器逐对打分，精度高但慢（每对一次推理）。

两者都能提升排序质量，但代价差两个数量级。分开实现才能测出
"rerank 那 200ms 换来的 MRR 提升值不值"。

## 关于"能不能回答"

`docs/PROMPT.md` 要求"无依据拒答"。A11 实测发现**不能用距离阈值判断**：

    unanswerable 查询的 top1 距离：0.467 / 0.437
    但 4 条有答案的查询距离更高：0.436 / 0.386 / 0.386 / 0.366

阈值设 0.45 会误杀"客户名单属于哪个密级"（距离 0.436 但有答案）；
设 0.50 又拦不住"公司股票期权怎么兑现"（0.467）。

所以拒答必须靠**生成后的自省**（LLM 判断"给定材料能否回答"），
而不是检索分数。这是 A11 用数据推翻直觉的一个具体例子。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag_adapters.retrieval.fusion import DEFAULT_RRF_K
from rag_core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """检索链路配置。

    **每个能力一个开关**，是为了能"只加一个节点"地做对比实验。
    把多个开关打包成"高级模式"会让指标变化无法归因。
    """

    # ---- 召回 ----
    use_keyword: bool = False
    """是否启用关键词检索并与向量检索融合（混合检索）。"""

    rrf_k: int = DEFAULT_RRF_K
    """RRF 平滑常数。默认 60（原论文值）。"""

    vector_limit: int = 20
    """向量检索召回条数（融合前）。比最终 limit 大，给融合/重排留余量。"""

    keyword_limit: int = 20
    """关键词检索召回条数。"""

    final_limit: int = 5
    """最终送给 LLM 的材料条数。太多会稀释上下文且增加成本。"""

    # ---- 改写 ----
    use_rewrite: bool = False
    """是否用 LLM 改写查询。

    适用：口语化提问、指代不明（"这个怎么报销"）、多轮对话。
    代价：一次额外的 LLM 调用（约 300~800ms）。
    """

    # ---- 重排 ----
    use_rerank: bool = False
    """是否用 reranker 重排候选。

    适用：候选多（20+）且 embedding 区分度不足时。
    代价：每对 (query, doc) 一次推理，20 条候选约 100~300ms。
    """

    rerank_top_n: int = 10
    """重排的候选数（从融合结果里取前 N 条去重排）。"""

    # ---- 自省 ----
    use_self_check: bool = False
    """是否在生成后自省"答案是否有依据"。

    这是**唯一可靠的拒答机制**（距离阈值已验证不可行，见模块 docstring）。
    代价：一次额外的 LLM 调用。
    """


@dataclass(slots=True)
class RetrievedChunk:
    """一条检索结果（图内部流转用）。"""

    chunk_id: str
    content: str
    score: float
    """当前阶段的分数。语义随阶段变化：向量距离 → RRF 分 → rerank 分。

    这就是为什么要在 `stage` 里记录它是什么分数——
    混用不同阶段的分数做阈值判断是经典错误。
    """

    stage: str = "vector"
    """当前分数来自哪个阶段：vector / rrf / rerank。"""

    ranks: dict[str, int] = field(default_factory=dict)
    """各路的排名（混合检索时有用）。"""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalState:
    """图的状态（在节点之间流转）。

    设计成"累积式"：每个节点只往里面加自己产出的东西，
    不删不改已有字段。这样任何节点的输出都能被后续节点与
    外层观测者（链路可视化）读到。
    """

    query: str
    """用户原始查询。**永不改写**——改写结果放 `rewritten_query`，
    这样对比实验与排障时能看出"改写导致的差异"。"""

    rewritten_query: str | None = None
    candidates: list[RetrievedChunk] = field(default_factory=list)
    """当前候选集（每阶段替换顺序，但保留 chunk_id 集合）。"""

    final: list[RetrievedChunk] = field(default_factory=list)
    """最终送给 LLM 的材料。"""

    answer: str | None = None
    self_check_passed: bool | None = None
    """自省结果：给定材料能否回答。None 表示未执行自省。"""

    timings: dict[str, float] = field(default_factory=dict)
    """各节点耗时（毫秒）。B4 的链路可视化直接读它。"""

    def record_timing(self, node: str, started: float) -> None:
        self.timings[node] = (time.perf_counter() - started) * 1000


__all__ = [
    "RetrievalConfig",
    "RetrievalState",
    "RetrievedChunk",
]
