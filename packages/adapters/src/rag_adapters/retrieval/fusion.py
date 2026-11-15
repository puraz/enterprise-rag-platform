"""RRF（Reciprocal Rank Fusion）：融合多路检索结果。

## 为什么需要融合

向量检索擅长语义相近（"住宿标准" ↔ "每晚不超过 600 元"），
但不擅长精确关键词（专有名词、编号、错误码）：
embedding 会把"E1024"和"E1025"编码得几乎一样。

关键词检索（BM25/全文）正好相反：精确匹配强，同义改写弱
（"迟到扣钱" 匹配不到 "迟到超过三十分钟计为旷工"）。

两路都跑再融合，覆盖率明显高于单路。这是混合检索的标准做法。

## 为什么用 RRF 而不是分数加权

分数加权（`0.7 * vec_score + 0.3 * bm25_score`）看起来更"精确"，
但有个致命问题：**两路分数量纲不同且不稳定**。

- 余弦距离范围 0~2，实际集中在 0.2~0.6；
- `ts_rank` 范围依赖查询长度与文档长度，可能是 0.001 也可能是 0.3。

归一化能"解决"量纲问题，但归一化本身依赖当前结果集的分布——
同一个 chunk，在不同查询里归一化后的分数完全不同。于是权重
`0.7/0.3` 的含义变得不可解释，调参无从下手。

RRF 只用**排名**，不用分数：

    score(d) = Σ_i 1 / (k + rank_i(d))

优点：
- 免疫量纲差异；
- 对单路结果的分数异常（某个检索器的分数尺度漂移）不敏感；
- 只有一个参数 k（默认 60，来自原论文），且它对结果影响很小。

## k 的作用

k 越大，排名靠前与靠后的差距越小（更"民主"）；k 越小，
头部结果权重越集中。60 是原论文的经验值，在多数场景下接近最优。
本实现不做自动调参——A11 的评估会用黄金集验证它是否真的有效。

## 一个 chunk 只出现一次

融合后按 chunk_id 去重。不去重的话，两路都命中的 chunk 会出现两条，
占掉结果位（用户看到重复引用），而它本该因为"两路都命中"而排名更靠前。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_core.logging import get_logger

log = get_logger(__name__)

# RRF 的标准平滑常数。原论文《Reciprocal Rank Fusion outperforms Condorcet
# and individual Rank Learning Methods》(2009) 用 60。
DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class RankedItem:
    """一路检索结果里的一个条目。

    `score` 只用于调试与展示（如向量距离），**不参与融合计算**——
    RRF 只用 `rank`。
    """

    item_id: str
    rank: int
    score: float


@dataclass(slots=True)
class FusedItem:
    """融合后的条目。"""

    item_id: str
    rrf_score: float
    """RRF 融合分，越大越相关。"""

    ranks: dict[str, int]
    """各路的排名（如 {"vector": 1, "keyword": 5}），用于调试与展示。"""


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedItem]:
    """融合多路排名结果。

    Args:
        ranked_lists: 来源名 → 按相关性降序排列的 ID 列表。
                      来源名进日志与 `ranks`，便于回答"这条是哪路召回的"。
        k: RRF 平滑常数，默认 60。

    Returns:
        按 RRF 分数降序排列的融合结果。

    **注意 rank 从 1 开始**（不是 0）：`1/(k + 0)` 会让第一名
    得到 `1/60` 而非 `1/61`，虽然差别小，但与论文公式不一致，
    且会让 `k=0` 时出现除零。
    """
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}
    ranks_by_item: dict[str, dict[str, int]] = {}

    for source, ids in ranked_lists.items():
        seen_in_source: set[str] = set()
        for rank, item_id in enumerate(ids, start=1):
            # 同一路里重复出现只算第一次的排名。
            # 检索器不该返回重复项，但真出现时按"最好的排名"计，
            # 否则重复项会被双重计分，把一条结果推到不该有的位置。
            if item_id in seen_in_source:
                log.warning(
                    "rrf_duplicate_in_source",
                    source=source,
                    item_id=str(item_id),
                    rank=rank,
                )
                continue
            seen_in_source.add(item_id)

            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            ranks_by_item.setdefault(item_id, {})[source] = rank

    fused = [
        FusedItem(item_id=item_id, rrf_score=score, ranks=ranks_by_item[item_id])
        for item_id, score in scores.items()
    ]
    # 稳定排序：分数相同时按 item_id 排，保证可复现。
    # 不稳定排序会让同样的输入产生不同顺序，评估结果不可比。
    fused.sort(key=lambda item: (-item.rrf_score, item.item_id))
    return fused


def fuse_two(
    vector_ids: Sequence[str],
    keyword_ids: Sequence[str],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedItem]:
    """混合检索的便捷入口（向量 + 关键词两路）。"""
    return reciprocal_rank_fusion({"vector": vector_ids, "keyword": keyword_ids}, k=k)


__all__ = [
    "DEFAULT_RRF_K",
    "FusedItem",
    "RankedItem",
    "fuse_two",
    "reciprocal_rank_fusion",
]
