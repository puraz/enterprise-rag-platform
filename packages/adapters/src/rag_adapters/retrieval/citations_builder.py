"""答案引用组装：从 LLM 输出 + 材料列表产出 `AnswerWithCitations`。

## 完整流程

    LLM 输出: "住宿上限 600 元 [1]，连续出差可上浮 [2]"
      │
      ├─ 1. 解析 `[n]` 标记 → 引用了第 1、2 条材料
      ├─ 2. 提取每条引用对应的**答案片段**
      │      （`[1]` 前的那个句子"住宿上限 600 元"）
      ├─ 3. 在该材料的 chunk 内定位这段文字 → 字符区间
      ├─ 4. 字符区间 → 页码（用 pages 映射，处理跨页 chunk）
      └─ 5. 组装 Citation + 校验（index 连续、越权检查）

## 为什么第 2 步要"提取答案片段"而不是整条引用就用整个 chunk

因为高亮整个 chunk 对用户没用——一个 chunk 500 字，
用户看到一大片黄色，不知道哪句是依据。

提取"含 `[n]` 的那个句子"是最自然的粒度：
它就是模型认为"这句话需要依据"的那句话。

## 解析 `[n]` 的三种真实形态

1. `[1]` —— 标准形态（提示词要求的）
2. `[1][2]` / `[1,2]` / `[1-3]` —— 模型常见的合并写法
3. `【1】` —— 全角括号（中文模型偶发）

不处理 2、3 两种情况会让"模型明明引用了但解析不到"，
表现为引用列表缺失——而答案里明明有 `[2]` 标记，用户点了没反应。

## 引用越界时怎么办

模型可能引用 `[5]` 但只提供了 3 条材料（幻觉编号）。
**必须丢弃越界引用并记录**，而不是抛异常：
- 抛异常会让整个问答失败（因为编号错就丢答案，代价太大）；
- 静默保留会让前端拿着一个不存在的 index 去查，显示错误或空白。

记录日志 + 丢弃，让答案仍可用，同时在可观测性上暴露问题。
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from rag_adapters.retrieval.citation import (
    LocationResult,
    PageSpan,
    build_quote,
    locate_in_chunk,
    page_for_offset,
)
from rag_contracts.api.citations import MAX_QUOTE_LENGTH, AnswerWithCitations, Citation
from rag_core.logging import get_logger

log = get_logger(__name__)

# 引用标记的正则。
#
# 覆盖 `[1]`、`[1,2]`、`[1-3]`、`[ 1 ]`、`【1】`。
# 用 finditer 而不是 findall：需要标记在原文中的**位置**，
# 以便提取"标记前的那个句子"作为答案片段。
_CITATION_PATTERN = re.compile(
    r"[\[【]\s*(\d+(?:\s*[,，\-–]\s*\d+)*)\s*[\]】]",
)

# 句末标点。用于切分"含引用的那个句子"。
_SENTENCE_ENDINGS = "。！？；.!?;\n"


@dataclass(frozen=True, slots=True)
class Material:
    """一条检索材料（组装引用所需的全部信息）。

    与 `RetrievedChunk` 的区别：这里带**数据库侧的定位信息**
    （page_no / char_start / pages），因为组装引用需要它们。
    `RetrievedChunk` 只带检索相关信息（内容、分数）。
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    content: str
    char_start: int | None = None
    char_end: int | None = None
    page_no: int | None = None
    bbox: dict[str, float] | None = None
    section_path: tuple[str, ...] = ()
    relevance_score: float | None = None
    pages: tuple[PageSpan, ...] = ()
    """该文档的页区间（用于把 char 偏移映射成页码）。"""


@dataclass(slots=True)
class AssemblyStats:
    """组装过程的统计（供可观测性与评估）。"""

    markers_found: int = 0
    """答案里出现的引用标记总数（含越界的）。"""

    citations_built: int = 0
    """成功产出且通过校验的引用数。"""

    out_of_range: int = 0
    """越界编号数（模型引用了不存在的材料）。"""

    locate_failed: int = 0
    """定位失败数（引用了但找不到对应原文，可能引用错配）。"""

    @property
    def locate_success_rate(self) -> float:
        if self.citations_built == 0:
            return 0.0
        return (self.citations_built - self.locate_failed) / self.citations_built


def parse_citation_markers(answer: str) -> list[tuple[int, int, int]]:
    """解析答案里的引用标记。

    Returns:
        `(材料序号, 标记起点, 标记终点)` 列表，按出现顺序。

    展开 `[1,2]` 这类合并写法：一个标记可能对应多条材料。
    """
    results: list[tuple[int, int, int]] = []
    for match in _CITATION_PATTERN.finditer(answer):
        numbers_text = match.group(1)
        for number in _expand_numbers(numbers_text):
            results.append((number, match.start(), match.end()))
    return results


def _expand_numbers(text: str) -> list[int]:
    """展开 `1,2` / `1-3` 形式的编号列表。"""
    numbers: list[int] = []
    for part in re.split(r"\s*[,，]\s*", text):
        part = part.strip()
        if not part:
            continue
        # 区间写法 `1-3`
        range_match = re.fullmatch(r"(\d+)\s*[\-–]\s*(\d+)", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            # 防御：区间过大（如 `1-9999`）会导致引用列表爆炸
            if 0 < start <= end <= start + 20:
                numbers.extend(range(start, end + 1))
            continue
        if part.isdigit():
            numbers.append(int(part))
    return numbers


def extract_answer_fragment(answer: str, *, marker_start: int) -> str:
    """提取引用标记所在的那个句子。

    从标记位置向前找最近的句末标点（或段首），向后找最近的句末标点。
    这段文字就是"模型认为需要依据的那句话"。

    **为什么向后也要找**：有些写法是 `[1] 住宿上限是 600 元`（标记在前）。
    只向前找会得到空片段。
    """
    # 向前找句子起点
    start = 0
    for index in range(marker_start - 1, -1, -1):
        if answer[index] in _SENTENCE_ENDINGS:
            start = index + 1
            break

    # 向后找句子终点：从标记之后开始（跳过标记本身）
    end = len(answer)
    search_from = marker_start
    # 先跳过标记（`[1]` 这种），避免把标记里的字符当句末
    bracket_end = answer.find("]", search_from)
    fullwidth_end = answer.find("】", search_from)
    candidates = [pos for pos in (bracket_end, fullwidth_end) if pos != -1]
    if candidates:
        search_from = min(candidates) + 1
    for index in range(search_from, len(answer)):
        if answer[index] in _SENTENCE_ENDINGS:
            end = index + 1
            break

    fragment = answer[start:end].strip()
    # 去掉标记本身，只留文字（标记不是"答案内容"）
    fragment = _CITATION_PATTERN.sub("", fragment).strip()
    return fragment


def assemble_citations(
    *,
    answer: str,
    materials: Sequence[Material],
    visible_document_ids: set[uuid.UUID] | None = None,
) -> tuple[AnswerWithCitations, AssemblyStats]:
    """把 LLM 输出组装成带引用的答案。

    Args:
        answer: LLM 原始输出（含 `[n]` 标记）。
        materials: 提供给 LLM 的材料，**顺序与提示词里的编号一致**
            （第 1 条对应 `[1]`）。
        visible_document_ids: 若提供，引用的文档必须在此集合内
            （防越权引用的第二道校验）。

    Returns:
        `(带引用的答案, 统计)`。
    """
    stats = AssemblyStats()
    markers = parse_citation_markers(answer)
    stats.markers_found = len(markers)

    # 按材料序号去重：同一材料被多次引用只保留第一条（index 连续的要求）。
    # 记录"首次出现顺序"，让引用编号按答案里第一次出现的顺序排。
    seen: dict[int, int] = {}  # 材料序号 → 引用 index
    citations: list[Citation] = []

    for material_number, marker_start, _marker_end in markers:
        if material_number in seen:
            continue

        material_index = material_number - 1
        if material_index < 0 or material_index >= len(materials):
            stats.out_of_range += 1
            log.warning(
                "citation_out_of_range",
                marker=material_number,
                material_count=len(materials),
                note="模型引用了不存在的材料编号，已丢弃该引用",
            )
            continue

        material = materials[material_index]

        # 越权校验：引用的文档必须可见
        if visible_document_ids is not None and material.document_id not in visible_document_ids:
            log.error(
                "citation_document_not_visible",
                document_id=str(material.document_id),
                chunk_id=str(material.chunk_id),
                note="引用了用户无权访问的文档，已丢弃（这是安全事件，需排查）",
            )
            continue

        fragment = extract_answer_fragment(answer, marker_start=marker_start)
        location = _locate(material=material, fragment=fragment)

        if not location.located:
            stats.locate_failed += 1

        citation_index = len(citations) + 1
        seen[material_number] = citation_index
        citations.append(
            Citation(
                index=citation_index,
                document_id=material.document_id,
                document_title=material.document_title,
                chunk_id=material.chunk_id,
                chunk_index=material.chunk_index,
                page_no=location.page_no,
                char_start=location.char_start,
                char_end=location.char_end,
                bbox=material.bbox,
                section_path=list(material.section_path),
                quote=location.quote
                or build_quote(
                    material.content,
                    start=0,
                    end=len(material.content),
                    max_length=MAX_QUOTE_LENGTH,
                ),
                relevance_score=material.relevance_score,
            )
        )

    stats.citations_built = len(citations)

    result = AnswerWithCitations(
        answer=answer,
        refused=False,
        citations=citations,
        degraded=False,
    )
    log.info(
        "citations_assembled",
        markers=stats.markers_found,
        built=stats.citations_built,
        out_of_range=stats.out_of_range,
        locate_failed=stats.locate_failed,
        locate_success_rate=round(stats.locate_success_rate, 3),
    )
    return result, stats


def _locate(*, material: Material, fragment: str) -> LocationResult:
    """在材料内定位答案片段。"""
    if not fragment:
        # 提取不到片段（标记孤立，如答案就是 "[1]"）：退化为整块定位
        return _whole_chunk_location(material, reason="no_fragment")

    window = locate_in_chunk(
        answer_fragment=fragment,
        chunk_text=material.content,
        chunk_char_start=material.char_start or 0,
    )
    if window is None:
        # 定位失败：这是**有价值的信号**（引用可能错配，
        # 或 LLM 大幅改写了原文）。返回整块位置让前端仍能高亮。
        return _whole_chunk_location(material, reason="no_match")

    relative_start, relative_end = window
    quote = build_quote(
        material.content,
        start=relative_start,
        end=relative_end,
        max_length=MAX_QUOTE_LENGTH,
    )

    # 相对偏移 → 文档全文偏移
    absolute_start: int | None = None
    absolute_end: int | None = None
    if material.char_start is not None:
        absolute_start = material.char_start + relative_start
        absolute_end = absolute_start + (relative_end - relative_start)

    # 页码：优先用"定位到的位置"映射（处理跨页 chunk），
    # 退而用 chunk 的起始页。
    page_no = material.page_no
    if absolute_start is not None and material.pages:
        page_no = page_for_offset(material.pages, absolute_start) or material.page_no

    return LocationResult(
        located=True,
        char_start=absolute_start,
        char_end=absolute_end,
        page_no=page_no,
        quote=quote,
    )


def _whole_chunk_location(material: Material, *, reason: str) -> LocationResult:
    """降级：无法精确定位时的返回。

    ## 为什么 `page_no` 返回 None 而不是 chunk 的起始页

    实测发现的坑（4 页 PDF，每页一个 chunk）：如果答案引用的内容
    **不在**这个 chunk 里（引用错配），返回 chunk 起始页会让用户
    跳到完全错误的位置——比"不知道在哪"更糟。

    实测记录：

        引用第3页内容 → 声称 page=1 ✗
        引用第4页内容 → 声称 page=1 ✗

    正确做法是**诚实地说"定位失败"**：

    - `located=False` 告诉上层这是降级结果；
    - `page_no=None` 让前端知道"无法跳页"（可只显示 quote 文本）；
    - `quote` 仍给整块内容，前端可退化为"显示这段原文"。

    宁可说"我不知道在哪一页"，也不要给一个错的页码——
    错页码会让用户在一个明显不相关的位置寻找依据，
    而"无法跳转"至少是诚实的。
    """
    quote = build_quote(
        material.content,
        start=0,
        end=len(material.content),
        max_length=MAX_QUOTE_LENGTH,
    )
    return LocationResult(
        located=False,
        # 整块的 char 区间仍是真实信息（这段文字在全文的这个范围），
        # 只是"引用具体用了哪一小段"不确定。
        char_start=material.char_start,
        char_end=material.char_end,
        # **不给页码**：无法确定引用内容在哪一页。
        page_no=None,
        quote=quote,
        reason=reason,
    )


__all__ = [
    "AssemblyStats",
    "Material",
    "assemble_citations",
    "extract_answer_fragment",
    "parse_citation_markers",
]
