"""引用对齐测试。

## 三层

1. **契约测试**：`Citation` / `AnswerWithCitations` 的校验规则
   （index 连续、不允许跳号、字段命名）。
2. **定位算法测试**：`locate_in_chunk` / `page_for_offset`
   —— 这是 A12 的技术核心。
3. **组装测试**：从 LLM 输出到 `AnswerWithCitations` 的全流程，
   含三种真实形态（标准标记、合并标记、全角标记）与异常情况
   （越界编号、引用错配、越权文档）。

## 定位算法为什么需要这么多测试

A12 的验收标准是"引用位置能回到原文对应页"。定位错了的表现是
**答案对、高亮错**——端到端测试几乎发现不了（要人工比对），
所以必须用单测把边界情况钉死。
"""

from __future__ import annotations

import uuid

import pytest

from rag_adapters.retrieval.citation import (
    PageSpan,
    build_quote,
    locate_in_chunk,
    page_for_offset,
)
from rag_adapters.retrieval.citations_builder import (
    Material,
    assemble_citations,
    extract_answer_fragment,
    parse_citation_markers,
)
from rag_contracts.api.citations import MAX_QUOTE_LENGTH, AnswerWithCitations, Citation

# ==================================================== 契约


def _citation(**overrides: object) -> Citation:
    base: dict[str, object] = {
        "index": 1,
        "document_id": uuid.uuid4(),
        "document_title": "制度.pdf",
        "chunk_id": uuid.uuid4(),
        "chunk_index": 0,
        "quote": "一线城市每晚不超过 600 元",
    }
    base.update(overrides)
    return Citation(**base)  # type: ignore[arg-type]


def test_citation_accepts_minimal_fields() -> None:
    citation = _citation()
    assert citation.index == 1
    assert citation.page_no is None
    assert citation.bbox is None
    assert citation.section_path == []


def test_citation_rejects_negative_index() -> None:
    with pytest.raises(ValueError):
        _citation(index=0)


def test_citation_rejects_page_zero() -> None:
    """页码从 1 开始（`ge=1` 拦住 0）。

    0-based 页码会让所有引用偏一页——用户翻到上一页找不到。
    """
    with pytest.raises(ValueError):
        _citation(page_no=0)


def test_citation_quote_length_limited() -> None:
    with pytest.raises(ValueError):
        _citation(quote="x" * (MAX_QUOTE_LENGTH + 1))


def test_citation_has_position() -> None:
    assert _citation().has_position() is False
    assert _citation(page_no=3).has_position() is True
    assert _citation(char_start=10, char_end=20).has_position() is True


def test_answer_requires_contiguous_citation_indexes() -> None:
    """引用 index 必须从 1 开始连续。

    跳号（1, 3）会让前端按 index 找引用时拿到错条目，
    而且不会有任何报错——只是高亮位置莫名其妙。
    """
    with pytest.raises(ValueError, match="连续"):
        AnswerWithCitations(
            answer="答案 [1][3]",
            citations=[_citation(index=1), _citation(index=3)],
        )


def test_answer_requires_sorted_citations() -> None:
    with pytest.raises(ValueError, match="升序"):
        AnswerWithCitations(
            answer="答案",
            citations=[_citation(index=2), _citation(index=1)],
        )


def test_answer_rejects_duplicate_indexes() -> None:
    with pytest.raises(ValueError, match="重复"):
        AnswerWithCitations(
            answer="答案",
            citations=[_citation(index=1), _citation(index=1)],
        )


def test_answer_accepts_empty_citations() -> None:
    """拒答时没有引用是合法的。"""
    answer = AnswerWithCitations(answer="无法回答", refused=True)
    assert answer.citations == []
    assert answer.refused is True


# ==================================================== 标记解析


def test_parse_single_marker() -> None:
    markers = parse_citation_markers("住宿上限 600 元 [1]。")
    assert [number for number, _s, _e in markers] == [1]


def test_parse_multiple_markers() -> None:
    markers = parse_citation_markers("第一点 [1]，第二点 [2]，第三点 [3]")
    assert [number for number, _s, _e in markers] == [1, 2, 3]


def test_parse_merged_marker() -> None:
    """`[1,2]` 这种合并写法要展开。"""
    markers = parse_citation_markers("这个结论来自多处 [1,2]")
    assert [number for number, _s, _e in markers] == [1, 2]


def test_parse_range_marker() -> None:
    """`[1-3]` 区间写法要展开。"""
    markers = parse_citation_markers("综合多处 [1-3]")
    assert [number for number, _s, _e in markers] == [1, 2, 3]


def test_parse_fullwidth_marker() -> None:
    """全角括号 `【1】` 是中文模型偶发写法。"""
    markers = parse_citation_markers("依据是【1】")
    assert [number for number, _s, _e in markers] == [1]


def test_parse_rejects_absurd_range() -> None:
    """`[1-9999]` 不该把引用列表炸开。"""
    markers = parse_citation_markers("恶意区间 [1-9999]")
    # 区间过大被忽略，不产生任何编号
    assert markers == []


def test_parse_no_markers() -> None:
    assert parse_citation_markers("这个答案没有引用。") == []


# ==================================================== 片段提取


def test_extract_fragment_before_marker() -> None:
    """标准形态：句子在前，标记在后。"""
    answer = "一线城市住宿上限是 600 元 [1]。二线城市是 400 元 [2]。"
    fragment = extract_answer_fragment(answer, marker_start=answer.index("[1]"))
    assert "600" in fragment
    assert "400" not in fragment


def test_extract_fragment_after_marker() -> None:
    """标记在前的写法也要能提取到内容。"""
    answer = "根据规定 [1] 住宿上限是 600 元。"
    fragment = extract_answer_fragment(answer, marker_start=answer.index("[1]"))
    assert "600" in fragment


def test_extract_fragment_strips_marker() -> None:
    """提取出的片段不含标记本身（标记不是答案内容）。"""
    answer = "住宿上限是 600 元 [1]。"
    fragment = extract_answer_fragment(answer, marker_start=answer.index("[1]"))
    assert "[1]" not in fragment
    assert "600" in fragment


def test_extract_fragment_at_start_of_answer() -> None:
    answer = "[1] 住宿上限是 600 元。"
    fragment = extract_answer_fragment(answer, marker_start=0)
    assert "600" in fragment


# ==================================================== 定位算法


def test_locate_exact_substring() -> None:
    """答案片段与原文完全一致时，应精确定位到该子串。"""
    chunk = "员工出差前须提交申请。一线城市每晚不超过 600 元，二线城市不超过 400 元。"
    window = locate_in_chunk(
        answer_fragment="一线城市每晚不超过 600 元",
        chunk_text=chunk,
        chunk_char_start=0,
    )
    assert window is not None
    start, end = window
    located = chunk[start:end]
    assert "600" in located
    assert len(located) >= len("一线城市每晚不超过 600 元") - 5


def test_locate_with_paraphrase() -> None:
    """答案被改写后仍应能定位（靠关键词覆盖度，不靠精确匹配）。"""
    chunk = (
        "差旅住宿标准：一线城市（北京、上海、广州、深圳）每晚不超过 600 元；"
        "二线城市每晚不超过 400 元；其它城市每晚不超过 300 元。"
    )
    window = locate_in_chunk(
        answer_fragment="出差住宿上限一线城市是 600 元",
        chunk_text=chunk,
        chunk_char_start=0,
    )
    assert window is not None
    start, end = window
    assert "600" in chunk[start:end]


def test_locate_prefers_rare_token_as_anchor() -> None:
    """数字这类稀有 token 应决定定位位置。

    原文里"城市"出现多次、"600"只出现一次。若用"城市"做锚点，
    会随机落在某个"城市"上；用"600"才能定位到正确位置。
    """
    chunk = (
        "一线城市每晚不超过 600 元，二线城市不超过 400 元，"
        "三线城市不超过 300 元，其它城市不超过 200 元。"
    )
    window = locate_in_chunk(
        answer_fragment="一线城市 600 元",
        chunk_text=chunk,
        chunk_char_start=0,
    )
    assert window is not None
    start, end = window
    assert "600" in chunk[start:end]
    # 不该定位到 400/300/200 那几段
    assert "400" not in chunk[start:end]


def test_locate_fails_on_unrelated_fragment() -> None:
    """完全无关的片段应定位失败（返回 None），而不是随便返回一个位置。"""
    chunk = "一线城市每晚不超过 600 元。"
    window = locate_in_chunk(
        answer_fragment="公司股票期权的行权价格是多少",
        chunk_text=chunk,
        chunk_char_start=0,
    )
    assert window is None


def test_locate_handles_empty_inputs() -> None:
    assert locate_in_chunk(answer_fragment="", chunk_text="内容", chunk_char_start=0) is None
    assert locate_in_chunk(answer_fragment="内容", chunk_text="", chunk_char_start=0) is None


def test_locate_english_and_numbers_kept_intact() -> None:
    """英文串与数字整体保留（不拆成单字），它们是定位最强的锚点。"""
    chunk = "报告渠道为 security@example.com 或内线 8888。其他渠道无效。"
    window = locate_in_chunk(
        answer_fragment="可以打 8888 或发邮件到 security@example.com",
        chunk_text=chunk,
        chunk_char_start=0,
    )
    assert window is not None
    start, end = window
    located = chunk[start:end]
    assert "8888" in located or "security@example.com" in located


# ==================================================== 页码映射


def test_page_for_offset_basic() -> None:
    pages = [
        PageSpan(page_no=1, char_start=0, char_end=100),
        PageSpan(page_no=2, char_start=100, char_end=250),
    ]
    assert page_for_offset(pages, 0) == 1
    assert page_for_offset(pages, 99) == 1
    assert page_for_offset(pages, 100) == 2
    assert page_for_offset(pages, 249) == 2


def test_page_for_offset_handles_page_boundary() -> None:
    """偏移正好落在页末尾时归前一页（char_end 是开区间）。"""
    pages = [
        PageSpan(page_no=1, char_start=0, char_end=100),
        PageSpan(page_no=2, char_start=100, char_end=200),
    ]
    assert page_for_offset(pages, 100) == 2


def test_page_for_offset_out_of_range() -> None:
    pages = [PageSpan(page_no=1, char_start=0, char_end=100)]
    assert page_for_offset(pages, 500) is None
    assert page_for_offset([], 0) is None


def test_quote_truncation_keeps_head() -> None:
    """截断保留开头（开头通常含关键信息）。"""
    text = "一线城市每晚不超过 600 元。" + "无关内容" * 100
    quote = build_quote(text, start=0, end=len(text), max_length=30)
    assert len(quote) <= 30
    assert quote.startswith("一线城市")
    assert quote.endswith("…")


def test_quote_short_text_not_truncated() -> None:
    quote = build_quote("短文本", start=0, end=3, max_length=100)
    assert quote == "短文本"
    assert not quote.endswith("…")


# ==================================================== 组装全流程


def _material(
    *,
    content: str,
    char_start: int = 0,
    page_no: int = 1,
    pages: tuple[PageSpan, ...] = (),
) -> Material:
    return Material(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="制度.pdf",
        chunk_index=0,
        content=content,
        char_start=char_start,
        char_end=char_start + len(content),
        page_no=page_no,
        relevance_score=0.25,
        pages=pages,
    )


def test_assemble_basic_citation() -> None:
    material = _material(content="一线城市每晚不超过 600 元，二线城市不超过 400 元。")
    answer = "一线城市住宿上限是 600 元 [1]。"

    result, stats = assemble_citations(answer=answer, materials=[material])

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.index == 1
    assert citation.document_id == material.document_id
    assert citation.relevance_score == 0.25
    assert stats.markers_found == 1
    assert stats.citations_built == 1
    assert stats.locate_failed == 0


def test_assemble_deduplicates_same_material() -> None:
    """同一材料被引用多次，只产出一条 Citation（index 必须连续）。"""
    material = _material(content="住宿上限 600 元。")
    answer = "住宿 600 元 [1]，所以推荐订 600 以内的 [1]。"

    result, _stats = assemble_citations(answer=answer, materials=[material])
    assert len(result.citations) == 1


def test_assemble_numbers_by_first_appearance() -> None:
    """引用编号按答案里第一次出现的顺序，而不是材料顺序。"""
    materials = [
        _material(content="一线城市 600 元。"),
        _material(content="二线城市 400 元。"),
    ]
    # 先引用材料 2，再引用材料 1
    answer = "二线城市是 400 元 [2]。一线城市是 600 元 [1]。"

    result, _stats = assemble_citations(answer=answer, materials=materials)

    assert [c.index for c in result.citations] == [1, 2]
    # index 1 对应材料 2（二线城市）
    assert result.citations[0].document_id == materials[1].document_id
    assert "400" in result.citations[0].quote


def test_assemble_drops_out_of_range_marker() -> None:
    """模型引用了不存在的材料编号时丢弃该引用，但保留答案。

    抛异常会让整个问答失败（编号错就丢答案，代价太大）；
    静默保留会让前端拿不存在的 index 去查。
    """
    material = _material(content="一线城市 600 元。")
    answer = "住宿 600 元 [1]，另外 [5] 也有说明。"

    result, stats = assemble_citations(answer=answer, materials=[material])

    assert len(result.citations) == 1
    assert stats.out_of_range == 1
    # 答案本身不受影响
    assert result.answer == answer


def test_assemble_marks_locate_failure() -> None:
    """引用错配（答案说的内容不在该材料里）应记录为定位失败。"""
    material = _material(content="公司为员工缴纳五险一金。")
    # 模型声称这段讲住宿，实际不是
    answer = "住宿上限是 600 元 [1]。"

    result, stats = assemble_citations(answer=answer, materials=[material])

    assert len(result.citations) == 1
    assert stats.locate_failed == 1
    # 仍返回引用（降级为整块），但 has_position 仍可能为 True（用了 chunk 的位置）
    assert result.citations[0].quote  # 至少有整块内容


def test_assemble_rejects_invisible_document() -> None:
    """引用了不可见文档时必须丢弃（安全兜底）。

    SQL 已过滤，但模型可能"记错来源"，压缩/回填也可能引入未过滤内容。
    引用越界是最隐蔽的泄漏形态——它嵌在答案里。
    """
    material = _material(content="机密文档内容。")
    answer = "这是机密内容 [1]。"

    result, stats = assemble_citations(
        answer=answer,
        materials=[material],
        visible_document_ids={uuid.uuid4()},  # 不包含该材料的文档
    )

    assert result.citations == []
    assert stats.citations_built == 0


def test_assemble_accepts_visible_document() -> None:
    material = _material(content="一线城市 600 元。")
    answer = "住宿 600 元 [1]。"

    result, _stats = assemble_citations(
        answer=answer,
        materials=[material],
        visible_document_ids={material.document_id},
    )
    assert len(result.citations) == 1


def test_assemble_no_markers() -> None:
    """答案里没有引用标记时返回空引用列表（不是错误）。"""
    result, stats = assemble_citations(
        answer="这是一个不需要引用的答案。",
        materials=[_material(content="内容")],
    )
    assert result.citations == []
    assert stats.markers_found == 0


def test_assemble_uses_correct_page_for_cross_page_chunk() -> None:
    """跨页 chunk 的引用应定位到**内容实际所在页**。

    `chunk.page_no` 只记录起始页。若答案引用的是 chunk 后半段
    （第 3 页的内容），用起始页（第 2 页）会让用户翻错页。
    """
    # chunk 从第 2 页末尾开始，跨到第 3 页
    content = (
        "二线城市的住宿标准是每晚不超过 400 元。" + "补充说明。" * 5 + "三线城市不超过 300 元。"
    )
    pages = (
        PageSpan(page_no=2, char_start=0, char_end=30),
        PageSpan(page_no=3, char_start=30, char_end=len(content)),
    )
    material = _material(content=content, char_start=0, page_no=2, pages=pages)

    # 引用指向 chunk 后半段（第 3 页）
    answer = "三线城市不超过 300 元 [1]。"
    result, _stats = assemble_citations(answer=answer, materials=[material])

    assert len(result.citations) == 1
    assert result.citations[0].page_no == 3, "跨页 chunk 应定位到内容实际所在的第 3 页"


def test_assemble_carries_section_path() -> None:
    """章节路径要传递到引用（前端面包屑用）。"""
    material = Material(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="制度.pdf",
        chunk_index=5,
        content="一线城市 600 元。",
        char_start=0,
        char_end=9,
        page_no=1,
        section_path=("第三章 差旅管理", "3.2 住宿标准"),
    )
    answer = "一线城市 600 元 [1]。"
    result, _stats = assemble_citations(answer=answer, materials=[material])

    assert result.citations[0].section_path == ["第三章 差旅管理", "3.2 住宿标准"]
    assert result.citations[0].chunk_index == 5


# ==================================================== 引用错配检测（实测补充）


def test_locate_rejects_chunk_sharing_only_stopwords() -> None:
    """只共享停用词的 chunk 不该被判为"定位成功"。

    ## 这是实测发现的真实误判

    答案"claims must be filed within 30 days" 与 chunk
    "travel request must be approved before booking" 只共享
    "must" 与 "be"。在短 chunk 里所有词都"出现 ≤2 次"，
    按"稀有 token"的定义它们都算稀有 → 误判为定位成功 →
    引用跳到完全错误的页。

    修复：加英文停用词表（must/be/the/policy 等），
    让这类通用词不参与定位判据。
    """
    chunk = "Page one: travel request must be approved before booking."
    window = locate_in_chunk(
        answer_fragment="The policy states claims must be filed within 30 days",
        chunk_text=chunk,
        chunk_char_start=0,
    )
    assert window is None, "只共享停用词不应定位成功"


def test_locate_accepts_chunk_with_real_keywords() -> None:
    """真正含关键词的 chunk 必须能定位（与上一条对照）。"""
    chunk = "Page three: claims must be filed within 30 days with receipts."
    window = locate_in_chunk(
        answer_fragment="The policy states claims must be filed within 30 days",
        chunk_text=chunk,
        chunk_char_start=0,
    )
    assert window is not None
    start, end = window
    assert "30" in chunk[start:end] or "claims" in chunk[start:end]


def test_assemble_reports_none_page_on_mismatch() -> None:
    """引用错配时页码必须是 None（诚实），不能拿 chunk 起始页冒充。

    实测场景：4 页 PDF，每页一个 chunk。答案引用第 3 页的内容，
    但被关联到第 1 页的 chunk。返回 page=1 会让用户翻到完全错误的位置
    ——比"不知道在哪"更糟。
    """
    material = _material(
        content="Page one: travel request must be approved before booking.",
        page_no=1,
    )
    answer = "The policy states claims must be filed within 30 days [1]."

    result, stats = assemble_citations(answer=answer, materials=[material])

    assert stats.locate_failed == 1
    assert result.citations[0].page_no is None, "定位失败时不该给页码"
    # 但 quote 仍给出（前端可退化为显示原文片段）
    assert result.citations[0].quote


def test_multi_page_citation_uses_content_page_not_chunk_start() -> None:
    """多页文档：引用页码必须是内容实际所在页，不是 chunk 起始页。"""
    # 三页内容拼成一个跨页 chunk。
    # **页边界必须用真实偏移**：手算容易错（我第一次写成 50/93，
    # 实际是 43/85），用 find() 保证与内容严格一致。
    page_texts = [
        "Page one: travel request must be approved. ",
        "Page two: hotel cap is 600 CNY per night. ",
        "Page three: claims must be filed within 30 days.",
    ]
    content = "".join(page_texts)
    boundaries = []
    cursor = 0
    for page_no, page_text in enumerate(page_texts, start=1):
        boundaries.append(
            PageSpan(page_no=page_no, char_start=cursor, char_end=cursor + len(page_text))
        )
        cursor += len(page_text)
    pages = tuple(boundaries)
    material = _material(content=content, char_start=0, page_no=1, pages=pages)

    # 引用第 3 页的内容
    answer = "Claims must be filed within 30 days [1]."
    result, _stats = assemble_citations(answer=answer, materials=[material])

    assert result.citations[0].page_no == 3, "应定位到内容所在的第 3 页"
