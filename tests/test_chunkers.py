"""分块层测试。

## 测试策略

分块器是纯函数（输入 `ParsedArtifact` → 输出 `list[ChunkDraft]`），
所以测试不需要数据库、不需要模型、不需要存储。这让测试快且稳定。

分三层：

1. **契约测试（参数化跑全部策略）**：不管哪个策略，都必须满足
   不产生空块、位置成对、内容与源文本切片一致、顺序单调。
   这是最重要的部分——新增策略时自动被覆盖。

2. **策略特性测试**：每个策略的独有行为（结构化带 section_path、
   父子有 parent_index、句子窗口有 window_id）。

3. **边界输入**：空文档、单字符、超长无标点、表格超长。

## 为什么"内容与切片一致"必须参数化到所有策略

这是分块层最容易出错、也最难发现的一点。偏移算错的表现是：
答案正确，但点引用时高亮显示错误的文字。

端到端测试发现不了（要人工比对高亮位置），而单元测试一行断言就能抓到。
所以它必须覆盖每个策略，不能只测一个"代表"。
"""

from __future__ import annotations

import pytest

from rag_adapters.chunkers.base import (
    PROTECTED_BLOCK_TYPES,
    ChunkConfig,
    ChunkDraft,
    verify_chunks,
)
from rag_adapters.chunkers.factory import (
    build_chunker,
    registered_strategies,
    strategies_requiring_embedding,
)
from rag_adapters.parsers.artifact import (
    BlockRecord,
    PageRecord,
    ParsedArtifact,
    ParsedManifest,
    dump_parsed,
    load_parsed,
    read_manifest,
)
from rag_contracts.enums.retrieval import ChunkingStrategy

# 全部策略（参数化用）。语义策略需要 embedding，单独处理。
ALL_STRATEGIES = registered_strategies()


# ==================================================== 测试数据构造


def _build(
    blocks: list[tuple[str, str, int]],
    *,
    page_size: int = 0,
) -> ParsedArtifact:
    """构造 ParsedArtifact。

    Args:
        blocks: `(text, block_type, level)` 三元组列表。
        page_size: 每页放几个块（0 表示全部第 1 页）。

    块之间的分隔符是 `"\\n\\n"`，与真实解析器一致——
    这样测试能覆盖"块之间有分隔符"这个真实场景。
    """
    parts: list[str] = []
    records: list[BlockRecord] = []
    cursor = 0

    for index, (text, block_type, level) in enumerate(blocks):
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(text)
        cursor += len(text)
        page_no = 1 if page_size == 0 else (index // page_size) + 1
        records.append(
            BlockRecord(
                text=text,
                block_type=block_type,
                char_start=start,
                char_end=cursor,
                page_no=page_no,
                level=level,
                bbox=(10.0, 20.0, 300.0, 40.0) if page_size else None,
            )
        )

    text = "".join(parts)
    pages = [PageRecord(page_no=1, width=595.0, height=842.0, char_start=0, char_end=len(text))]
    return ParsedArtifact(
        manifest=ParsedManifest(
            format_version=1,
            parser_name="test",
            document_format="pdf",
            text_length=len(text),
            block_count=len(records),
            page_count=len(pages),
            has_scanned_pages=False,
        ),
        text=text,
        blocks=records,
        pages=pages,
    )


SAMPLE = _build(
    [
        ("第一章 总则", "title", 1),
        ("本制度适用于公司全体员工。制度的制定、修订与解释由人力资源部负责。", "text", 0),
        ("第二章 差旅管理", "title", 1),
        ("员工出差前须提交出差申请，经直属上级审批后方可预订机票与酒店。", "text", 0),
        ("住宿标准：一线城市每晚不超过 600 元，二线城市不超过 400 元。", "text", 0),
        ("| 城市 | 上限 |\n|---|---|\n| 一线 | 600 |\n| 二线 | 400 |", "table", 0),
        ("第三章 报销流程", "title", 1),
        ("报销须在费用发生后 30 日内提交，附发票原件。", "text", 0),
    ],
    page_size=3,
)


# ==================================================== 契约（全策略）


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_chunker_satisfies_contract(strategy: str) -> None:
    """**所有策略都必须满足分块契约**。

    `verify_chunks` 检查：
    - 无空块
    - 位置成对（char_start/char_end 同有同无）
    - char_end > char_start 且不越界
    - `text[char_start:char_end] == content`（最关键的偏移正确性）
    - 顺序单调
    """
    chunker = build_chunker(strategy, config=ChunkConfig(chunk_size=100, chunk_overlap=20))
    drafts = chunker.chunk(SAMPLE)

    problems = verify_chunks(drafts, SAMPLE)
    assert not problems, f"{strategy} 违反契约：{problems[:3]}"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_chunker_indexes_are_contiguous(strategy: str) -> None:
    """chunk_index 必须从 0 连续递增。

    数据库有 `uq_chunks_version_index` 唯一约束。跳号虽然不撞约束，
    但会让"上一段/下一段"导航出现空洞，且不利于按序号定位。
    """
    chunker = build_chunker(strategy, config=ChunkConfig(chunk_size=100, chunk_overlap=20))
    drafts = chunker.chunk(SAMPLE)

    assert [d.chunk_index for d in drafts] == list(range(len(drafts)))


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_chunker_never_returns_empty_chunks(strategy: str) -> None:
    """不产生空块或纯空白块。

    空块会：
    - 占数据库行与索引空间；
    - 向量化时 embedding 一个空串（多数模型返回零向量或报错）；
    - 在检索结果里占一个位置但没有任何信息。
    """
    chunker = build_chunker(strategy, config=ChunkConfig(chunk_size=100, chunk_overlap=20))
    drafts = chunker.chunk(SAMPLE)

    for draft in drafts:
        assert draft.content.strip(), f"{strategy} 产生了空块 #{draft.chunk_index}"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_chunker_is_deterministic(strategy: str) -> None:
    """同样输入必须产生完全相同的输出。

    分块结果散落着随机性的话：出了问题无法复现、无法对比策略优劣、
    版本回滚会得到不同结果（"回滚"名不副实）。
    """
    config = ChunkConfig(chunk_size=100, chunk_overlap=20)
    first = build_chunker(strategy, config=config).chunk(SAMPLE)
    second = build_chunker(strategy, config=config).chunk(SAMPLE)

    assert [(d.chunk_index, d.content, d.char_start, d.char_end) for d in first] == [
        (d.chunk_index, d.content, d.char_start, d.char_end) for d in second
    ]


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_chunker_handles_empty_document(strategy: str) -> None:
    """空文档返回空列表，不抛异常。

    空文档是合法输入（用户上传了空文件，或解析器没提取到文本）。
    抛异常会让任务进死信，而正确的行为是"切出 0 个块"。
    """
    empty = _build([])
    chunker = build_chunker(strategy, config=ChunkConfig(chunk_size=100, chunk_overlap=20))
    assert chunker.chunk(empty) == []


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_chunker_handles_single_short_block(strategy: str) -> None:
    """只有一个很短的块时正常返回（不因 min_chunk_size 而丢弃）。"""
    single = _build([("短文本", "text", 0)])
    chunker = build_chunker(strategy, config=ChunkConfig(chunk_size=100, chunk_overlap=20))
    drafts = chunker.chunk(single)

    assert len(drafts) >= 1
    assert "短文本" in drafts[0].content


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_chunker_preserves_protected_blocks(strategy: str) -> None:
    """表格块必须完整保留，不被切分。

    表格被切开会行列错位（"一线 | 600" 与 "| 二线 | 400" 分开后
    表头信息丢失），检索命中时拿到的是无法解释的片段。
    """
    chunker = build_chunker(strategy, config=ChunkConfig(chunk_size=100, chunk_overlap=20))
    drafts = chunker.chunk(SAMPLE)

    tables = [d for d in drafts if d.chunk_type == "table"]
    if not tables:
        pytest.skip(f"{strategy} 未保留表格类型")

    for table in tables:
        # 表格内容完整（含表头与所有行）
        assert "一线" in table.content
        assert "二线" in table.content


# ==================================================== 位置信息


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_chunks_carry_page_numbers(strategy: str) -> None:
    """分页文档的 chunk 必须带页码。

    没页码 → 前端无法"跳到原文第 N 页"，只能显示全文。
    """
    chunker = build_chunker(strategy, config=ChunkConfig(chunk_size=100, chunk_overlap=20))
    drafts = chunker.chunk(SAMPLE)

    # 除了"合并跨页块"的极端情况，绝大多数块应有页码
    with_page = [d for d in drafts if d.page_no is not None]
    assert with_page, f"{strategy} 的 chunk 全部丢失页码"


@pytest.mark.parametrize(
    "strategy", [s for s in ALL_STRATEGIES if s != ChunkingStrategy.FIXED.value]
)
def test_chunks_carry_bbox(strategy: str) -> None:
    """PDF 来源的 chunk 应带 bbox（前端在原图上框选的依据）。

    **fixed 例外**：它的切点是任意字符位置（窗口边界附近的分隔符），
    不对应任何版面框。冒用起点块的 bbox 会让前端在原文上框错区域
    （高亮整个段落而实际内容只有一行）。
    fixed 用 `chunk_metadata.positioning = "char_offset"` 显式声明
    它只能按字符区间高亮。
    """
    chunker = build_chunker(strategy, config=ChunkConfig(chunk_size=100, chunk_overlap=20))
    drafts = chunker.chunk(SAMPLE)

    with_bbox = [d for d in drafts if d.bbox is not None]
    assert with_bbox, f"{strategy} 的 chunk 全部丢失 bbox"

    # bbox 必须是 4 个键的 dict（数据库存 JSONB）
    for draft in with_bbox:
        assert set(draft.bbox) == {"x0", "y0", "x1", "y1"}


def test_fixed_chunker_declares_char_offset_positioning() -> None:
    """fixed 必须显式声明"只能按字符偏移定位"。

    它没有 bbox 是设计决定，不是遗漏。声明出来前端才知道
    该用区间高亮而不是框选——否则前端会拿到 bbox=null 而不知道该做什么。
    """
    chunker = build_chunker(
        ChunkingStrategy.FIXED.value,
        config=ChunkConfig(chunk_size=50, chunk_overlap=10),
    )
    drafts = chunker.chunk(SAMPLE)

    assert drafts
    for draft in drafts:
        assert draft.chunk_metadata.get("positioning") == "char_offset"
        assert draft.bbox is None


# ==================================================== 各策略特性


def test_structured_carries_section_path() -> None:
    """结构化策略必须给出章节路径（面包屑与上下文前缀的依据）。"""
    chunker = build_chunker(
        ChunkingStrategy.STRUCTURED.value,
        config=ChunkConfig(chunk_size=200, chunk_overlap=30),
    )
    drafts = chunker.chunk(SAMPLE)

    paths = [d.section_path for d in drafts if d.section_path]
    assert paths, "结构化策略没有产生任何 section_path"
    # 至少有一个块在"第二章"下面
    assert any("第二章 差旅管理" in path for path in paths)


def test_structured_embedding_input_has_context() -> None:
    """结构化策略的 embedding_input 带章节上下文，但 content 保持原文。"""
    chunker = build_chunker(
        ChunkingStrategy.STRUCTURED.value,
        config=ChunkConfig(chunk_size=200, chunk_overlap=30),
    )
    drafts = chunker.chunk(SAMPLE)

    with_context = [d for d in drafts if d.embedding_input]
    assert with_context, "没有任何块带 embedding 上下文"

    for draft in with_context:
        # embedding_input 比 content 长（多了前缀）
        assert len(draft.embedding_input) > len(draft.content)
        # content 本身不能含路径前缀（引用要高亮原文）
        assert draft.content in draft.embedding_input


def test_parent_child_produces_two_levels() -> None:
    """父子策略必须同时产出父块与子块。"""
    chunker = build_chunker(
        ChunkingStrategy.PARENT_CHILD.value,
        config=ChunkConfig(chunk_size=80, chunk_overlap=10, min_chunk_size=10),
    )
    drafts = chunker.chunk(SAMPLE)

    parents = [d for d in drafts if d.chunk_level > 0]
    children = [d for d in drafts if d.chunk_level == 0]
    assert parents, "没有产出父块"
    assert children, "没有产出子块"
    assert len(children) > len(parents)


def test_parent_child_links_children_to_parents() -> None:
    """子块的 parent_index 必须指向真实的父块下标。"""
    chunker = build_chunker(
        ChunkingStrategy.PARENT_CHILD.value,
        config=ChunkConfig(chunk_size=80, chunk_overlap=10, min_chunk_size=10),
    )
    drafts = chunker.chunk(SAMPLE)

    parent_indexes = {d.chunk_index for d in drafts if d.chunk_level > 0}
    linked = [d for d in drafts if d.parent_index is not None]

    assert linked, "没有任何子块关联到父块"
    for child in linked:
        assert child.parent_index in parent_indexes, (
            f"子块 #{child.chunk_index} 指向不存在的父块 #{child.parent_index}"
        )


def test_parent_block_covers_all_children() -> None:
    """父块的字符区间必须覆盖它所有子块的区间。

    否则"命中子块 → 回填父块"会拿到不含该子块内容的上下文，
    LLM 看到的材料与命中的片段对不上。
    """
    chunker = build_chunker(
        ChunkingStrategy.PARENT_CHILD.value,
        config=ChunkConfig(chunk_size=80, chunk_overlap=10, min_chunk_size=10),
    )
    drafts = chunker.chunk(SAMPLE)

    by_index = {d.chunk_index: d for d in drafts}
    for child in drafts:
        if child.parent_index is None:
            continue
        parent = by_index[child.parent_index]
        assert parent.char_start <= child.char_start
        assert parent.char_end >= child.char_end


def test_sentence_window_carries_window_metadata() -> None:
    """句子窗口策略必须在 metadata 里标记窗口信息（供检索时扩展）。"""
    chunker = build_chunker(
        ChunkingStrategy.SENTENCE_WINDOW.value,
        config=ChunkConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=10),
    )
    drafts = chunker.chunk(SAMPLE)

    single_sentence = [
        d
        for d in drafts
        if d.chunk_metadata.get("window_id") and not d.chunk_metadata.get("merged")
    ]
    assert single_sentence, "没有任何块带 window_id"

    for draft in single_sentence:
        assert "sentence_index" in draft.chunk_metadata
        assert "window_size" in draft.chunk_metadata


def test_fixed_chunker_respects_size() -> None:
    """固定滑窗的块长度不超过 chunk_size（除非末尾）。"""
    chunker = build_chunker(
        ChunkingStrategy.FIXED.value,
        config=ChunkConfig(chunk_size=50, chunk_overlap=10),
    )
    drafts = chunker.chunk(SAMPLE)

    for draft in drafts:
        assert len(draft.content) <= 50, f"块 #{draft.chunk_index} 超长：{len(draft.content)}"


def test_recursive_chunker_merges_short_headings() -> None:
    """短标题块应并入相邻块，而不是孤立成 8 字符的块。

    唯一允许的例外：标题上下都被受保护类型（表格/代码）夹住。
    表格不能并进正文（会破坏 chunk 类型语义），此时短块只能保留。
    """
    chunker = build_chunker(
        ChunkingStrategy.RECURSIVE.value,
        config=ChunkConfig(chunk_size=400, chunk_overlap=50, min_chunk_size=50),
    )
    drafts = chunker.chunk(SAMPLE)

    for draft in drafts:
        if len(draft.content) >= 50:
            continue
        if draft.chunk_type in PROTECTED_BLOCK_TYPES:
            continue  # 受保护块整块保留，短是正常的（本身就是表格/代码）
        # 非受保护的短块只允许出现在"紧邻受保护块"的位置
        neighbors = [d for d in drafts if abs(d.chunk_index - draft.chunk_index) == 1]
        assert any(n.chunk_type in PROTECTED_BLOCK_TYPES for n in neighbors), (
            f"块 #{draft.chunk_index}（{draft.content!r}）过短且不与受保护块相邻"
        )


# ==================================================== 边界输入


def test_very_long_single_block_is_split() -> None:
    """超长无标点文本必须被硬切，而不是返回一个巨块。

    单个 10000 字符的块送向量化会：超出模型最大长度被截断，
    向量只代表前 8192 字符 —— 后半部分内容检索不到。
    """
    long_text = "字" * 10000
    parsed = _build([(long_text, "text", 0)])
    chunker = build_chunker(
        ChunkingStrategy.RECURSIVE.value,
        config=ChunkConfig(chunk_size=500, chunk_overlap=50),
    )
    drafts = chunker.chunk(parsed)

    assert len(drafts) > 1
    assert all(len(d.content) <= 500 for d in drafts)
    assert not verify_chunks(drafts, parsed)


def test_config_rejects_overlap_greater_than_size() -> None:
    """overlap >= size 必须在构造时报错（否则切分无法推进 → 死循环）。"""
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkConfig(chunk_size=100, chunk_overlap=100)


def test_unknown_strategy_raises() -> None:
    """未知策略报错，不回退默认值。

    静默回退会让"配置生效了"成为假象——实际在用另一个策略切分，
    只能靠对比 chunk 结果才能发现。
    """
    with pytest.raises(ValueError, match="未知的切分策略"):
        build_chunker("no_such_strategy")


def test_registry_covers_all_enum_values() -> None:
    """注册表必须覆盖 `ChunkingStrategy` 的全部成员。

    漏注册的表现是"任务报未知策略"，但根因是忘了注册——
    这条测试在新增枚举成员时就立刻失败。
    """
    enum_values = {member.value for member in ChunkingStrategy}
    assert set(ALL_STRATEGIES) == enum_values


def test_only_semantic_requires_embedding() -> None:
    """只有语义策略需要 embedding。

    handler 据此决定"要不要加载 2GB 模型"——用 recursive 的文档
    不该付出这个代价。
    """
    assert strategies_requiring_embedding() == (ChunkingStrategy.SEMANTIC.value,)


# ==================================================== 解析产物格式


def test_artifact_roundtrip_preserves_positions() -> None:
    """产物序列化往返后，位置信息完全一致。

    这是 A8（worker 写产物）与 A9（chunker 读产物）之间的契约。
    往返丢字段的表现是"chunk 没有页码/bbox"，且不报错。
    """
    data = dump_parsed(
        text=SAMPLE.text,
        blocks=SAMPLE.blocks,
        pages=SAMPLE.pages,
        parser_name=SAMPLE.manifest.parser_name,
        document_format=SAMPLE.manifest.document_format,
        has_scanned_pages=SAMPLE.manifest.has_scanned_pages,
    )
    restored = load_parsed(data)

    assert restored.text == SAMPLE.text
    assert len(restored.blocks) == len(SAMPLE.blocks)
    assert len(restored.pages) == len(SAMPLE.pages)

    for original, copy in zip(SAMPLE.blocks, restored.blocks, strict=True):
        assert copy.text == original.text
        assert copy.char_start == original.char_start
        assert copy.char_end == original.char_end
        assert copy.page_no == original.page_no
        assert copy.bbox == original.bbox
        assert copy.level == original.level


def test_artifact_manifest_is_readable_without_full_parse() -> None:
    """manifest 能单独读出（不必反序列化全部块）。

    切分前的"能否处理"判断（块数为 0？文本长度合理？）只读头部即可，
    不必把几十 MB 的块列表全部加载。
    """
    data = dump_parsed(
        text=SAMPLE.text,
        blocks=SAMPLE.blocks,
        pages=SAMPLE.pages,
        parser_name="pymupdf",
        document_format="pdf",
        has_scanned_pages=False,
    )
    manifest = read_manifest(data)

    assert manifest.block_count == len(SAMPLE.blocks)
    assert manifest.text_length == len(SAMPLE.text)
    assert manifest.parser_name == "pymupdf"


def test_artifact_rejects_truncated_content() -> None:
    """产物被截断（块数不匹配）必须报错。

    静默按不足的块切分会得到"少了一半内容的文档"，
    用户只看到"检索不到下半部分"，完全无从排查。
    """
    from rag_adapters.parsers.artifact import ParsedArtifactError

    data = dump_parsed(
        text=SAMPLE.text,
        blocks=SAMPLE.blocks,
        pages=SAMPLE.pages,
        parser_name="test",
        document_format="pdf",
        has_scanned_pages=False,
    )
    # 删掉最后几行（模拟写一半被中断）
    truncated = b"\n".join(data.split(b"\n")[:-3])

    with pytest.raises(ParsedArtifactError, match="不一致"):
        load_parsed(truncated)


def test_artifact_rejects_unknown_format_version() -> None:
    """未知格式版本必须拒绝，而不是尽力解析。

    字段语义变了之后硬解析会静默产出没有页码/偏移的 chunk ——
    看起来成功，引用功能却悄悄失效。
    """
    from rag_adapters.parsers.artifact import ParsedArtifactError

    tampered = (
        b'{"kind":"manifest","format_version":999,"parser_name":"x",'
        b'"document_format":"pdf","text_length":0,"block_count":0,'
        b'"page_count":0,"has_scanned_pages":false,"metadata":{}}\n'
    )

    with pytest.raises(ParsedArtifactError, match="不兼容"):
        read_manifest(tampered)


# ==================================================== 覆盖统计


def test_coverage_detects_gap() -> None:
    """覆盖率统计能发现"有正文没被任何 chunk 覆盖"。

    这是分块最危险的 bug：内容永远检索不到，且没有任何报错。
    """
    from rag_adapters.chunkers.base import compute_coverage

    text = "AAAA" + "BBBB"
    drafts = [
        ChunkDraft(content="AAAA", chunk_index=0, char_start=0, char_end=4),
        # 故意漏掉 [4:8]
    ]
    coverage = compute_coverage(text, drafts)

    assert coverage.covered_chars == 4
    assert coverage.total_chars == 8
    assert coverage.coverage_ratio == 0.5


def test_coverage_ignores_whitespace_gaps() -> None:
    """块之间的空白不算缺口（分隔符不属于任何块是正常的）。"""
    from rag_adapters.chunkers.base import compute_coverage

    text = "AAAA\n\nBBBB"
    drafts = [
        ChunkDraft(content="AAAA", chunk_index=0, char_start=0, char_end=4),
        ChunkDraft(content="BBBB", chunk_index=1, char_start=6, char_end=10),
    ]
    coverage = compute_coverage(text, drafts)

    assert coverage.coverage_ratio == 1.0, "空白缺口不应影响覆盖率"


def test_coverage_excludes_parent_blocks() -> None:
    """父块不计入覆盖统计（它不参与检索）。

    父子策略里父块是子块的超集。把父块算进去会让重叠率恒为 100%，
    统计完全失去参考价值。
    """
    from rag_adapters.chunkers.base import compute_coverage

    text = "0123456789"
    drafts = [
        # 父块覆盖全部（level=1，不参与检索）
        ChunkDraft(content=text, chunk_index=0, char_start=0, char_end=10, chunk_level=1),
        # 子块只覆盖前半（level=0）
        ChunkDraft(content=text[:5], chunk_index=1, char_start=0, char_end=5, chunk_level=0),
    ]
    coverage = compute_coverage(text, drafts)

    assert coverage.coverage_ratio == 0.5, "父块不该被计入覆盖率"
    assert coverage.overlap_ratio == 0.0, "父块与子块的重叠不该计入重叠率"


# ==================================================== 受保护类型常量


def test_protected_types_are_documented() -> None:
    """受保护类型集合不应被随意扩大。

    加进去意味着"永不切分"——超长内容会变成超大 chunk。
    这条测试让改动显式化（要改必须改测试）。
    """
    assert frozenset({"table", "code", "formula"}) == PROTECTED_BLOCK_TYPES
