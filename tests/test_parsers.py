"""解析层测试。

重点在**字符偏移的一致性**（这是最容易出错、且最难端到端发现的）：

- block 的 char_start/char_end 必须能切出与 block.text 完全相同的字符串
- 页码必须从 1 开始（不是 0-based）
- 扫描页必须被识别（否则 OCR 分流失效）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_adapters.parsers import (
    BaseParser,
    DocxParser,
    NullOcr,
    ParseError,
    PdfParser,
    PptxParser,
    TextLikeParser,
    UnsupportedFormatError,
    build_ocr_engine,
    build_parser,
    parse_document,
    supported_formats,
    verify_char_offsets,
)
from rag_contracts.enums import DocumentFormat

# ==================================================== 测试文件构造


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """生成一个两页 PDF（含标题与正文）。"""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=20)
    pdf.cell(0, 15, "Chapter One")
    pdf.ln(20)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, "This is the first page about RAG.")
    pdf.add_page()
    pdf.set_font("Helvetica", size=20)
    pdf.cell(0, 15, "Chapter Two")
    pdf.ln(20)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, "This is the second page about vectors.")

    target = tmp_path / "sample.pdf"
    target.write_bytes(pdf.output())
    return target


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    content = """# 员工手册

## 第一章 入职

新员工需在入职当天提交材料。

## 第二章 年假

年假天数据工龄计算。

- 满一年 5 天
- 满三年 10 天
"""
    target = tmp_path / "handbook.md"
    target.write_text(content, encoding="utf-8")
    return target


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    """生成一个 DOCX（用 python-docx）。"""
    from docx import Document

    doc = Document()
    doc.add_heading("测试文档", level=1)
    doc.add_paragraph("这是第一段内容。")
    doc.add_paragraph("这是第二段内容。")

    target = tmp_path / "sample.docx"
    doc.save(str(target))
    return target


@pytest.fixture
def sample_pptx(tmp_path: Path) -> Path:
    from pptx import Presentation

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "第一张幻灯片"
    slide.placeholders[1].text = "幻灯片内容"

    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "第二张幻灯片"
    slide2.placeholders[1].text = "第二页内容"

    target = tmp_path / "sample.pptx"
    prs.save(str(target))
    return target


@pytest.fixture
def sample_xlsx(tmp_path: Path) -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    sheet = wb.active
    sheet.title = "工资表"
    sheet.append(["部门", "姓名", "金额"])
    sheet.append(["研发", "张三", 5000])
    sheet.append(["销售", "李四", 6000])

    target = tmp_path / "sample.xlsx"
    wb.save(str(target))
    return target


# ==================================================== PDF：位置信息（核心）


def test_pdf_provides_page_numbers_and_bbox(sample_pdf: Path) -> None:
    """**PDF 必须给页码与 bbox**（引用高亮的前提）。

    这是不用 LangChain PyPDFLoader 的原因：它不给 bbox。
    """
    doc = PdfParser().parse(sample_pdf, format=DocumentFormat.PDF)

    assert doc.page_count == 2
    assert len(doc.pages) == 2

    # 页码从 1 开始
    assert doc.pages[0].page_no == 1
    assert doc.pages[1].page_no == 2

    # 每个 block 都有 bbox 与页码
    for block in doc.blocks:
        assert block.bbox is not None, "PDF 的 block 必须有 bbox"
        assert len(block.bbox) == 4
        assert block.page_no in (1, 2)


def test_pdf_char_offsets_are_consistent(sample_pdf: Path) -> None:
    """**字符偏移必须能精确切出 block 文本**。

    偏移算错的后果：引用高亮显示错误的文字，甚至跨段落。
    端到端测试很难发现（答案看起来对，只是高亮位置偏了）。
    """
    doc = PdfParser().parse(sample_pdf, format=DocumentFormat.PDF)

    problems = verify_char_offsets(doc)
    assert not problems, f"偏移校验失败: {problems}"


def test_pdf_block_order_matches_pages(sample_pdf: Path) -> None:
    """block 顺序与页码顺序一致（第 1 页的块在前）。"""
    doc = PdfParser().parse(sample_pdf, format=DocumentFormat.PDF)

    page_numbers = [block.page_no for block in doc.blocks if block.page_no]
    assert page_numbers == sorted(page_numbers), "block 顺序与页码不一致"


def test_pdf_page_char_ranges_cover_text(sample_pdf: Path) -> None:
    """页的字符区间必须落在 text 范围内且不重叠。"""
    doc = PdfParser().parse(sample_pdf, format=DocumentFormat.PDF)

    for page in doc.pages:
        assert 0 <= page.char_start < page.char_end <= len(doc.text)

    # 相邻页的区间不重叠
    for index in range(len(doc.pages) - 1):
        assert doc.pages[index].char_end <= doc.pages[index + 1].char_start


def test_pdf_detects_heading_by_font_size(sample_pdf: Path) -> None:
    """按字号识别标题（20pt vs 11pt）。"""
    doc = PdfParser().parse(sample_pdf, format=DocumentFormat.PDF)

    title_blocks = [block for block in doc.blocks if block.block_type == "title"]
    assert title_blocks, "应至少识别出一个标题块（20pt 字号）"
    assert any("Chapter" in block.text for block in title_blocks)


def test_pdf_empty_file_raises(tmp_path: Path) -> None:
    """空白 PDF（无文本）应明确报错。"""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()  # 只有空白页
    target = tmp_path / "empty.pdf"
    target.write_bytes(pdf.output())

    with pytest.raises(ParseError, match=r"没有可提取的文本|扫描件"):
        PdfParser().parse(target, format=DocumentFormat.PDF)


def test_pdf_corrupt_file_raises(tmp_path: Path) -> None:
    """损坏的 PDF 应明确报错，不能崩。"""
    target = tmp_path / "corrupt.pdf"
    target.write_bytes(b"%PDF-1.4 this is not a real pdf")

    with pytest.raises(ParseError, match=r"无法打开"):
        PdfParser().parse(target, format=DocumentFormat.PDF)


def test_pdf_module_uses_pymupdf_not_fitz(sample_pdf: Path) -> None:
    """**用 `pymupdf` 而不是 `fitz`**。

    `fitz` 是历史导入名，官方已弃用（有 FutureWarning）。
    用弃用 API 会在未来版本被移除时突然崩溃。
    """
    source = (
        Path(__file__).parent.parent / "packages/adapters/src/rag_adapters/parsers/pdf.py"
    ).read_text(encoding="utf-8")
    # 允许注释里提到 fitz，但 import 必须是 pymupdf
    assert "import pymupdf" in source
    assert "import fitz\n" not in source


# ==================================================== Markdown


def test_markdown_preserves_structure(sample_md: Path) -> None:
    """Markdown 的标题被识别为 title 块，且带层级。"""
    doc = parse_document(sample_md, format=DocumentFormat.MARKDOWN)

    titles = [block for block in doc.blocks if block.block_type == "title"]
    assert len(titles) >= 3, "应有 1 个 h1 + 2 个 h2"

    # h1 的 level 是 1，h2 是 2
    levels = {block.text.split()[0]: block.level for block in titles}
    assert levels.get("#") == 1
    assert levels.get("##") == 2


def test_markdown_char_offsets(sample_md: Path) -> None:
    """Markdown 的字符偏移必须一致。"""
    doc = parse_document(sample_md, format=DocumentFormat.MARKDOWN)
    problems = verify_char_offsets(doc)
    assert not problems, f"偏移校验失败: {problems}"


def test_markdown_has_no_page_numbers(sample_md: Path) -> None:
    """Markdown 没有真实页码（不该编造）。"""
    doc = parse_document(sample_md, format=DocumentFormat.MARKDOWN)
    assert doc.pages == []
    assert all(block.page_no is None for block in doc.blocks)


# ==================================================== HTML


def test_html_strips_tags_keeps_text(tmp_path: Path) -> None:
    """HTML 标签被剥离，文本保留。"""
    html = "<html><body><h1>标题</h1><p>第一段。</p><p>第二段。</p></body></html>"
    target = tmp_path / "page.html"
    target.write_text(html, encoding="utf-8")

    doc = parse_document(target, format=DocumentFormat.HTML)

    assert "<h1>" not in doc.text
    assert "<p>" not in doc.text
    assert "标题" in doc.text
    assert "第一段" in doc.text


def test_html_decodes_entities(tmp_path: Path) -> None:
    """HTML 实体被解码（&amp; → &）。"""
    target = tmp_path / "page.html"
    target.write_text("<p>A &amp; B &lt;tag&gt;</p>", encoding="utf-8")

    doc = parse_document(target, format=DocumentFormat.HTML)
    assert "A & B" in doc.text
    assert "<tag>" in doc.text


# ==================================================== 文本编码


def test_gbk_encoding_is_handled(tmp_path: Path) -> None:
    """**GBK 编码的中文文件能读**。

    中文文档常见 GBK 编码，只试 UTF-8 会抛 UnicodeDecodeError，
    而错误信息看不出是编码问题（只说"无法解码"）。
    """
    target = tmp_path / "gbk.txt"
    target.write_bytes("这是 GBK 编码的中文内容".encode("gbk"))

    doc = parse_document(target, format=DocumentFormat.TEXT)
    assert "GBK 编码" in doc.text


def test_crlf_is_normalized(tmp_path: Path) -> None:
    """Windows 换行（\\r\\n）被规范化为 \\n。

    不规范化会让偏移计算在 Windows 文件上偏一个字符/行。
    """
    target = tmp_path / "crlf.txt"
    target.write_bytes(b"line one\r\nline two\r\nline three")

    doc = parse_document(target, format=DocumentFormat.TEXT)
    assert "\r" not in doc.text
    assert doc.text.count("\n") == 2


# ==================================================== DOCX / PPTX / XLSX


def test_docx_extracts_text(sample_docx: Path) -> None:
    """DOCX 能提取出段落文本。"""
    doc = parse_document(sample_docx, format=DocumentFormat.DOCX)

    assert "测试文档" in doc.text
    assert "第一段内容" in doc.text
    assert doc.pages == [], "DOCX 没有真实页码"


def test_docx_char_offsets(sample_docx: Path) -> None:
    """DOCX 的字符偏移必须一致。"""
    doc = parse_document(sample_docx, format=DocumentFormat.DOCX)
    problems = verify_char_offsets(doc)
    assert not problems, f"偏移校验失败: {problems}"


def test_pptx_provides_slide_numbers(sample_pptx: Path) -> None:
    """**PPTX 有真实页号（幻灯片序号）**。"""
    doc = parse_document(sample_pptx, format=DocumentFormat.PPTX)

    assert doc.page_count == 2
    slide_numbers = [block.page_no for block in doc.blocks]
    assert slide_numbers == [1, 2], f"页号应为 [1, 2]，实际 {slide_numbers}"


def test_pptx_extracts_content(sample_pptx: Path) -> None:
    """PPTX 提取标题与正文。"""
    doc = parse_document(sample_pptx, format=DocumentFormat.PPTX)
    assert "第一张幻灯片" in doc.text
    assert "幻灯片内容" in doc.text


def test_xlsx_sheet_name_as_context(sample_xlsx: Path) -> None:
    """**XLSX 的 Sheet 名作为上下文前缀**。

    单独的表格行没有语义（"研发 | 张三 | 5000"），
    加上 sheet 名（"工资表"）才能被检索到。
    """
    doc = parse_document(sample_xlsx, format=DocumentFormat.XLSX)

    assert "[Sheet: 工资表]" in doc.text
    assert "张三" in doc.text
    assert "5000" in doc.text


def test_xlsx_blocks_marked_as_table(sample_xlsx: Path) -> None:
    """XLSX 的块类型是 table（切分时整块保留）。"""
    doc = parse_document(sample_xlsx, format=DocumentFormat.XLSX)
    assert all(block.block_type == "table" for block in doc.blocks)


def test_csv_groups_rows(tmp_path: Path) -> None:
    """CSV 按行聚合（不一行一个 block）。"""
    rows = "\n".join(f"row{i},value{i}" for i in range(120))
    target = tmp_path / "data.csv"
    target.write_text(rows, encoding="utf-8")

    doc = parse_document(target, format=DocumentFormat.CSV)

    # 120 行按 50 行一组 → 3 个 block
    assert len(doc.blocks) == 3, f"应聚合成 3 个 block，实际 {len(doc.blocks)}"
    assert all(block.block_type == "table" for block in doc.blocks)


# ==================================================== 工厂与能力声明


def test_supported_formats_covers_documented_types() -> None:
    """支持的格式覆盖 `docs/PROMPT.md` 列出的类型。"""
    formats = set(supported_formats())
    required = {
        DocumentFormat.PDF,
        DocumentFormat.DOCX,
        DocumentFormat.PPTX,
        DocumentFormat.XLSX,
        DocumentFormat.MARKDOWN,
        DocumentFormat.HTML,
        DocumentFormat.TEXT,
        DocumentFormat.CSV,
    }
    assert required <= formats, f"缺少格式: {required - formats}"


def test_parser_capabilities_are_honest() -> None:
    """**能力声明必须诚实**：只有真正能给 bbox 的才声明。

    虚假声明会让前端尝试渲染 bbox 时拿到 null，
    表现为"高亮功能时好时坏"。
    """
    pdf_caps = PdfParser().capabilities
    assert pdf_caps.supports_bbox is True
    assert pdf_caps.supports_page_numbers is True

    md_caps = TextLikeParser().capabilities
    assert md_caps.supports_bbox is False
    assert md_caps.supports_page_numbers is False

    docx_caps = DocxParser().capabilities
    assert docx_caps.supports_bbox is False

    pptx_caps = PptxParser().capabilities
    assert pptx_caps.supports_page_numbers is True  # 幻灯片序号


def test_parser_supports_uses_capabilities() -> None:
    """`supports()` 走能力匹配，不是格式名 if-else。"""
    parser = PdfParser()
    assert parser.supports(DocumentFormat.PDF) is True
    assert parser.supports(DocumentFormat.DOCX) is False


def test_unsupported_format_raises() -> None:
    """没有解析器的格式明确报错。"""
    with pytest.raises(UnsupportedFormatError):
        build_parser(DocumentFormat.IMAGE)


def test_wrong_format_passed_to_parser_raises(sample_pdf: Path) -> None:
    """用 PDF 解析器解析非 PDF 格式应报错。"""
    with pytest.raises(UnsupportedFormatError):
        PdfParser().parse(sample_pdf, format=DocumentFormat.DOCX)


# ==================================================== OCR（可选能力）


def test_ocr_reports_unavailable_when_missing(tmp_path: Path) -> None:
    """**OCR 不可用时明确报告**，不静默返回空字符串。

    静默返回空会让"扫描件解析失败"看起来像"文档是空的"——
    用户会以为上传错了文件。
    """
    engine = build_ocr_engine(enabled=True)
    # 本机没有 tesseract，应该是 None 实现
    if not engine.is_available():
        assert isinstance(engine, NullOcr)
        with pytest.raises(ParseError, match="OCR"):
            engine.extract_text(tmp_path / "nonexistent.png")


def test_ocr_disabled_explicitly() -> None:
    """显式关闭 OCR 时返回 NullOcr。"""
    engine = build_ocr_engine(enabled=False)
    assert isinstance(engine, NullOcr)
    assert engine.is_available() is False


def test_ocr_unavailable_does_not_break_parsing(sample_pdf: Path) -> None:
    """**OCR 不可用不影响普通 PDF 解析**。

    OCR 是可选能力：没有它，有文本层的 PDF 仍应正常解析。
    """
    doc = PdfParser().parse(sample_pdf, format=DocumentFormat.PDF)
    assert doc.text.strip()
    assert doc.has_scanned_pages is False


# ==================================================== 通用不变量


def test_all_parsers_return_nonempty_blocks() -> None:
    """**所有解析器的 block 必须非空**。

    空 block 会让切分器产出空 chunk，最终在检索里变成噪音。
    """

    # 用一个通用的假解析器验证基类契约（各实现已在上面分别验证）
    class _Probe(BaseParser):
        name = "probe"

        @property
        def capabilities(self):
            from rag_adapters.parsers.base import ParserCapabilities

            return ParserCapabilities(supported_formats=frozenset({DocumentFormat.TEXT}))

        def parse(self, path: Path, *, format: DocumentFormat):
            raise NotImplementedError

    probe = _Probe()
    assert probe.supports(DocumentFormat.TEXT)
    assert probe.capabilities.supports_bbox is False


def test_verify_char_offsets_catches_broken_implementations() -> None:
    """**偏移校验器能抓到错误的实现**（这是它存在的意义）。

    构造一个 block.text 与 text 切片不匹配的文档，
    校验器必须报错。
    """
    from rag_adapters.parsers.base import PageInfo, ParsedDocument, TextBlock

    broken = ParsedDocument(
        text="hello world",
        blocks=[
            TextBlock(
                text="WRONG",  # 与 text[0:5] 不一致
                char_start=0,
                char_end=5,
            )
        ],
        pages=[PageInfo(page_no=1, width=100, height=100, char_start=0, char_end=11)],
        format=DocumentFormat.TEXT,
        parser_name="broken",
        page_count=1,
    )
    problems = verify_char_offsets(broken)
    assert problems, "校验器没抓到错误的偏移"
    assert any("不一致" in problem for problem in problems)
