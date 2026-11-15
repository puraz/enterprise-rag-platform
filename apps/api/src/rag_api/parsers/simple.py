"""轻量格式解析器：DOCX / PPTX / XLSX / Markdown / HTML / TXT。

## 为什么用 LangChain 的 loader

这些格式**不需要 bbox**（它们是流式文档，没有"画布坐标"概念）：
- DOCX/PPTX 是段落流，页码取决于渲染时的排版（不同软件不同）；
- Markdown/HTML/TXT 本身就是文本，偏移即位置。

所以 LangChain 的 loader 足够，省去大量格式细节（OOXML 命名空间、
样式继承、表格合并单元格…）。

## 但必须做"输出规范化"

不同 loader 的返回形状不一致：
- `Docx2txtLoader` 返回单个 Document（整篇文本）；
- `UnstructuredWordDocumentLoader` 返回多个（按元素分块）；
- `TextLoader` 返回单个。

如果下游直接消费 loader 的输出，换 loader 就要改切分器。
所以本模块统一转成 `ParsedDocument`：

- 把 loader 的 `page_content` 拼成 `text`；
- 每段（按空行或 loader 的元素）算 `char_start/char_end`；
- **页码留空**（这些格式没有真实页码）；
- `bbox` 留空。

## 为什么不用 `Unstructured` 系列

`unstructured` 是重依赖（含 detectron2/onnx 等可选组件，安装后
1GB+），且它对 PDF 也会接管（我们不希望——PDF 走 PyMuPDF 以保留 bbox）。

`langchain_community` 里的轻量 loader（`Docx2txtLoader` /
`UnstructuredPowerPointLoader` 的轻量替代 / `CSVLoader`）足够。
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_api.parsers.base import (
    BaseParser,
    ParsedDocument,
    ParseError,
    ParserCapabilities,
    TextBlock,
    UnsupportedFormatError,
)
from rag_contracts.enums import DocumentFormat
from rag_core.logging import get_logger

log = get_logger(__name__)

# 段落分隔：连续空行（两个以上换行）
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")

# Markdown 标题：`#` 开头
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

# HTML 标签（用于粗略剥离，保留结构与文本）
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_BLOCK = re.compile(r"</(p|div|h[1-6]|li|tr|br)\s*>", re.IGNORECASE)


class TextLikeParser(BaseParser):
    """纯文本类解析器（Markdown / HTML / TXT / CSV）。

    这些格式不需要外部库：读文件 + 规范化换行 + 按段落切块。

    CSV 的"段落"是行（每行一个记录），但为了避免一行一个 block
    造成 block 数量爆炸，按 50 行聚合。
    """

    name = "text_like"

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            supported_formats=frozenset(
                {
                    DocumentFormat.MARKDOWN,
                    DocumentFormat.HTML,
                    DocumentFormat.TEXT,
                    DocumentFormat.CSV,
                }
            ),
            # 纯文本没有真实页码
            supports_page_numbers=False,
            supports_bbox=False,
            supports_tables=DocumentFormat.CSV in {DocumentFormat.CSV},
            supports_images=False,
        )

    def parse(self, path: Path, *, format: DocumentFormat) -> ParsedDocument:
        if not self.supports(format):
            raise UnsupportedFormatError(format.value)

        raw = self._read_text(path)

        # 统一换行符：\r\n / \r → \n
        # 不统一会让偏移计算在 Windows 换行上错乱
        text = raw.replace("\r\n", "\n").replace("\r", "\n")

        if format is DocumentFormat.HTML:
            text = self._html_to_text(text)
        elif format is DocumentFormat.MARKDOWN:
            # Markdown 保留原文（切分器要读标题标记）
            pass
        elif format is DocumentFormat.CSV:
            # CSV 保留原文（切分器按行切，并识别表头）
            pass

        if not text.strip():
            raise ParseError("文件没有可提取的文本", retryable=False)

        blocks = (
            self._blocks_from_csv(text)
            if format is DocumentFormat.CSV
            else self._blocks_from_paragraphs(text, format)
        )

        return ParsedDocument(
            text=text,
            blocks=blocks,
            pages=[],  # 纯文本没有页码
            format=format,
            parser_name=self.name,
            page_count=0,
            has_scanned_pages=False,
        )

    @staticmethod
    def _read_text(path: Path) -> str:
        """读文本文件，处理编码。

        **顺序尝试 UTF-8 / GBK**：中文文档常见 GBK 编码，
        只试 UTF-8 会抛 UnicodeDecodeError，而错误信息看不出是编码问题。

        用 `errors="replace"` 作为最后兜底：宁可有几个乱码字符，
        也不要整篇解析失败（用户至少能看到大部分内容）。
        """
        for encoding in ("utf-8", "gbk", "gb18030"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        # 全部失败：用 replace 兜底
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _html_to_text(html: str) -> str:
        """HTML 转纯文本（剥离标签，保留段落结构）。

        不用 BeautifulSoup：多一个依赖且对 RAG 场景过度。
        正则足够——我们只需要"文字 + 段落边界"，不需要 DOM。

        **注意 XSS 不是这里的问题**：解析结果是给模型看的纯文本，
        不会渲染到浏览器。前端的引用展示会做转义。
        """
        # 块级标签转换成换行（保留段落边界）
        text = _HTML_BLOCK.sub("\n\n", html)
        # 剥离剩余标签
        text = _HTML_TAG.sub("", text)
        # 解码常见实体（顺序：先解码 &amp; 避免二次解码）
        entities = {
            "&nbsp;": " ",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&amp;": "&",
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)
        # 折叠过多空行
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _blocks_from_paragraphs(self, text: str, format: DocumentFormat) -> list[TextBlock]:
        """按段落切块，记录每块的字符偏移。

        用 `finditer` 而不是 `split`：`split` 会丢失分隔符位置，
        导致无法还原偏移。
        """
        blocks: list[TextBlock] = []
        cursor = 0

        for match in _PARAGRAPH_SPLIT.finditer(text):
            segment = text[cursor : match.start()]
            if segment.strip():
                blocks.append(self._make_block(segment, cursor, format))
            # 跳过段落间的空行（它们属于 text，但不属于任何 block）
            cursor = match.end()

        # 最后一段
        tail = text[cursor:]
        if tail.strip():
            blocks.append(self._make_block(tail, cursor, format))

        return blocks

    def _make_block(self, segment: str, start: int, format: DocumentFormat) -> TextBlock:
        """构造一个 block（识别 Markdown 标题）。"""
        block_type = "text"
        level = 0

        if format is DocumentFormat.MARKDOWN:
            first_line = segment.split("\n", 1)[0]
            heading = _MD_HEADING.match(first_line)
            if heading:
                block_type = "title"
                level = len(heading.group(1))

        return TextBlock(
            text=segment,
            block_type=block_type,
            page_no=None,  # 无页码
            bbox=None,  # 无版面坐标
            char_start=start,
            char_end=start + len(segment),
            level=level,
        )

    def _blocks_from_csv(self, text: str) -> list[TextBlock]:
        """CSV 按行聚合（每 50 行一个 block）。

        一行一个 block 会让 block 数量爆炸（万行 CSV → 万个 block），
        而切分器真正需要的是"这几十行是一个表格片段"。
        """
        lines = text.split("\n")
        blocks: list[TextBlock] = []
        group_size = 50
        cursor = 0

        for index in range(0, len(lines), group_size):
            group = "\n".join(lines[index : index + group_size])
            if not group.strip():
                cursor += len(group) + 1
                continue
            blocks.append(
                TextBlock(
                    text=group,
                    block_type="table",
                    page_no=None,
                    bbox=None,
                    char_start=cursor,
                    char_end=cursor + len(group),
                    metadata={"is_header": str(index == 0)},
                )
            )
            cursor += len(group) + 1  # +1 是换行符
        return blocks


class DocxParser(BaseParser):
    """DOCX 解析器（用 LangChain 的 Docx2txtLoader）。

    **为什么不自己解 OOXML**：docx 是 zip 包，内含 `word/document.xml`。
    手写解析要处理命名空间、样式继承、表格嵌套、脚注——几千行代码
    且容易漏。`docx2txt` 是成熟实现，几十行包装就够。
    """

    name = "docx_langchain"

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            supported_formats=frozenset({DocumentFormat.DOCX}),
            supports_page_numbers=False,
            supports_bbox=False,
            supports_tables=True,
            supports_images=False,
        )

    def parse(self, path: Path, *, format: DocumentFormat) -> ParsedDocument:
        if format is not DocumentFormat.DOCX:
            raise UnsupportedFormatError(format.value)

        # 用 docx2txt（轻量，只依赖 python-docx 的替代实现）
        try:
            import docx2txt
        except ImportError as exc:  # pragma: no cover
            raise ParseError("缺少 docx2txt 依赖", retryable=False) from exc

        try:
            text = docx2txt.process(str(path))
        except Exception as exc:
            raise ParseError("无法解析 DOCX", retryable=False) from exc

        # docx2txt 返回的文本里段落之间是 \n，没有空行
        # 统一成 \n\n 分隔，便于按段落切块
        normalized = re.sub(r"\n{2,}", "\n\n", text.replace("\r\n", "\n"))
        if not normalized.strip():
            raise ParseError("DOCX 没有可提取的文本", retryable=False)

        parser = TextLikeParser()
        blocks = parser._blocks_from_paragraphs(normalized, DocumentFormat.DOCX)

        return ParsedDocument(
            text=normalized,
            blocks=blocks,
            pages=[],
            format=DocumentFormat.DOCX,
            parser_name=self.name,
            page_count=0,
            has_scanned_pages=False,
        )


class UnsupportedParser(BaseParser):
    """占位实现：对不支持的格式明确报错。

    为什么需要它：工厂要能对每个 DocumentFormat 返回一个 parser
    （否则调用方要处理 None）。用这个占位让"不支持"变成运行时
    的明确异常，而不是 AttributeError。
    """

    name = "unsupported"

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(supported_formats=frozenset())

    def parse(self, path: Path, *, format: DocumentFormat) -> ParsedDocument:
        raise UnsupportedFormatError(format.value)


class PptxParser(BaseParser):
    """PPTX 解析器（python-pptx）。

    幻灯片天然有"页"概念（slide），所以能给出**真实页号**——
    这与 PDF 的 page 语义一致，引用可以直接跳到第 N 张幻灯片。

    但注意：这里的 `page_no` 是 slide 序号，不是渲染后的物理页
    （一张幻灯片可能打印成多页）。对引用跳转够用。
    """

    name = "pptx_python_pptx"

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            supported_formats=frozenset({DocumentFormat.PPTX}),
            supports_page_numbers=True,
            supports_bbox=False,
            supports_tables=True,
            supports_images=False,
        )

    def parse(self, path: Path, *, format: DocumentFormat) -> ParsedDocument:
        if format is not DocumentFormat.PPTX:
            raise UnsupportedFormatError(format.value)

        from pptx import Presentation

        try:
            presentation = Presentation(str(path))
        except Exception as exc:
            raise ParseError("无法解析 PPTX", retryable=False) from exc

        parts: list[str] = []
        blocks: list[TextBlock] = []
        cursor = 0
        slide_count = 0

        for slide_index, slide in enumerate(presentation.slides):
            slide_count = slide_index + 1
            slide_no = slide_index + 1
            slide_chunks: list[str] = []

            for shape in slide.shapes:
                # 文本框
                if shape.has_text_frame:
                    text = shape.text_frame.text.strip()
                    if text:
                        slide_chunks.append(text)
                # 表格
                elif shape.has_table:
                    rows = []
                    for row in shape.table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        if any(cells):
                            rows.append("\t".join(cells))
                    if rows:
                        slide_chunks.append("\n".join(rows))

            if not slide_chunks:
                continue

            slide_text = "\n\n".join(slide_chunks)

            if parts:
                parts.append("\n\n")
                cursor += 2
            start = cursor
            parts.append(slide_text)
            cursor += len(slide_text)

            blocks.append(
                TextBlock(
                    text=slide_text,
                    block_type="text",
                    page_no=slide_no,
                    bbox=None,
                    char_start=start,
                    char_end=cursor,
                    metadata={"slide": str(slide_no)},
                )
            )

        full_text = "".join(parts)
        if not full_text.strip():
            raise ParseError("PPTX 没有可提取的文本", retryable=False)

        return ParsedDocument(
            text=full_text,
            blocks=blocks,
            pages=[],  # 幻灯片不算"分页"，页码信息在 block 上
            format=DocumentFormat.PPTX,
            parser_name=self.name,
            page_count=slide_count,
            has_scanned_pages=False,
        )


class XlsxParser(BaseParser):
    """XLSX 解析器（openpyxl）。

    **Sheet 名作为上下文前缀**：单独的表格行没有语义
    （"研发部 | 张三 | 5000"），加上 sheet 名（"工资表"）才有。
    所以每个 block 的文本以 `[Sheet: 名称]` 开头。
    """

    name = "xlsx_openpyxl"

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            supported_formats=frozenset({DocumentFormat.XLSX}),
            supports_page_numbers=False,
            supports_bbox=False,
            supports_tables=True,
            supports_images=False,
        )

    def parse(self, path: Path, *, format: DocumentFormat) -> ParsedDocument:
        if format is not DocumentFormat.XLSX:
            raise UnsupportedFormatError(format.value)

        from openpyxl import load_workbook

        try:
            # read_only=True：流式读，大表格不会整本进内存
            # data_only=True：读计算后的值而不是公式
            workbook = load_workbook(str(path), read_only=True, data_only=True)
        except Exception as exc:
            raise ParseError("无法解析 XLSX", retryable=False) from exc

        parts: list[str] = []
        blocks: list[TextBlock] = []
        cursor = 0
        sheet_count = 0

        try:
            for sheet in workbook.worksheets:
                sheet_count += 1
                rows_text: list[str] = []

                # 每个 sheet 最多取 5000 行（防止超大表撑爆内存与上下文）
                for row_index, row in enumerate(sheet.iter_rows(values_only=True)):
                    if row_index >= 5000:
                        rows_text.append("...（表格过大，已截断）")
                        break
                    cells = [str(value).strip() if value is not None else "" for value in row]
                    if any(cells):
                        rows_text.append("\t".join(cells))

                if not rows_text:
                    continue

                # Sheet 名作为上下文前缀
                sheet_text = f"[Sheet: {sheet.title}]\n" + "\n".join(rows_text)

                if parts:
                    parts.append("\n\n")
                    cursor += 2
                start = cursor
                parts.append(sheet_text)
                cursor += len(sheet_text)

                blocks.append(
                    TextBlock(
                        text=sheet_text,
                        block_type="table",
                        page_no=None,
                        bbox=None,
                        char_start=start,
                        char_end=cursor,
                        metadata={"sheet": str(sheet.title)},
                    )
                )
        finally:
            workbook.close()

        full_text = "".join(parts)
        if not full_text.strip():
            raise ParseError("XLSX 没有可提取的内容", retryable=False)

        return ParsedDocument(
            text=full_text,
            blocks=blocks,
            pages=[],
            format=DocumentFormat.XLSX,
            parser_name=self.name,
            page_count=sheet_count,
            has_scanned_pages=False,
        )
