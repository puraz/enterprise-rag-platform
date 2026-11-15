"""PDF 解析器（PyMuPDF）。

## 为什么用 PyMuPDF 而不是 LangChain 的 PyPDFLoader

实测对比（2026-11，见 A8 的设计讨论）：

| 能力 | PyPDFLoader | PyMuPDF |
|---|---|---|
| 页码 | ✅ | ✅ |
| **bbox（版面坐标）** | ❌ | ✅ |
| **字符偏移** | ❌ | ✅（需自己算） |
| 字体/字号 | ❌ | ✅ |
| 扫描页检测 | ❌ | ✅（`page.get_images()`） |
| 维护状态 | sunset | 活跃 |

**bbox 是决定性因素**：A3-3 的 `chunks.bbox` 用于前端在 PDF 原图上
框选高亮。没有它，引用只能跳到某页而无法指出具体位置。

## 字符偏移的一致性难题（本实现最关键的地方）

`chunks.char_start/char_end` 指向 `ParsedDocument.text` 里的位置。
切分器按 `text` 切，chunk 的偏移必须能反查回原文。

**难点**：PyMuPDF 的 span 是按版面顺序返回的，跨页、跨栏时
顺序可能与"阅读顺序"不同。而且块之间需要插入分隔符（换行），
这些分隔符也必须计入偏移。

**策略**：
1. 按页、按 block、按 line、按 span 遍历（PyMuPDF 的 dict 结构）；
2. 每写一段文本到 `text` 累加器，就记录它在累加器里的区间；
3. **块之间插入 `\n`**（跨行）与 `\n\n`（跨段落），这些都是 `text` 的一部分，
   偏移按实际写入的长度累加。

不这样做（比如"先拼 text 再按内容搜索定位"）会在重复文本上出错：
文档里有两段相同的话时，`str.find` 永远返回第一处。

## 表格与图片

- **表格**：PyMuPDF 的 `page.find_tables()` 能识别表格边界。
  表格文本单独成一个 block，`block_type="table"`，切分时整块保留。
- **图片**：`page.get_images()` 给出图片位置。当前只记录"这里有图"
  （供 OCR 分流），不提取图片内容（OCR 是可选能力）。

## 扫描页检测

判据：**该页文本很少（< 50 字符）但有图像对象**。

为什么不用"文本为空"：有些页有页码或页眉，文本非空但主体是图片。
为什么用字符数阈值而不是精确判断：启发式足够——误判成扫描页会走
OCR（可选能力，未安装时降级为"文本很少"），漏判则丢失内容。
阈值 50 是经验值（一页扫描件通常能提取出 0-10 个字符的水印/页码）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_adapters.parsers.base import (
    BaseParser,
    PageInfo,
    ParsedDocument,
    ParseError,
    ParserCapabilities,
    TextBlock,
)
from rag_contracts.enums import DocumentFormat
from rag_core.logging import get_logger

log = get_logger(__name__)

# 判定"扫描页"的文本量阈值（字符数）
_SCANNED_PAGE_CHAR_THRESHOLD = 50
# 提取出的文档级元数据里允许保留的键（白名单，防止把奇怪字段写进库）
_SAFE_META_KEYS = ("title", "author", "subject", "keywords", "creator", "producer")


class PdfParser(BaseParser):
    """PDF 解析器（PyMuPDF）。

    **不用 `import fitz`**：`fitz` 是 PyMuPDF 的历史导入名，
    官方已标记为弃用（实测有 FutureWarning）。用 `import pymupdf`。
    """

    name = "pdf_pymupdf"

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            supported_formats=frozenset({DocumentFormat.PDF}),
            supports_page_numbers=True,
            supports_bbox=True,
            supports_tables=True,
            supports_images=True,
        )

    def parse(self, path: Path, *, format: DocumentFormat) -> ParsedDocument:
        if format is not DocumentFormat.PDF:
            from rag_adapters.parsers.base import UnsupportedFormatError

            raise UnsupportedFormatError(format.value)

        # 延迟 import：PyMuPDF 加载需要几百毫秒，
        # 不解析 PDF 的进程（比如只处理 Markdown）不该付这个成本
        import pymupdf

        try:
            # pymupdf 无类型存根（untyped call），但运行时行为稳定
            document: Any = pymupdf.open(path)  # type: ignore[no-untyped-call]
        except Exception as exc:
            # 不把异常原文透出去：可能含文件路径与内部结构
            raise ParseError("无法打开 PDF（文件损坏或加密）", retryable=False) from exc

        try:
            return self._extract(document)
        finally:
            document.close()

    def _extract(self, document: Any) -> ParsedDocument:
        """遍历页面提取文本与结构。"""
        # 累加器：文本与当前长度
        parts: list[str] = []
        cursor = 0

        blocks: list[TextBlock] = []
        pages: list[PageInfo] = []
        scanned_pages = 0

        doc = document  # 类型窄化用
        page_count = int(doc.page_count)

        for page_index in range(page_count):
            page = doc[page_index]
            # **页码从 1 开始**：用户看到的是"第 1 页"，
            # 不是"第 0 页"。存 0-based 会让所有引用偏一页。
            page_no = page_index + 1
            rect = page.rect
            page_text_start = cursor

            # ---- 表格（先提取，避免其文本被当正文重复计入）----
            # `find_tables` 可能抛异常（某些 PDF 结构异常），
            # 失败时降级为不识别表格，不中断整篇解析
            tables = self._find_tables(page)

            # ---- 正文 ----
            page_dict = page.get_text("dict")
            for raw_block in page_dict.get("blocks", []):
                # type 0 = 文本块，1 = 图片块
                if raw_block.get("type") != 0:
                    continue

                block_text, block_bbox, first_line = self._assemble_block(raw_block)
                if not block_text.strip():
                    continue

                # 块之间插入分隔：段落之间空行
                if parts:
                    separator = "\n\n"
                    parts.append(separator)
                    cursor += len(separator)

                start = cursor
                parts.append(block_text)
                cursor += len(block_text)
                end = cursor

                blocks.append(
                    TextBlock(
                        text=block_text,
                        block_type=self._classify_block(first_line),
                        page_no=page_no,
                        bbox=block_bbox,
                        char_start=start,
                        char_end=end,
                        # 标题层级：按字号推测（大于正文平均字号 1.3 倍视为标题）
                        level=self._guess_level(first_line),
                    )
                )

            # ---- 图片（记录位置，供 OCR 分流）----
            images = self._list_images(page)
            for image_index, image_bbox in enumerate(images):
                # 图片块只占位，文本用占位符（切分时通常被过滤或单独处理）
                placeholder = f"[图片 {page_no}-{image_index + 1}]"
                if parts:
                    parts.append("\n")
                    cursor += 1
                start = cursor
                parts.append(placeholder)
                cursor += len(placeholder)
                blocks.append(
                    TextBlock(
                        text=placeholder,
                        block_type="image_caption",
                        page_no=page_no,
                        bbox=image_bbox,
                        char_start=start,
                        char_end=cursor,
                    )
                )

            # ---- 扫描页判定 ----
            page_text_len = cursor - page_text_start
            is_scanned = page_text_len < _SCANNED_PAGE_CHAR_THRESHOLD and len(images) > 0
            if is_scanned:
                scanned_pages += 1

            pages.append(
                PageInfo(
                    page_no=page_no,
                    width=float(rect.width),
                    height=float(rect.height),
                    char_start=page_text_start,
                    char_end=cursor,
                    is_scanned=is_scanned,
                )
            )

            # 表格单独成块（追加在正文之后，保留其自身文本）
            for table_index, table_text in enumerate(tables):
                if parts:
                    parts.append("\n\n")
                    cursor += 2
                start = cursor
                parts.append(table_text)
                cursor += len(table_text)
                blocks.append(
                    TextBlock(
                        text=table_text,
                        block_type="table",
                        page_no=page_no,
                        char_start=start,
                        char_end=cursor,
                        metadata={"table_index": str(table_index)},
                    )
                )
                pages[-1] = PageInfo(
                    page_no=page_no,
                    width=float(rect.width),
                    height=float(rect.height),
                    char_start=pages[-1].char_start,
                    char_end=cursor,
                    is_scanned=is_scanned,
                )

        full_text = "".join(parts)

        if not full_text.strip():
            # 完全没有文本：可能是纯扫描件（全页图片）
            if scanned_pages == page_count and page_count > 0:
                raise ParseError("PDF 是扫描件，没有文本层（需要 OCR）", retryable=False)
            raise ParseError("PDF 没有可提取的文本", retryable=False)

        return ParsedDocument(
            text=full_text,
            blocks=blocks,
            pages=pages,
            format=DocumentFormat.PDF,
            parser_name=self.name,
            page_count=page_count,
            has_scanned_pages=scanned_pages > 0,
            metadata=self._safe_metadata(doc),
        )

    @staticmethod
    def _assemble_block(
        raw_block: dict[str, Any],
    ) -> tuple[str, tuple[float, float, float, float] | None, dict[str, Any]]:
        """把一个 PyMuPDF 文本块拼成字符串，并算整体 bbox。

        行之间用 `\\n` 连接（保留版面换行）。
        bbox 取所有行 bbox 的并集。
        """
        lines: list[str] = []
        bbox: list[float] | None = None
        first_line: dict[str, Any] = {}

        for line in raw_block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text = "".join(str(span.get("text", "")) for span in spans)
            if not line_text:
                continue

            if not first_line:
                first_line = spans[0]

            lines.append(line_text)

            line_bbox = line.get("bbox")
            if line_bbox:
                if bbox is None:
                    bbox = [float(v) for v in line_bbox]
                else:
                    bbox[0] = min(bbox[0], float(line_bbox[0]))
                    bbox[1] = min(bbox[1], float(line_bbox[1]))
                    bbox[2] = max(bbox[2], float(line_bbox[2]))
                    bbox[3] = max(bbox[3], float(line_bbox[3]))

        text = "\n".join(lines)
        return (
            text,
            (bbox[0], bbox[1], bbox[2], bbox[3]) if bbox else None,
            first_line,
        )

    @staticmethod
    def _classify_block(first_span: dict[str, Any]) -> str:
        """按字号/字体推测块类型。

        **只是启发式**：真正确认标题要靠 A9 的结构化切分器
        （它会结合编号模式、上下文）。这里只给一个初始提示。
        """
        size = float(first_span.get("size", 0) or 0)
        flags = int(first_span.get("flags", 0) or 0)
        # flags 的 bit 4 (16) 表示 bold
        if size >= 16 or (size >= 14 and flags & 16):
            return "title"
        return "text"

    @staticmethod
    def _guess_level(first_span: dict[str, Any]) -> int:
        """按字号推测标题层级（1 最大）。

        同样只是启发式：14pt+ → 1 级，12.5pt+ → 2 级，否则正文。
        """
        size = float(first_span.get("size", 0) or 0)
        if size >= 20:
            return 1
        if size >= 16:
            return 2
        if size >= 14:
            return 3
        return 0

    @staticmethod
    def _find_tables(page: Any) -> list[str]:
        """提取表格文本（失败时静默降级为空）。

        `find_tables` 在结构异常的 PDF 上会抛异常。
        表格识别失败不该让整篇解析失败——正文仍然有价值。
        """
        try:
            finder = page.find_tables()
        except Exception as exc:
            log.debug("pdf_table_detection_failed", error=type(exc).__name__)
            return []

        tables: list[str] = []
        for table in getattr(finder, "tables", []):
            try:
                rows = table.extract()
            except Exception:  # noqa: S112 - 单表失败不影响其它表
                continue
            # 用制表符分隔单元格：切分器能识别"这是表格"，
            # 且保留列对齐信息
            lines = [
                "\t".join(str(cell or "").strip() for cell in row)
                for row in rows
                if any(cell for cell in row)
            ]
            if lines:
                tables.append("\n".join(lines))
        return tables

    @staticmethod
    def _list_images(page: Any) -> list[tuple[float, float, float, float]]:
        """列出页面上图片的 bbox（用于 OCR 分流与占位）。"""
        try:
            infos = page.get_image_info()
        except Exception:
            return []
        boxes: list[tuple[float, float, float, float]] = []
        for info in infos:
            bbox = info.get("bbox")
            if bbox and len(bbox) == 4:
                boxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
        return boxes

    @staticmethod
    def _safe_metadata(document: Any) -> dict[str, str]:
        """提取文档级元数据（白名单键）。

        **白名单而不是黑名单**：PDF 元数据里可能有任意自定义字段，
        黑名单永远不完备。只取标准字段。
        """
        try:
            raw = document.metadata or {}
        except Exception:
            return {}
        return {key: str(raw[key]) for key in _SAFE_META_KEYS if raw.get(key)}
