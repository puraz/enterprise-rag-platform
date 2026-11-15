"""文档解析层：按格式选择实现，统一输出 `ParsedDocument`。

## 支持的格式与位置信息能力

| 格式 | 解析器 | 页码 | bbox | 说明 |
|---|---|---|---|---|
| PDF | PyMuPDF | ✅ | ✅ | 唯一能给版面坐标（引用高亮） |
| PPTX | python-pptx | ✅ | — | 页号 = 幻灯片序号 |
| DOCX | docx2txt | — | — | 流式文档无真实页码 |
| XLSX | openpyxl | — | — | Sheet 名作上下文前缀 |
| MD/HTML/TXT/CSV | 自研 | — | — | 按段落/行切块 |

**PDF 不用 LangChain 的 PyPDFLoader**：实测它只给 page 不给 bbox，
而 bbox 是前端"点击引用在 PDF 原图上框选"的前提。
其它格式不需要 bbox，可用轻量库。

## OCR 是可选能力

`build_ocr_engine()` 检测不到 tesseract 时返回 `NullOcr`，
调用方据此降级（扫描页只保留占位符）。启动日志会明确告警。
"""

from rag_adapters.parsers.artifact import (
    BlockRecord,
    PageRecord,
    ParsedArtifact,
    ParsedArtifactError,
    ParsedManifest,
    dump_parsed,
    load_parsed,
    read_manifest,
)
from rag_adapters.parsers.base import (
    BaseParser,
    PageInfo,
    ParsedDocument,
    ParseError,
    ParserCapabilities,
    TextBlock,
    UnsupportedFormatError,
    verify_char_offsets,
)
from rag_adapters.parsers.factory import (
    NullOcr,
    OcrEngine,
    TesseractOcr,
    build_ocr_engine,
    build_parser,
    parse_document,
    supported_formats,
)
from rag_adapters.parsers.pdf import PdfParser
from rag_adapters.parsers.simple import DocxParser, PptxParser, TextLikeParser, XlsxParser

__all__ = [
    "BaseParser",
    "BlockRecord",
    "DocxParser",
    "NullOcr",
    "OcrEngine",
    "PageInfo",
    "PageRecord",
    "ParseError",
    "ParsedArtifact",
    "ParsedArtifactError",
    "ParsedDocument",
    "ParsedManifest",
    "ParserCapabilities",
    "PdfParser",
    "PptxParser",
    "TesseractOcr",
    "TextBlock",
    "TextLikeParser",
    "UnsupportedFormatError",
    "XlsxParser",
    "build_ocr_engine",
    "build_parser",
    "dump_parsed",
    "load_parsed",
    "parse_document",
    "read_manifest",
    "supported_formats",
    "verify_char_offsets",
]
