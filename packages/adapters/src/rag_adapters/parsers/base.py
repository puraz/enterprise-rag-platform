"""文档解析器抽象。

## 解析层要产出什么

不是"一段文本"，而是**带位置信息的结构化文本**：

    ParsedDocument
      ├── text: 完整的纯文本（用于切分）
      ├── pages: 分页信息（页码 + 该页的文本区间 + 尺寸）
      └── blocks: 结构块（段落/标题/表格/图片说明）+ bbox + 字符偏移

**位置信息是引用高亮的前提**。`docs/PROMPT.md` 的验收标准是
"随机抽 20 个答案，引用位置能回到原文对应页"——如果解析器只给纯文本，
切分后的 chunk 就无法反查它在原文档的哪里。

A3-3 的 `chunks` 表因此有 `page_no` / `bbox` / `char_start` / `char_end`。
本层的职责就是把它们填上。

## 为什么 PDF 不用 LangChain 的 PyPDFLoader

实测对比（2026-11）：

| loader | page | bbox | char offset |
|---|---|---|---|
| `PyPDFLoader`（langchain-community） | ✅ | ❌ | ❌ |
| **PyMuPDF** | ✅ | ✅ | ✅ |

PyMuPDF 的 `page.get_text("dict")` 给出每个 span 的
`(x0, y0, x1, y1)`——前端据此在 PDF 原图上框选高亮。

另外 `langchain-community` 已官方宣布 sunset（实测有 DeprecationWarning），
`PyPDFLoader` 后续会被迁移到独立包，不适合作为长期依赖。

## 为什么其它格式可以用 LangChain

DOCX/PPTX/XLSX 不需要 bbox（它们是流式文档，没有"画布坐标"概念），
只需要段落结构与字符偏移。LangChain 的 loader 足够，且省去大量格式细节。

**但要注意**：`langchain-community` 的 loader 质量参差，
需要在本层做**输出规范化**（统一成 `ParsedDocument`），
这样换 loader 不影响下游。

## CPU 密集与进程隔离

PDF 解析是纯 CPU 工作（解析 PDF 结构、解码字体、提取文本）。
`docs/PROMPT.md` 要求"CPU 密集（解析/OCR/embedding）隔离到进程池或 worker"。

本层的 `parse` 是同步方法——它由 worker 直接调用（worker 本身就是独立进程）。
**如果在 API 进程里调用，必须放进程池**（见 `parse_in_process_pool`）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from rag_contracts.enums import DocumentFormat


@dataclass(frozen=True, slots=True)
class TextBlock:
    """一个结构块（段落/标题/表格等）。

    这些字段直接映射到 `chunks` 表的位置列。
    """

    text: str
    block_type: str = "text"
    """块类型：text / title / table / code / formula / image_caption / list。"""

    page_no: int | None = None
    """页码（从 1 开始）。非分页格式为 None。"""

    bbox: tuple[float, float, float, float] | None = None
    """版面坐标 `(x0, y0, x1, y1)`，单位是 PDF point。

    只有 PDF 这类"画布式"文档有。前端用它原图框选。
    """

    char_start: int = 0
    """在 `ParsedDocument.text` 里的起始字符偏移（含）。"""

    char_end: int = 0
    """结束偏移（不含）。`char_end > char_start` 恒成立。"""

    level: int = 0
    """标题层级（1-6），0 表示正文。"""

    metadata: dict[str, str] = field(default_factory=dict)
    """块级元数据（表格列数、代码语言等）。不含量级数据。"""


@dataclass(frozen=True, slots=True)
class PageInfo:
    """一页的信息。"""

    page_no: int
    """从 1 开始（与用户看到的页码一致，不是 0-based 索引）。"""

    width: float
    height: float
    char_start: int
    char_end: int
    is_scanned: bool = False
    """是否疑似扫描页（无文本层）。

    扫描页需要走 OCR（可选能力）。检测方法是"该页提取出的文本很少
    但有图像对象"。
    """


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """解析结果。"""

    text: str
    """完整纯文本。切分器以它为输入。"""

    blocks: list[TextBlock]
    """结构块列表，按文档顺序。"""

    pages: list[PageInfo]
    """分页信息。非分页格式为空列表。"""

    format: DocumentFormat
    parser_name: str
    """实际使用的解析器（进审计，便于追溯"这批 chunk 是哪个版本解析的"）。"""

    page_count: int = 0
    has_scanned_pages: bool = False
    """是否含扫描页（需要 OCR）。

    单独标记而不是只看 `pages[i].is_scanned`：
    调用方通常只关心"整篇是否需要 OCR"。
    """

    metadata: dict[str, str] = field(default_factory=dict)
    """文档级元数据（标题、作者、创建时间）。

    **不含量级或用户数据**：可能进日志与审计。
    """


class ParseError(Exception):
    """解析失败。

    不继承 AppError（与 Provider/Storage 层同样理由：
    本层不依赖 core 的业务异常，便于独立测试）。
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class UnsupportedFormatError(ParseError):
    """格式不支持。**不可重试**——换多少次都是不支持。"""

    def __init__(self, format_name: str) -> None:
        super().__init__(f"不支持的文档格式: {format_name}", retryable=False)
        self.format_name = format_name


@dataclass(frozen=True, slots=True)
class ParserCapabilities:
    """解析器能力声明。上层据此路由，不做格式名的 if-else。"""

    supported_formats: frozenset[DocumentFormat]
    supports_page_numbers: bool = False
    supports_bbox: bool = False
    supports_tables: bool = False
    supports_images: bool = False


class BaseParser(ABC):
    """文档解析器接口。

    实现约定：
    - `parse` 是**同步**方法（CPU 密集，由 worker 进程直接调用）；
    - 必须保证 `blocks` 按文档顺序、`char_start/char_end` 与 `text` 一致；
    - 失败抛 `ParseError` 或其子类，不吞异常；
    - **不记录文档内容**（日志、异常里都不能出现正文）。
    """

    name: str = "unknown"

    @property
    @abstractmethod
    def capabilities(self) -> ParserCapabilities:
        raise NotImplementedError

    @abstractmethod
    def parse(self, path: Path, *, format: DocumentFormat) -> ParsedDocument:
        """解析一个文件。

        Args:
            path: 本地文件路径（worker 已从存储下载到临时目录）。
            format: 文档格式（由上传时的扩展名 + 魔数确定）。

        Raises:
            UnsupportedFormatError: 格式不支持。
            ParseError: 解析失败（文件损坏、加密、无文本层等）。
        """
        raise NotImplementedError

    def supports(self, format: DocumentFormat) -> bool:
        """是否支持该格式（能力匹配，不做名字判断）。"""
        return format in self.capabilities.supported_formats


def verify_char_offsets(document: ParsedDocument) -> list[str]:
    """校验 blocks 的字符偏移与 text 一致。

    **这是解析器实现最容易出错的地方**：偏移算错会让引用高亮
    显示错误的文字（甚至跨段落）。而且这种错误在端到端测试里
    很难发现——答案看起来是对的，只是高亮位置偏了。

    返回问题列表（空表示通过）。由解析器的单测调用。
    """
    problems: list[str] = []
    previous_end = -1

    for index, block in enumerate(document.blocks):
        if block.char_end <= block.char_start:
            problems.append(
                f"block[{index}] 的 char_end({block.char_end}) <= char_start({block.char_start})"
            )
            continue
        if block.char_end > len(document.text):
            problems.append(
                f"block[{index}] 的 char_end({block.char_end}) 超出 text 长度({len(document.text)})"
            )
            continue
        # 检查 text 切片是否与 block.text 一致
        actual = document.text[block.char_start : block.char_end]
        if actual != block.text:
            problems.append(
                f"block[{index}] 的 text 与切片不一致：期望 {block.text[:30]!r}，"
                f"实际 {actual[:30]!r}"
            )
        # 顺序检查（允许重叠，比如父块包含子块）
        if block.char_start < previous_end - 1:
            # 重叠超过 1 字符时告警（负偏移说明顺序错乱）
            problems.append(
                f"block[{index}] 的 char_start({block.char_start}) 早于前一块的结束({previous_end})"
            )
        previous_end = max(previous_end, block.char_end)

    for page in document.pages:
        if page.char_start < 0 or page.char_end > len(document.text):
            problems.append(
                f"page[{page.page_no}] 的字符区间 [{page.char_start}, {page.char_end}] 越界"
            )
        if page.char_end <= page.char_start:
            problems.append(f"page[{page.page_no}] 的字符区间为空")

    return problems
