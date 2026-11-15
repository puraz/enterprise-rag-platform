"""解析器工厂：按格式选择实现。

## 为什么用注册表而不是 if-else

`docs/PROMPT.md` 要求"见到 `if provider == "xxx"` 一律改成能力匹配或策略注册"。
解析器是最典型的场景：新增格式只加一个类 + 一行注册。

## 为什么按 `DocumentFormat` 而不是文件扩展名路由

`chunks.format` 存的是 `DocumentFormat`（上传时已由扩展名 + 魔数确定）。
用枚举做键：拼错的格式名在类型检查阶段就暴露，而不是运行时。

## OCR 是可选能力

`docs/PROMPT.md` 要求"图片 OCR"，但本机没有 tesseract（需 sudo 安装）。
所以：

- `OcrEngine` 是接口；
- `TesseractOcr` 实现只在检测到 tesseract 可用时才构造；
- 未安装时 `NullOcr` 返回"未启用"，调用方据此降级
  （扫描页只保留"这是图片"的占位，不报错）。

**不静默失败**：`NullOcr.is_available()` 返回 False，
调用方与启动日志都能看到"OCR 未启用"。
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from rag_adapters.parsers.base import (
    BaseParser,
    ParsedDocument,
    ParseError,
    UnsupportedFormatError,
)
from rag_adapters.parsers.pdf import PdfParser
from rag_adapters.parsers.simple import (
    DocxParser,
    PptxParser,
    TextLikeParser,
    XlsxParser,
)
from rag_contracts.enums import DocumentFormat
from rag_core.logging import get_logger

log = get_logger(__name__)

# 格式 → 解析器工厂
# 新增格式：加一个解析器类 + 这里一行
_PARSER_BUILDERS: dict[DocumentFormat, Callable[[], BaseParser]] = {
    DocumentFormat.PDF: PdfParser,
    DocumentFormat.DOCX: DocxParser,
    DocumentFormat.PPTX: PptxParser,
    DocumentFormat.XLSX: XlsxParser,
    DocumentFormat.MARKDOWN: TextLikeParser,
    DocumentFormat.HTML: TextLikeParser,
    DocumentFormat.TEXT: TextLikeParser,
    DocumentFormat.CSV: TextLikeParser,
    # IMAGE 走 OCR 路径（见 ocr 模块），不走这里
}


def build_parser(format: DocumentFormat) -> BaseParser:
    """按格式构造解析器。

    Raises:
        UnsupportedFormatError: 格式没有对应解析器。
    """
    builder = _PARSER_BUILDERS.get(format)
    if builder is None:
        raise UnsupportedFormatError(format.value)
    return builder()


def supported_formats() -> tuple[DocumentFormat, ...]:
    """当前支持的格式（供启动日志与上传校验）。"""
    return tuple(sorted(_PARSER_BUILDERS, key=lambda item: item.value))


def parse_document(path: Path, *, format: DocumentFormat) -> ParsedDocument:
    """解析文档（便捷入口）。

    这是 worker 调用的主入口。
    """
    parser = build_parser(format)
    return parser.parse(path, format=format)


# ---------------------------------------------------------------- OCR（可选）


class OcrEngine:
    """OCR 引擎接口（可选能力）。

    实现约定：
    - `is_available()` 必须在**构造后即可调用**（不触发重依赖加载）；
    - `extract_text(image_path)` 失败时抛 `ParseError`；
    - **不记录图片内容**（日志里不能出现 OCR 出的文字）。
    """

    name: str = "none"

    def is_available(self) -> bool:
        """是否可用。False 表示调用方应降级。"""
        return False

    def extract_text(self, image_path: Path) -> str:
        raise ParseError("OCR 未启用", retryable=False)


class NullOcr(OcrEngine):
    """未启用 OCR 的占位实现。

    行为：明确报告"不可用"，让调用方降级而不是静默返回空字符串
    （静默返回空会让"扫描件解析失败"看起来像"文档是空的"）。
    """

    name = "none"

    def is_available(self) -> bool:
        return False


class TesseractOcr(OcrEngine):
    """Tesseract OCR（需要系统安装 tesseract 与 pytesseract）。

    **本机通常没有**（需 sudo pacman -S tesseract tesseract-data-chi_sim）。
    构造时只检查可执行文件是否存在，不做实际 OCR。
    """

    name = "tesseract"

    def __init__(self) -> None:
        self._binary_found = shutil.which("tesseract") is not None

    def is_available(self) -> bool:
        if not self._binary_found:
            return False
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return False
        return True

    def extract_text(self, image_path: Path) -> str:
        if not self.is_available():
            raise ParseError("OCR 不可用（缺少 tesseract 或 pytesseract）", retryable=False)

        import pytesseract
        from PIL import Image

        try:
            # lang="chi_sim+eng"：中文简体 + 英文
            # 语言包没装时 tesseract 会报错，这里捕获后降级到 eng
            try:
                return str(pytesseract.image_to_string(Image.open(image_path), lang="chi_sim+eng"))
            except pytesseract.TesseractError:
                return str(pytesseract.image_to_string(Image.open(image_path), lang="eng"))
        except Exception as exc:
            raise ParseError("OCR 失败", retryable=True) from exc


def build_ocr_engine(*, enabled: bool = True) -> OcrEngine:
    """构造 OCR 引擎。

    检测不到可用实现时返回 `NullOcr`（而非抛异常）：
    OCR 是可选能力，不该让整个 worker 起不来。

    但**必须在启动日志里告警**，否则"扫描件解析不出内容"
    会变成没人知道原因的谜。
    """
    if not enabled:
        return NullOcr()

    engine = TesseractOcr()
    if engine.is_available():
        log.info("ocr_engine_ready", engine=engine.name)
        return engine

    log.warning(
        "ocr_disabled",
        reason="tesseract 或 pytesseract 不可用",
        install_hint="sudo pacman -S tesseract tesseract-data-chi_sim && uv add pytesseract",
        impact="扫描件与图片将只保留占位符，无法提取文字",
    )
    return NullOcr()
