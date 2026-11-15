"""上传安全校验。

## 为什么这些校验必须在服务端

客户端声明的 `Content-Type` 和文件名**完全不可信**：
- 攻击者可以上传 `.exe` 而声明 `Content-Type: application/pdf`；
- 文件名可以含路径穿越（`../../etc/passwd`）、超长字符、控制字符；
- "图片"可能是 zip bomb、polyglot 文件，或伪装成图片的脚本。

`docs/PROMPT.md` 要求："魔数校验、大小上限、文件名净化、防 zip bomb
（比率/条目/总大小）、防路径穿越、病毒扫描扩展位"。

## 魔数（magic bytes）校验

读文件头几个字节，与**真实**格式签名比对，而不是相信 `Content-Type`。

为什么不只校验扩展名：扩展名是字符串，攻击者随意改。魔数是内容的
客观属性（PDF 一定以 `%PDF-` 开头）。

**局限（必须诚实说明）**：
- 魔数只能证明"文件开头像 PDF"，不能证明"整个文件是合法 PDF"；
- polyglot 文件（同时是合法 PDF 和合法 ZIP）能通过魔数校验；
- 真正的"文件是否能被安全解析"由 A8 的 parser 负责（解析失败进 FAILED）。

所以魔数是**第一道粗筛**，不能替代解析器的健壮性。

## 防 zip bomb

三个维度都要限制，缺一不可：
- **解压后总大小**（比率）：100KB 的 zip 解出 10GB → 拒绝；
- **条目数**：一个 zip 里有 100 万个文件 → 拒绝；
- **单条目大小**：单个文件解出 5GB → 拒绝。

只限总大小不够：攻击者可以传很多"恰好达标"的小炸弹。

## 文件名净化

不做净化的后果：`Content-Disposition` 头被注入换行 → 响应头注入攻击；
文件名里的路径分隔符 → 某些客户端会写到本地任意位置。
"""

from __future__ import annotations

import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

from rag_core.errors import BadRequestError, UnsupportedMediaTypeError

# 允许的文件扩展名 → 对应魔数规则
# 新增格式时：加进这里 + 在 A8 加对应 parser
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".csv",
        ".md",
        ".markdown",
        ".txt",
        ".html",
        ".htm",
        ".json",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }
)

# 魔数规则：(格式名, 可能的文件头前缀, 最小头长度)
# 一个文件可能匹配多个规则（如 docx 也是 zip）
_MAGIC_SIGNATURES: tuple[tuple[str, tuple[bytes, ...]], ...] = (
    # PDF: %PDF-
    ("pdf", (b"%PDF-",)),
    # ZIP 容器（docx/xlsx/pptx 都是 zip）
    ("zip", (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")),
    # 旧版 Office（doc/xls/ppt）的 OLE2 容器
    ("ole2", (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",)),
    # PNG
    ("png", (b"\x89PNG\r\n\x1a\n",)),
    # JPEG
    ("jpeg", (b"\xff\xd8\xff",)),
    # GIF
    ("gif", (b"GIF87a", b"GIF89a")),
    # WEBP: RIFF....WEBP
    ("webp", (b"RIFF",)),
    # BMP
    ("bmp", (b"BM",)),
    # TIFF
    ("tiff", (b"II*\x00", b"MM\x00*")),
)

# 扩展名 → 允许的魔数格式
_EXTENSION_TO_MAGIC: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"pdf"}),
    # docx/pptx/xlsx 都是 zip 容器
    ".docx": frozenset({"zip"}),
    ".pptx": frozenset({"zip"}),
    ".xlsx": frozenset({"zip"}),
    # 纯文本类没有魔数，只能靠扩展名 + 编码校验
    ".csv": frozenset({"text"}),
    ".md": frozenset({"text"}),
    ".markdown": frozenset({"text"}),
    ".txt": frozenset({"text"}),
    ".html": frozenset({"text"}),
    ".htm": frozenset({"text"}),
    ".json": frozenset({"text"}),
    ".png": frozenset({"png"}),
    ".jpg": frozenset({"jpeg"}),
    ".jpeg": frozenset({"jpeg"}),
    ".gif": frozenset({"gif"}),
    ".webp": frozenset({"webp"}),
    ".bmp": frozenset({"bmp"}),
    ".tif": frozenset({"tiff"}),
    ".tiff": frozenset({"tiff"}),
}

# 视为纯文本的扩展名（无魔数）
_TEXT_EXTENSIONS = frozenset({".csv", ".md", ".markdown", ".txt", ".html", ".htm", ".json"})

# 文件名最大长度（字节）。Linux 的 NAME_MAX 是 255
MAX_FILENAME_LENGTH = 200

# zip bomb 防护阈值
ZIP_MAX_TOTAL_UNCOMPRESSED = 2 * 1024**3  # 解压后总大小上限 2GB
ZIP_MAX_ENTRIES = 10_000  # 条目数上限
ZIP_MAX_RATIO = 100  # 压缩比上限（100:1）
ZIP_MAX_SINGLE_ENTRY = 512 * 1024**2  # 单条目上限 512MB

# 文件名里禁止的字符（控制字符会被单独处理）
_FORBIDDEN_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True, slots=True)
class FileInspection:
    """文件检查结果。"""

    detected_format: str
    """魔数识别出的格式（pdf/zip/png/...）。纯文本为 `text`。"""

    extension: str
    safe_filename: str
    """净化后的文件名（可安全用于 Content-Disposition）。"""


def sanitize_filename(raw: str) -> str:
    """净化文件名。

    处理顺序有讲究：

    1. **Unicode 规范化（NFC）**：不同编码方式的同名文件会被视为不同文件，
       导致去重失效；也防止用组合字符伪造扩展名；
    2. 取 basename：去掉任何路径成分（`../`、`C:\\`、`/etc/`）；
    3. 去控制字符与保留字符；
    4. 折叠连续的点（`....` → `.`）：防止 `file....pdf` 这类绕过扩展名检查；
    5. 长度截断但**保留扩展名**（截断掉扩展名会导致格式识别失败）；
    6. 空文件名回退到默认值。

    Raises:
        BadRequestError: 文件名净化后为空或只剩危险字符。
    """
    if not raw or not raw.strip():
        raise BadRequestError("文件名不能为空")

    # 1. Unicode 规范化
    name = unicodedata.normalize("NFC", raw.strip())

    # 2. 取纯文件名：同时处理 / 与 \ 两种分隔符
    #    PurePosixPath 只处理 /；Windows 路径要手动替换
    name = name.replace("\\", "/")
    name = Path(name).name

    # 3. 去控制字符与保留字符
    name = _FORBIDDEN_FILENAME_CHARS.sub("_", name)

    # 4. 折叠连续的点（但保留扩展名前的那个点）
    name = re.sub(r"\.{2,}", ".", name)

    # 5. 去掉首尾的点与空格（Windows 不允许，且 `.hidden` 无意义）
    name = name.strip(". ")

    if not name:
        raise BadRequestError("文件名净化后为空")

    # 6. 长度截断，保留扩展名
    if len(name) > MAX_FILENAME_LENGTH:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 16:
            keep = MAX_FILENAME_LENGTH - len(ext) - 1
            name = f"{stem[:keep]}.{ext}" if keep > 0 else f"{ext}"
        else:
            name = name[:MAX_FILENAME_LENGTH]

    return name


def extract_extension(filename: str) -> str:
    """提取小写扩展名（含点）。无扩展名返回空串。"""
    dot = filename.rfind(".")
    if dot <= 0 or dot == len(filename) - 1:
        return ""
    return filename[dot:].lower()


def validate_extension(filename: str) -> str:
    """校验扩展名在白名单内。

    Raises:
        UnsupportedMediaTypeError: 扩展名不在白名单。
    """
    ext = extract_extension(filename)
    if not ext or ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedMediaTypeError(
            f"不支持的文件类型: {ext or '无扩展名'}",
            details={"allowed_count": len(ALLOWED_EXTENSIONS)},
        )
    return ext


def detect_format(head: bytes) -> str:
    """按文件头识别真实格式。

    Args:
        head: 文件前若干字节（至少 16 字节；纯文本检测需要更多）。

    Returns:
        格式名（`pdf`/`zip`/`png`/`text`/...）；无法识别返回 `unknown`。
    """
    if not head:
        return "unknown"

    for format_name, signatures in _MAGIC_SIGNATURES:
        for signature in signatures:
            if head.startswith(signature):
                # WEBP 需要额外确认 RIFF 后面第 8-12 字节是 "WEBP"
                if format_name == "webp":
                    if len(head) >= 12 and head[8:12] == b"WEBP":
                        return "webp"
                    continue
                return format_name

    # 纯文本检测：前 4KB 全是可打印字符或常见空白
    if _looks_like_text(head):
        return "text"

    return "unknown"


def _looks_like_text(data: bytes) -> bool:
    """判断字节序列是否像纯文本。

    用 UTF-8 解码试探：能解码且不含大量控制字符就认为是文本。
    不做严格校验（A8 的 parser 会做真正的格式验证）。
    """
    sample = data[:4096]
    if b"\x00" in sample:  # 空字节几乎不可能出现在文本里
        return False
    try:
        decoded = sample.decode("utf-8")
    except UnicodeDecodeError:
        # 可能是截断在多字节字符中间，尝试去掉末尾 3 字节再试
        try:
            decoded = sample[:-3].decode("utf-8")
        except UnicodeDecodeError:
            return False
    # 控制字符比例过高则不像文本（允许 \t \n \r）
    control = sum(1 for ch in decoded if ord(ch) < 32 and ch not in "\t\n\r")
    return control / max(len(decoded), 1) < 0.05


def inspect_file(*, filename: str, head: bytes | None = None) -> FileInspection:
    """综合检查：文件名净化 + 扩展名校验 +（可选）魔数校验。

    **顺序不能反**：先净化文件名（避免用未净化的名字做扩展名判断，
    比如 `a.pdf/../x.exe`），再校验扩展名，最后校验魔数。

    Args:
        head: 文件头字节。**传 None 表示"还没有内容可校验"** ——
            `initiate` 阶段只能校验文件名与扩展名，
            魔数校验要等分片上传后才能做（见 `complete` 流程）。

            传空 bytes（`b""`）与 None 的区别：空 bytes 表示"内容是空的"，
            会被判为 unknown 并拒绝；None 表示"还没读内容"，跳过魔数检查。

    Raises:
        BadRequestError: 文件名非法。
        UnsupportedMediaTypeError: 扩展名不支持或魔数不匹配。
    """
    safe_name = sanitize_filename(filename)
    ext = validate_extension(safe_name)

    if head is None:
        # initiate 阶段：没有内容，只能校验名字
        return FileInspection(detected_format="pending", extension=ext, safe_filename=safe_name)

    detected = detect_format(head)

    allowed_formats = _EXTENSION_TO_MAGIC.get(ext, frozenset())
    if ext in _TEXT_EXTENSIONS:
        # 纯文本类：允许 text 或 unknown（短文件可能全是不常见字符）
        if detected not in ("text", "unknown"):
            raise UnsupportedMediaTypeError(
                f"文件内容与扩展名不符：{ext} 的文件实际是 {detected}",
                details={"extension": ext, "detected": detected},
            )
    elif detected not in allowed_formats:
        raise UnsupportedMediaTypeError(
            f"文件内容与扩展名不符：{ext} 的文件实际是 {detected}",
            details={"extension": ext, "detected": detected},
        )

    return FileInspection(detected_format=detected, extension=ext, safe_filename=safe_name)


def check_zip_bomb(path: Path) -> None:
    """检查 zip 容器（docx/xlsx/pptx）是否可能是 zip bomb。

    **必须在解压前检查**：解压炸弹时已经晚了（内存/磁盘已爆）。

    用 `ZipFile.infolist()` 读中央目录（不实际解压），
    从 `file_size`（声明解压后大小）判断。注意这个字段可以被伪造，
    所以 A8 的实际解压还要有"解压时监控实际字节数"的兜底。

    Raises:
        BadRequestError: 疑似 zip bomb。
    """
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise BadRequestError("压缩容器损坏") from exc
    except Exception as exc:
        raise BadRequestError(f"无法读取压缩容器: {type(exc).__name__}") from exc

    if len(entries) > ZIP_MAX_ENTRIES:
        raise BadRequestError(
            f"压缩包条目过多（{len(entries)} > {ZIP_MAX_ENTRIES}）",
            details={"entries": len(entries)},
        )

    total_uncompressed = sum(entry.file_size for entry in entries)
    if total_uncompressed > ZIP_MAX_TOTAL_UNCOMPRESSED:
        raise BadRequestError(
            f"压缩包解压后过大（{total_uncompressed // 1024 // 1024}MB）",
            details={"uncompressed_bytes": total_uncompressed},
        )

    compressed = sum(max(entry.compress_size, 1) for entry in entries)
    if total_uncompressed / compressed > ZIP_MAX_RATIO:
        raise BadRequestError(
            f"压缩比异常（{total_uncompressed // compressed}:1），疑似压缩炸弹",
            details={"ratio": total_uncompressed // compressed},
        )

    for entry in entries:
        if entry.file_size > ZIP_MAX_SINGLE_ENTRY:
            raise BadRequestError(
                f"压缩包内单个文件过大（{entry.filename}）",
                details={"filename": entry.filename, "size": entry.file_size},
            )
        # 条目的路径穿越（zip slip）：解压时会写到压缩包外的位置
        if ".." in entry.filename.split("/") or entry.filename.startswith("/"):
            raise BadRequestError(
                "压缩包内含非法路径（zip slip）",
                details={"filename": entry.filename},
            )
