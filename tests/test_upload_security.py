"""上传安全校验测试。

这些测试守的是**安全边界**，不是功能覆盖率：

- 魔数校验：声明 pdf 实际传 exe（绕过文件类型限制）
- zip bomb：小压缩包解出 10GB（DoS）
- zip slip：压缩包内路径穿越（写到系统任意位置）
- 文件名净化：路径穿越、控制字符注入、超长名
- 扩展名白名单：拒绝可执行文件
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from rag_api.services.upload_security import (
    ALLOWED_EXTENSIONS,
    ZIP_MAX_ENTRIES,
    check_zip_bomb,
    detect_format,
    extract_extension,
    inspect_file,
    sanitize_filename,
    validate_extension,
)
from rag_core.errors import BadRequestError, UnsupportedMediaTypeError

# ==================================================== 文件名净化


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("  报告.pdf  ", "报告.pdf"),
        # 控制字符替换成 _ 而不是删除：保持字符位置，
        # 避免 "file\x00.pdf" 变成 "file.pdf" 后与真实文件重名
        ("file\x00name.pdf", "file_name.pdf"),
        ("file\nname.pdf", "file_name.pdf"),
        ("file:name?.pdf", "file_name_.pdf"),  # Windows 保留字符
    ],
)
def test_sanitize_filename_normal_cases(raw: str, expected: str) -> None:
    """常规净化：去空白、控制字符、保留字符。"""
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize(
    "evil",
    [
        "../../etc/passwd",
        "../../../root/.ssh/id_rsa",
        "..\\..\\windows\\system32\\config",
        "/etc/passwd",
        "C:\\Windows\\System32\\cmd.exe",
    ],
)
def test_sanitize_filename_strips_path_components(evil: str) -> None:
    """**路径穿越防护**：文件名里的路径成分必须被剥离。

    不做净化的后果：服务端把文件名拼到存储路径上时会写到系统任意位置。
    这类漏洞在"文件上传"功能里真实出现过很多次。
    """
    result = sanitize_filename(evil)
    assert "/" not in result
    assert "\\" not in result
    assert ".." not in result
    assert not result.startswith("/")


def test_sanitize_filename_collapses_dots() -> None:
    """连续的点被折叠，防止 `file....pdf` 绕过扩展名检查。"""
    assert sanitize_filename("file....pdf") == "file.pdf"


def test_sanitize_filename_truncates_but_keeps_extension() -> None:
    """超长文件名截断但**保留扩展名**。

    截断掉扩展名会导致格式识别失败，文件变成"不支持的类型"。
    """
    long_name = "a" * 300 + ".pdf"
    result = sanitize_filename(long_name)
    assert len(result) <= 200
    assert result.endswith(".pdf")


def test_sanitize_filename_rejects_empty() -> None:
    """空文件名必须拒绝。"""
    with pytest.raises(BadRequestError):
        sanitize_filename("")
    with pytest.raises(BadRequestError):
        sanitize_filename("   ")


# ==================================================== 扩展名白名单


def test_extension_whitelist_blocks_executables() -> None:
    """可执行文件必须被拒绝。

    允许 .exe/.sh/.bat 上传，配合后续的"下载"功能就是
    一个文件托管平台——攻击者可以用它传播恶意软件。
    """
    for ext in (".exe", ".sh", ".bat", ".ps1", ".dll", ".so", ".js"):
        with pytest.raises(UnsupportedMediaTypeError):
            validate_extension(f"malware{ext}")


def test_extension_whitelist_allows_documents() -> None:
    """常见文档格式允许（RAG 的输入类型）。"""
    for ext in (".pdf", ".docx", ".xlsx", ".pptx", ".md", ".txt", ".csv"):
        assert validate_extension(f"doc{ext}") == ext


def test_extension_case_insensitive() -> None:
    """扩展名大小写不敏感（Windows 用户常见 .PDF）。"""
    assert extract_extension("REPORT.PDF") == ".pdf"
    assert validate_extension("REPORT.PDF") == ".pdf"


def test_no_extension_rejected() -> None:
    """无扩展名拒绝（无法判断格式）。"""
    with pytest.raises(UnsupportedMediaTypeError):
        validate_extension("noextension")


# ==================================================== 魔数检测


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"%PDF-1.7\n", "pdf"),
        (b"PK\x03\x04rest", "zip"),  # docx/xlsx/pptx 都是 zip
        (b"\x89PNG\r\n\x1a\n", "png"),
        (b"\xff\xd8\xff\xe0", "jpeg"),
        (b"GIF89a", "gif"),
        (b"RIFF\x00\x00\x00\x00WEBP", "webp"),
        (b"BM\x00\x00", "bmp"),
        (b"II*\x00", "tiff"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole2"),
    ],
)
def test_detect_format_by_magic(head: bytes, expected: str) -> None:
    """按文件头识别真实格式。"""
    assert detect_format(head) == expected


def test_detect_text_format() -> None:
    """纯文本能识别（用于 csv/md/txt）。"""
    assert detect_format("这是中文文本\n第二行".encode()) == "text"
    assert detect_format(b"hello world\n") == "text"


def test_detect_unknown_binary() -> None:
    """未知二进制格式返回 unknown（会与扩展名比对后拒绝）。"""
    assert detect_format(b"\x01\x02\x03\x04\x05\x06\x07\x08") == "unknown"


# ==================================================== 综合校验


def test_inspect_file_accepts_matching_pdf() -> None:
    """扩展名与魔数一致时通过。"""
    result = inspect_file(filename="doc.pdf", head=b"%PDF-1.7\ncontent")
    assert result.detected_format == "pdf"
    assert result.safe_filename == "doc.pdf"


def test_inspect_file_rejects_mismatched_magic() -> None:
    """**核心安全测试**：声明 pdf 实际传 exe。

    这是绕过文件类型限制的经典手法：`Content-Type` 和扩展名都可伪造，
    只有内容本身的魔数不会。
    """
    exe_head = b"MZ\x90\x00\x03\x00\x00\x00"  # PE 文件头

    with pytest.raises(UnsupportedMediaTypeError, match="文件内容与扩展名不符"):
        inspect_file(filename="innocent.pdf", head=exe_head)


def test_inspect_file_pending_mode() -> None:
    """`head=None` 表示"还没有内容"，只校验文件名与扩展名。

    `initiate` 阶段还没有上传任何字节，此时只能校验名字；
    魔数校验留到 `complete`。
    """
    result = inspect_file(filename="doc.pdf", head=None)
    assert result.detected_format == "pending"
    assert result.extension == ".pdf"


def test_inspect_file_empty_content_rejected() -> None:
    """空内容（head=b""）与"还没内容"（head=None）是两回事。

    空文件对 pdf 来说是非法的（没有魔数），必须拒绝。
    """
    with pytest.raises(UnsupportedMediaTypeError):
        inspect_file(filename="empty.pdf", head=b"")


def test_inspect_file_text_extension_allows_text() -> None:
    """csv/md/txt 允许纯文本内容（这些格式没有魔数）。"""
    result = inspect_file(filename="data.csv", head=b"a,b,c\n1,2,3")
    assert result.detected_format == "text"


# ==================================================== zip bomb


def _make_zip(files: dict[str, bytes], *, compresslevel: int = 9) -> Path:
    """构造测试用 zip。"""
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "test.zip"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return tmp


def test_zip_bomb_by_entry_count(tmp_path: Path) -> None:
    """**条目数攻击**：一个 zip 里塞几百万个小文件。

    解压时会创建海量 inode，打满文件系统。
    """
    path = _make_zip({f"f{i}.txt": b"x" for i in range(ZIP_MAX_ENTRIES + 1)})
    with pytest.raises(BadRequestError, match="条目过多"):
        check_zip_bomb(path)


def test_zip_bomb_by_compression_ratio(tmp_path: Path) -> None:
    """**压缩比攻击**：100KB 的 zip 解出 10GB。

    高重复度的内容压缩比可以到 1000:1，解压时会打爆内存/磁盘。
    """
    # 10MB 的零字节，压缩后约 10KB，比率约 1000:1
    path = _make_zip({"bomb.txt": b"\x00" * (10 * 1024 * 1024)})
    with pytest.raises(BadRequestError, match="压缩比异常"):
        check_zip_bomb(path)


def test_zip_normal_file_passes(tmp_path: Path) -> None:
    """正常文档（低压缩比、少量条目）必须通过。

    不能因为防炸弹把所有 docx 都拒了。
    """
    path = _make_zip(
        {"word/document.xml": b"<xml>" + b"hello world " * 100 + b"</xml>"},
        compresslevel=1,  # 低压缩比，像正常文档
    )
    check_zip_bomb(path)  # 不抛异常


def test_zip_slip_rejected(tmp_path: Path) -> None:
    """**zip slip**：压缩包内路径穿越。

    解压 `../../etc/passwd` 会写到压缩包目录之外——
    这是很多"上传后自动解压"功能的真实漏洞。
    """
    path = _make_zip({"../../evil.sh": b"malicious"})
    with pytest.raises(BadRequestError, match="非法路径"):
        check_zip_bomb(path)


def test_zip_absolute_path_rejected(tmp_path: Path) -> None:
    """压缩包内的绝对路径条目同样拒绝。"""
    path = _make_zip({"/etc/cron.d/evil": b"malicious"})
    with pytest.raises(BadRequestError, match="非法路径"):
        check_zip_bomb(path)


def test_broken_zip_rejected(tmp_path: Path) -> None:
    """损坏的 zip 必须明确报错，不能当正常文件处理。"""
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"PK\x03\x04 this is not a real zip")
    with pytest.raises(BadRequestError, match="损坏"):
        check_zip_bomb(broken)


# ==================================================== 白名单完整性


def test_allowed_extensions_are_lowercase() -> None:
    """白名单里的扩展名必须全是小写（内部比较用）。"""
    for ext in ALLOWED_EXTENSIONS:
        assert ext == ext.lower(), f"{ext} 不是小写"
        assert ext.startswith("."), f"{ext} 缺少前导点"


def test_no_dangerous_extensions_in_whitelist() -> None:
    """白名单里不能有可执行/脚本类扩展名。

    这条测试守的是"以后有人为了支持新格式顺手加错"。
    """
    dangerous = {".exe", ".sh", ".bat", ".cmd", ".ps1", ".dll", ".so", ".js", ".py"}
    overlap = ALLOWED_EXTENSIONS & dangerous
    assert not overlap, f"白名单含危险扩展名: {overlap}"
