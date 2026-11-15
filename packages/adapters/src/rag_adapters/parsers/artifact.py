"""解析产物的持久化格式（worker 产出，chunker 消费）。

## 为什么要单独定义一个格式，而不是"只存文本"

分块器必须产出 `chunks.page_no` / `bbox` / `char_start` / `char_end` /
`section_path` —— 这四类字段是前端"点击引用跳到原文高亮"的唯一依据。

如果解析阶段只把 `parsed.text` 落盘，那么：

- 字符偏移还能自己数（但脆：任何归一化都会让偏移错位）；
- **页码与 bbox 永久丢失**（文本里没有这个信息）；
- 结果：引用只能显示"第 N 段"，无法在原 PDF 上框选。

所以解析产物必须是**结构化的**：文本 + 块列表 + 页列表一起落盘。
A8-2 最初只存了纯文本（`.txt`），A9 补上 `.json`（结构化）并保留
`.txt`（供全文检索/调试比对）。两者写同一份解析结果，内容自洽。

## 格式选择：JSONL vs 单个 JSON

单个 JSON：一次 `json.loads` 全量进内存。一份 300 页 PDF 的块列表
可能有 2 万个块、几十 MB —— 反序列化后内存翻几倍。

JSONL（每行一个对象，首行是头部）：可以逐行流式读，只保留当前需要
的块。切分器本来就是顺序消费块的，流式读天然匹配。

**但**：`verify_char_offsets` 之类的整体校验需要全部块。
所以这里提供两种入口——`read_manifest()` 只读头部（O(1)），
`iter_blocks()` 流式读块（O(1) 内存），`load_parsed()` 全量读
（仅用于校验/调试，大文件慎用）。

## 为什么版本化格式（`format_version`）

产物是跨进程、跨时间传递的：A8 的 worker 写，A9 的 chunker 读，
中间可能隔了几天、甚至跨了一次部署。

没有版本号的话，改了字段名之后旧产物会被"成功读取但字段全空"——
表现为"切分出来的 chunk 没有页码"，且不报错。
带版本号 + 校验，就能明确拒绝或迁移。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

# 产物格式版本。**加减字段时必须递增**，并在 read 侧做兼容处理。
PARSED_FORMAT_VERSION = 1

# 存储键后缀：.jsonl 与 .txt 成对存在
PARSED_JSONL_SUFFIX = ".jsonl"
PARSED_TEXT_SUFFIX = ".txt"


class ParsedArtifactError(Exception):
    """产物读取/格式错误。

    不可重试：产物是上一次解析写下的，重读不会变。
    真正的修法是重跑解析任务（重新生成产物），或修 chunker 的兼容逻辑。
    """


@dataclass(frozen=True, slots=True)
class BlockRecord:
    """一个结构块的持久化形态（对应 `parsers.TextBlock`）。"""

    text: str
    block_type: str
    char_start: int
    char_end: int
    page_no: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    level: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PageRecord:
    """一页的持久化形态（对应 `parsers.PageInfo`）。"""

    page_no: int
    width: float
    height: float
    char_start: int
    char_end: int
    is_scanned: bool = False


@dataclass(frozen=True, slots=True)
class ParsedManifest:
    """产物头部（不含块，便于快速判断能否切分）。"""

    format_version: int
    parser_name: str
    document_format: str
    text_length: int
    block_count: int
    page_count: int
    has_scanned_pages: bool
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedArtifact:
    """全量产物（含块与页）。用于校验与小文件。"""

    manifest: ParsedManifest
    text: str
    blocks: list[BlockRecord]
    pages: list[PageRecord]


# ---------------------------------------------------------------- 序列化


def dump_parsed(
    *,
    text: str,
    blocks: list[BlockRecord],
    pages: list[PageRecord],
    parser_name: str,
    document_format: str,
    has_scanned_pages: bool,
    metadata: dict[str, str] | None = None,
) -> bytes:
    """序列化成 JSONL 字节串。

    第一行是 manifest，其后每行一个块，最后每行一个页
    （用 `kind` 字段区分，避免"第二段从哪儿开始"这种位置耦合）。

    为什么 manifest 单独一行而不是包在最外层对象里：
    流式读时先读第一行就能拿到 `block_count`/`text_length`，
    据此决定"要不要全量加载"，不必扫描整个文件。
    """
    lines: list[str] = []

    manifest = {
        "kind": "manifest",
        "format_version": PARSED_FORMAT_VERSION,
        "parser_name": parser_name,
        "document_format": document_format,
        "text_length": len(text),
        "block_count": len(blocks),
        "page_count": len(pages),
        "has_scanned_pages": has_scanned_pages,
        "metadata": metadata or {},
    }
    lines.append(json.dumps(manifest, ensure_ascii=False))

    # 全文单独一行（可能很长，但 json.dumps 会正确转义换行）。
    # 不拆成多行：拆了就要在读取侧重新拼装，容易与块的行搞混。
    lines.append(json.dumps({"kind": "text", "value": text}, ensure_ascii=False))

    for block in blocks:
        lines.append(
            json.dumps(
                {
                    "kind": "block",
                    "text": block.text,
                    "block_type": block.block_type,
                    "char_start": block.char_start,
                    "char_end": block.char_end,
                    "page_no": block.page_no,
                    # tuple → list：JSON 没有 tuple，读回时再转
                    "bbox": list(block.bbox) if block.bbox is not None else None,
                    "level": block.level,
                    "metadata": block.metadata,
                },
                ensure_ascii=False,
            )
        )

    for page in pages:
        lines.append(
            json.dumps(
                {
                    "kind": "page",
                    "page_no": page.page_no,
                    "width": page.width,
                    "height": page.height,
                    "char_start": page.char_start,
                    "char_end": page.char_end,
                    "is_scanned": page.is_scanned,
                },
                ensure_ascii=False,
            )
        )

    return ("\n".join(lines) + "\n").encode("utf-8")


def iter_lines(data: bytes) -> Iterator[dict[str, Any]]:
    """逐行解析 JSONL。空行跳过（尾部换行会产生空行）。"""
    for line_no, raw in enumerate(data.split(b"\n"), start=1):
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            # 带行号：产物损坏时要能定位到哪一行，而不是只知道"解析失败"
            raise ParsedArtifactError(f"产物第 {line_no} 行不是合法 JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ParsedArtifactError(f"产物第 {line_no} 行不是对象")
        yield parsed


def read_manifest(data: bytes) -> ParsedManifest:
    """只读头部。产物格式不对时抛 `ParsedArtifactError`。"""
    first = next(iter_lines(data), None)
    if first is None or first.get("kind") != "manifest":
        raise ParsedArtifactError("产物缺少 manifest 头（格式损坏或不是本格式）")

    version = first.get("format_version")
    if version != PARSED_FORMAT_VERSION:
        # 明确拒绝而不是"尽力解析"：字段语义变了，硬解析会静默产出
        # 没有页码/偏移的 chunk —— 看起来成功，引用功能却悄悄失效。
        raise ParsedArtifactError(
            f"产物格式版本 {version} 与当前版本 {PARSED_FORMAT_VERSION} 不兼容；"
            "请重跑解析任务重新生成产物"
        )

    return ParsedManifest(
        format_version=int(version),
        parser_name=str(first.get("parser_name", "")),
        document_format=str(first.get("document_format", "")),
        text_length=int(first.get("text_length", 0)),
        block_count=int(first.get("block_count", 0)),
        page_count=int(first.get("page_count", 0)),
        has_scanned_pages=bool(first.get("has_scanned_pages", False)),
        metadata=dict(first.get("metadata") or {}),
    )


def _bbox_from_json(raw: Any) -> tuple[float, float, float, float] | None:
    """把 JSON 里的 bbox 数组转成 4 元组。

    长度不是 4 说明产物被写坏或格式不兼容 —— 静默返回 None 会让
    "引用高亮时好时坏"，所以这里直接当格式错误处理。
    """
    if not raw:
        return None
    if not isinstance(raw, list) or len(raw) != 4:
        raise ParsedArtifactError(f"bbox 字段格式非法：{raw!r}")
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def load_parsed(data: bytes) -> ParsedArtifact:
    """全量读取。

    大文件（几十 MB）会把所有块读进内存 —— 只在需要整体校验时用。
    正常切分路径应使用 `read_manifest()` + 逐块流式消费。
    """
    manifest = read_manifest(data)

    text: str | None = None
    blocks: list[BlockRecord] = []
    pages: list[PageRecord] = []

    for item in iter_lines(data):
        kind = item.get("kind")
        if kind == "manifest":
            continue
        if kind == "text":
            text = str(item.get("value", ""))
        elif kind == "block":
            bbox_raw = item.get("bbox")
            blocks.append(
                BlockRecord(
                    text=str(item.get("text", "")),
                    block_type=str(item.get("block_type", "text")),
                    char_start=int(item.get("char_start", 0)),
                    char_end=int(item.get("char_end", 0)),
                    page_no=item.get("page_no"),
                    bbox=_bbox_from_json(bbox_raw),
                    level=int(item.get("level", 0)),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        elif kind == "page":
            pages.append(
                PageRecord(
                    page_no=int(item.get("page_no", 0)),
                    width=float(item.get("width", 0)),
                    height=float(item.get("height", 0)),
                    char_start=int(item.get("char_start", 0)),
                    char_end=int(item.get("char_end", 0)),
                    is_scanned=bool(item.get("is_scanned", False)),
                )
            )
        else:
            # 未知 kind：可能是新版本产物多写了字段。
            # 忽略而不是报错 —— 向前兼容（旧代码读新产物）。
            continue

    if text is None:
        raise ParsedArtifactError("产物缺少 text 行")

    # 一致性校验：数量对不上说明产物被截断或写坏了。
    # 这种情况必须失败：按不足的块切分会得到"少了一半内容的文档"，
    # 而且完全静默（用户只看到"检索不到下半部分"）。
    if len(blocks) != manifest.block_count:
        raise ParsedArtifactError(
            f"块数量不一致：manifest={manifest.block_count}，实际={len(blocks)}"
        )
    if len(pages) != manifest.page_count:
        raise ParsedArtifactError(
            f"页数量不一致：manifest={manifest.page_count}，实际={len(pages)}"
        )
    if len(text) != manifest.text_length:
        raise ParsedArtifactError(
            f"文本长度不一致：manifest={manifest.text_length}，实际={len(text)}"
        )

    return ParsedArtifact(manifest=manifest, text=text, blocks=blocks, pages=pages)


async def read_all(chunks: AsyncIterator[bytes]) -> bytes:
    """把异步迭代器拼成字节串。

    切分任务需要随机访问全文与块列表（不是纯顺序），
    所以必须整体读入。**只用于解析产物**——它已按版本隔离且通常
    几 MB 到几十 MB；原始文件（可能 GB 级）绝不这样读。
    """
    buffer = bytearray()
    async for chunk in chunks:
        buffer.extend(chunk)
    return bytes(buffer)


__all__ = [
    "PARSED_FORMAT_VERSION",
    "PARSED_JSONL_SUFFIX",
    "PARSED_TEXT_SUFFIX",
    "BlockRecord",
    "PageRecord",
    "ParsedArtifact",
    "ParsedArtifactError",
    "ParsedManifest",
    "dump_parsed",
    "iter_lines",
    "load_parsed",
    "read_all",
    "read_manifest",
]
