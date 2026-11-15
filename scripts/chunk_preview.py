"""分块效果可视化：把切分结果渲染成 HTML，肉眼验收边界质量。

## 为什么需要"肉眼验收"

自动化测试能验证：不越界、位置成对、内容与切片一致、无空块。
**验证不了**的是："这个切点是不是切在了句子中间？"

这类问题只有人眼看得出来：

    "员工出差前须提交出差申请，经直属上级审批后方可预订机票与"
    "酒店。未审批的差旅费用不予报销。"

每块单独看都合法、偏移都对，但第一块读起来是半句话 —— 向量语义漂移。
所以 `docs/PROMPT.md` 要求"chunk 可视化脚本（导出 HTML 肉眼验收边界质量）"。

## 输出的 HTML 长什么样

- 上半部分：**原文**，按 char_start/char_end 着色标出每个 chunk 覆盖的区间，
  重叠部分叠加。能一眼看出"哪段文字落在两个块里"、"哪段没被覆盖"。
- 下半部分：**chunk 列表**，每块显示序号、长度、类型、章节路径、正文。
- 顶部：统计（块数、长度分布、覆盖率、重叠率）。

## 覆盖率与重叠率为什么重要

- **覆盖率 < 100%**：有内容没进任何 chunk → 永远检索不到。
  短块合并逻辑有 bug 时最容易出现。
- **重叠率过高**：同一段内容重复向量化（花钱）+ 检索结果重复占位。

这两个指标是分块质量最直接的量化信号。

## 用法

    # 对比一份文档的全部策略
    uv run python scripts/chunk_preview.py --pdf path/to/doc.pdf --all-strategies

    # 指定策略与参数
    uv run python scripts/chunk_preview.py --text doc.txt --strategy recursive --size 500

    # 输出到指定文件
    uv run python scripts/chunk_preview.py --pdf doc.pdf --out /tmp/chunks.html
"""

from __future__ import annotations

import argparse
import html
import statistics
import sys
from pathlib import Path

from rag_adapters.chunkers.base import (
    ChunkConfig,
    ChunkDraft,
    Coverage,
    compute_coverage,
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
    dump_parsed,
    load_parsed,
)
from rag_adapters.parsers.factory import parse_document
from rag_contracts.enums import DocumentFormat

# 每块一个颜色（按索引循环）。用低饱和度背景色，保证文字可读。
_COLORS = (
    "#fff3cd",
    "#d1ecf1",
    "#d4edda",
    "#f8d7da",
    "#e2d9f3",
    "#ffe5b4",
    "#cfe8ff",
    "#e8f5c8",
    "#ffd9ec",
    "#d9f2e6",
)


def render_html(
    *,
    parsed: ParsedArtifact,
    strategy: str,
    config: ChunkConfig,
    drafts: list[ChunkDraft],
    problems: list[str],
) -> str:
    """渲染单份结果的 HTML。"""
    coverage = compute_coverage(parsed.text, drafts)
    lengths = [len(draft.content) for draft in drafts]

    # ---- 原文着色：按字符位置标记属于哪个 chunk ----
    # 用 span 包住每段被某块独占的区间；重叠区间单独标注。
    ownership: list[list[int]] = [[] for _ in range(len(parsed.text))]
    for position, draft in enumerate(drafts):
        if draft.char_start is None or draft.char_end is None:
            continue
        for offset in range(max(0, draft.char_start), min(len(parsed.text), draft.char_end)):
            ownership[offset].append(position)

    highlighted: list[str] = []
    cursor = 0
    text_length = len(parsed.text)
    while cursor < text_length:
        owners = tuple(ownership[cursor])
        run_end = cursor + 1
        while run_end < text_length and tuple(ownership[run_end]) == owners:
            run_end += 1

        segment = html.escape(parsed.text[cursor:run_end])
        if not owners:
            # 未覆盖区间：红底强调（这是问题，内容检索不到）
            highlighted.append(
                f'<span class="uncovered" title="未被任何 chunk 覆盖">{segment}</span>'
            )
        elif len(owners) == 1:
            color = _COLORS[owners[0] % len(_COLORS)]
            highlighted.append(
                f'<span style="background:{color}" title="chunk #{owners[0]}">{segment}</span>'
            )
        else:
            # 重叠区间：斜纹背景 + 标注所有覆盖它的 chunk
            label = ", ".join(f"#{owner}" for owner in owners)
            highlighted.append(
                f'<span class="overlap" title="被 chunks {label} 覆盖">{segment}</span>'
            )
        cursor = run_end

    rows: list[str] = []
    for position, draft in enumerate(drafts):
        color = _COLORS[position % len(_COLORS)]
        path_text = " > ".join(draft.section_path) if draft.section_path else "—"
        rows.append(
            f"""
        <tr>
          <td style="background:{color}">{draft.chunk_index}</td>
          <td>{len(draft.content)}</td>
          <td>{html.escape(draft.chunk_type)}</td>
          <td>{draft.chunk_level}</td>
          <td>{draft.page_no if draft.page_no is not None else "—"}</td>
          <td class="path">{html.escape(path_text)}</td>
          <td class="content"><pre>{html.escape(draft.content)}</pre></td>
        </tr>"""
        )

    problem_html = ""
    if problems:
        items = "".join(f"<li>{html.escape(problem)}</li>" for problem in problems[:20])
        problem_html = (
            f'<div class="problems"><h3>校验失败（{len(problems)} 项）</h3><ul>{items}</ul></div>'
        )

    stats = _stats_block(lengths)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>分块预览 · {html.escape(strategy)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Noto Sans CJK SC", sans-serif;
         max-width: 1200px; margin: 2rem auto; padding: 0 1rem; line-height: 1.7; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 2px solid #eee; padding-bottom: .3rem; }}
  .metrics {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }}
  .metric {{ background: #f8f9fa; border-radius: 6px; padding: .6rem 1rem; }}
  .metric b {{ display: block; font-size: 1.3rem; color: #0b5ed7; }}
  .source {{ white-space: pre-wrap; background: #fdfdfd; border: 1px solid #dee2e6;
             border-radius: 6px; padding: 1rem; max-height: 500px;
             overflow: auto; font-size: .92rem; }}
  .uncovered {{ background: #ff6b6b; color: #fff; }}
  .overlap {{ background: repeating-linear-gradient(
      45deg, #ffd43b, #ffd43b 6px, #ffe8a1 6px, #ffe8a1 12px); }}
  table {{ border-collapse: collapse; width: 100%; font-size: .88rem; margin-top: .5rem; }}
  th, td {{ border: 1px solid #dee2e6; padding: .4rem .5rem; vertical-align: top; }}
  th {{ background: #f1f3f5; text-align: left; position: sticky; top: 0; }}
  td.content pre {{ margin: 0; white-space: pre-wrap; word-break: break-word;
                   max-height: 160px; overflow: auto; }}
  td.path {{ font-size: .8rem; color: #666; max-width: 180px; }}
  .problems {{ background: #fff5f5; border-left: 4px solid #ff6b6b;
               padding: .8rem 1rem; margin: 1rem 0; }}
  .problems ul {{ margin: .3rem 0; padding-left: 1.2rem; }}
</style>
</head>
<body>
<h1>分块预览 · <code>{html.escape(strategy)}</code></h1>
<div class="metrics">
  <div class="metric">块数<b>{len(drafts)}</b></div>
  <div class="metric">覆盖率<b>{coverage.coverage_ratio:.1%}</b></div>
  <div class="metric">重叠率<b>{coverage.overlap_ratio:.1%}</b></div>
  <div class="metric">配置<b>{config.chunk_size} / {config.chunk_overlap}</b></div>
  <div class="metric">解析器<b>{html.escape(parsed.manifest.parser_name)}</b></div>
</div>
{stats}
{problem_html}

<h2>原文覆盖（颜色 = 所属 chunk，红色 = 未覆盖，斜纹 = 重叠）</h2>
<div class="source">{"".join(highlighted)}</div>

<h2>Chunk 列表</h2>
<table>
  <thead><tr>
    <th>#</th><th>长度</th><th>类型</th><th>层级</th><th>页</th><th>章节</th><th>内容</th>
  </tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</body>
</html>"""


def _stats_block(lengths: list[int]) -> str:
    """长度分布统计。

    为什么看 P90 而不只是平均值：平均值会被少数超长块（表格）掩盖，
    而 P90 反映"绝大多数块有多长"。分块参数调优看 P90 更准。
    """
    if not lengths:
        return "<p>无 chunk</p>"

    ordered = sorted(lengths)
    p90_index = min(int(len(ordered) * 0.9), len(ordered) - 1)
    return (
        '<div class="metrics">'
        f'<div class="metric">最短<b>{ordered[0]}</b></div>'
        f'<div class="metric">中位数<b>{statistics.median(ordered):.0f}</b></div>'
        f'<div class="metric">平均<b>{statistics.fmean(ordered):.0f}</b></div>'
        f'<div class="metric">P90<b>{ordered[p90_index]}</b></div>'
        f'<div class="metric">最长<b>{ordered[-1]}</b></div>'
        "</div>"
    )


def _parsed_from_source(source: Path, file_format: DocumentFormat) -> ParsedArtifact:
    """把源文件解析成 `ParsedArtifact`（走真实解析器 + 产物序列化往返）。

    往返一次（dump → load）是**故意的**：这样预览脚本验证的正是
    worker 实际走的路径（解析 → 写 JSONL → 切分读 JSONL）。
    直接用内存里的 `ParsedDocument` 会绕过产物格式，掩盖序列化 bug。
    """
    if source.suffix in {".jsonl", ".txt"} and file_format == DocumentFormat.TEXT:
        # 直接喂已存在的解析产物
        return load_parsed(source.read_bytes())

    parsed_document = parse_document(source, format=file_format)
    artifact_bytes = dump_parsed(
        text=parsed_document.text,
        blocks=[
            BlockRecord(
                text=block.text,
                block_type=block.block_type,
                char_start=block.char_start,
                char_end=block.char_end,
                page_no=block.page_no,
                bbox=block.bbox,
                level=block.level,
                metadata=dict(block.metadata),
            )
            for block in parsed_document.blocks
        ],
        pages=[
            PageRecord(
                page_no=page.page_no,
                width=page.width,
                height=page.height,
                char_start=page.char_start,
                char_end=page.char_end,
                is_scanned=page.is_scanned,
            )
            for page in parsed_document.pages
        ],
        parser_name=parsed_document.parser_name,
        document_format=parsed_document.format.value,
        has_scanned_pages=parsed_document.has_scanned_pages,
        metadata=dict(parsed_document.metadata),
    )
    return load_parsed(artifact_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分块效果可视化")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--pdf", type=Path, help="PDF 文件路径")
    source_group.add_argument("--text", type=Path, help="纯文本文件路径")
    source_group.add_argument("--markdown", type=Path, help="Markdown 文件路径")
    source_group.add_argument("--artifact", type=Path, help="已生成的 .jsonl 解析产物")

    parser.add_argument("--strategy", default="recursive", help="切分策略名")
    parser.add_argument(
        "--all-strategies", action="store_true", help="对全部策略各出一份 HTML（用于对比）"
    )
    parser.add_argument("--size", type=int, default=800, help="chunk_size")
    parser.add_argument("--overlap", type=int, default=100, help="chunk_overlap")
    parser.add_argument("--min-size", type=int, default=50, help="min_chunk_size")
    parser.add_argument("--out", type=Path, help="输出 HTML 路径（省略则打印到 stdout）")
    args = parser.parse_args(argv)

    # ---- 载入解析产物 ----
    if args.artifact:
        parsed = load_parsed(args.artifact.read_bytes())
    elif args.pdf:
        parsed = _parsed_from_source(args.pdf, DocumentFormat.PDF)
    elif args.markdown:
        parsed = _parsed_from_source(args.markdown, DocumentFormat.MARKDOWN)
    else:
        parsed = _parsed_from_source(args.text, DocumentFormat.TEXT)

    config = ChunkConfig(
        chunk_size=args.size, chunk_overlap=args.overlap, min_chunk_size=args.min_size
    )

    strategies = registered_strategies() if args.all_strategies else (args.strategy,)

    if args.all_strategies:
        _write_comparison(parsed, strategies, config, args.out)
        return 0

    html_text = _render_one(parsed, strategies[0], config)
    if args.out:
        args.out.write_text(html_text, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(html_text)
    return 0


def _render_one(parsed: ParsedArtifact, strategy: str, config: ChunkConfig) -> str:
    chunker = build_chunker(strategy, config=config)
    drafts = chunker.chunk(parsed)
    problems = verify_chunks(drafts, parsed)
    return render_html(
        parsed=parsed, strategy=strategy, config=config, drafts=drafts, problems=problems
    )


def _write_comparison(
    parsed: ParsedArtifact,
    strategies: tuple[str, ...],
    config: ChunkConfig,
    out: Path | None,
) -> None:
    """生成策略对比页（一张表 + 各策略独立详情）。

    对比表的列选择理由：
    - 块数：影响索引规模与检索候选量；
    - P90 长度：反映块长是否受控；
    - 覆盖率：**必须 100%**，否则有内容永远检索不到；
    - 重叠率：影响成本与检索结果重复度。
    """
    results: list[tuple[str, list[ChunkDraft], list[str], Coverage]] = []
    for strategy in strategies:
        chunker = build_chunker(strategy, config=config)
        drafts = chunker.chunk(parsed)
        problems = verify_chunks(drafts, parsed)
        coverage = compute_coverage(parsed.text, drafts)
        results.append((strategy, drafts, problems, coverage))

    rows: list[str] = []
    for strategy, drafts, problems, coverage in results:
        lengths = sorted(len(draft.content) for draft in drafts)
        p90 = lengths[min(int(len(lengths) * 0.9), len(lengths) - 1)] if lengths else 0
        status = (
            '<span style="color:#c92a2a">FAIL</span>'
            if problems
            else '<span style="color:#2b8a3e">OK</span>'
        )
        # 降级提示：语义策略缺 embedding 时会退化为 recursive，
        # 结果与 recursive 完全一致。不标出来会让对比表误导人
        # （"semantic 和 recursive 一样"其实是没测到 semantic）。
        degraded = (
            '<span style="color:#e8590c">降级为 recursive</span>'
            if strategy in strategies_requiring_embedding()
            else "—"
        )
        rows.append(
            f"<tr><td><code>{html.escape(strategy)}</code></td>"
            f"<td>{len(drafts)}</td>"
            f"<td>{lengths[0] if lengths else 0}</td>"
            f"<td>{p90}</td>"
            f"<td>{statistics.median(lengths):.0f}</td>"
            f"<td>{coverage.coverage_ratio:.1%}</td>"
            f"<td>{coverage.overlap_ratio:.1%}</td>"
            f"<td>{status}</td>"
            f"<td>{degraded}</td></tr>"
        )

    sections: list[str] = []
    for strategy, drafts, problems, _coverage in results:
        sections.append(
            f"<h2>策略 <code>{html.escape(strategy)}</code></h2>"
            + render_html(
                parsed=parsed,
                strategy=strategy,
                config=config,
                drafts=drafts,
                problems=problems,
            )
            .split("<body>", 1)[1]
            .replace("</body>", "")
            .replace("</html>", "")
        )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>分块策略对比</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Noto Sans CJK SC", sans-serif;
         max-width: 1400px; margin: 2rem auto; padding: 0 1rem; line-height: 1.7; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #dee2e6; padding: .45rem .6rem; text-align: left; }}
  th {{ background: #f1f3f5; }}
  h2 {{ margin-top: 3rem; border-top: 3px solid #0b5ed7; padding-top: 1rem; }}
</style></head><body>
<h1>分块策略对比</h1>
<p>文档：{parsed.manifest.parser_name} · {parsed.manifest.text_length} 字符 ·
   {parsed.manifest.block_count} 块 · {parsed.manifest.page_count} 页</p>
<h2>指标对比</h2>
<table>
  <thead><tr>
    <th>策略</th><th>块数</th><th>最短</th><th>P90</th><th>中位数</th>
    <th>覆盖率</th><th>重叠率</th><th>校验</th><th>备注</th>
  </tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>
<p><strong>覆盖率必须 100%</strong>：低于 100% 说明有内容没进任何 chunk，永远检索不到。<br>
<strong>重叠率高</strong>意味着重复向量化（成本）与检索结果重复占位。<br>
<strong>语义策略需要 embedding</strong>：预览脚本不加载模型，
所以 semantic 会降级为 recursive（表现为与 recursive 的结果完全一致）。</p>
{"".join(sections)}
</body></html>"""

    if out:
        out.write_text(page, encoding="utf-8")
        print(f"已写入对比页 {out}")
    else:
        print(page)


if __name__ == "__main__":
    sys.exit(main())
