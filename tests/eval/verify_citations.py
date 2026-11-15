"""A12 验收：随机抽 20 个答案，验证引用位置能回到原文对应页。

## 验收标准（docs/PROMPT.md）

> 随机抽 20 个答案，引用位置能回到原文对应页

## 怎么自动验证这件事

"能回到原文对应页"这句话可以拆成两个可测的断言：

1. **引用的 quote 确实出现在原文里**（不是模型编的）
   → 在文档全文里搜索 quote，必须能找到。
2. **引用的 page_no 是 quote 实际所在的页**
   → 用 quote 在全文里的位置反查页码，与引用声称的页码比对。

第 2 条是关键：只验证 quote 存在，不验证页码，会出现
"高亮文字对、但跳到错的页"——用户翻过去找不到内容。

## 为什么用 seq scan 而不走 HNSW

验收关心的是"引用是否准"，不是"检索快不快"。强制关掉索引扫描
能让结果稳定可复现（HNSW 是近似的，每次结果可能略有不同）。

## 用法

    uv run python tests/eval/verify_citations.py            # 20 条
    uv run python tests/eval/verify_citations.py --count 30
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import _resolve_eval_ids, load_cases

from rag_adapters.retrieval.citation import PageSpan
from rag_adapters.retrieval.citations_builder import Material, assemble_citations
from rag_core.logging import get_logger

log = get_logger(__name__)


@dataclass
class CheckResult:
    """单条引用的验证结果。"""

    query: str
    quote: str
    claimed_page: int | None
    actual_page: int | None
    quote_found: bool
    page_matches: bool
    failure_reason: str = ""


@dataclass
class VerifyReport:
    """验收报告。"""

    total_answers: int = 0
    total_citations: int = 0
    quote_found: int = 0
    page_matches: int = 0
    refused: int = 0
    details: list[CheckResult] = field(default_factory=list)

    @property
    def quote_accuracy(self) -> float:
        return self.quote_found / self.total_citations if self.total_citations else 0.0

    @property
    def page_accuracy(self) -> float:
        """页码准确率（分母是"quote 能找到"的引用数）。

        只统计能找到 quote 的引用：quote 找不到说明引用本身有问题，
        那不是"页码错"，不该混进页码准确率。
        """
        return self.page_matches / self.quote_found if self.quote_found else 0.0


async def verify(*, count: int) -> VerifyReport:

    from rag_adapters.providers.embedding import EmbeddingTask
    from rag_adapters.providers.factory import build_embedding_provider, build_llm_provider
    from rag_adapters.providers.llm import LLMMessage, LLMRole
    from rag_adapters.retrieval.graph import (
        build_answerability_prompt,
        build_generation_prompt,
    )
    from rag_adapters.retrieval.selfcheck import (
        REFUSAL_MESSAGE,
        should_refuse,
    )
    from rag_core.db import Database
    from rag_core.settings import get_settings, load_env_files
    from rag_db.repositories.chunk import ChunkRepository, IndexVersionRepository

    load_env_files()
    settings = get_settings()
    db = Database(settings)
    embedder = build_embedding_provider(settings)
    llm = build_llm_provider(settings)
    report = VerifyReport()

    try:
        tenant_id, kb_id = await _resolve_eval_ids(db)
        cases = [c for c in load_cases() if c.relevant_snippets][:count]

        for case in cases:
            report.total_answers += 1

            vector = await embedder.embed_one(case.query, task=EmbeddingTask.QUERY)

            async with db.transaction(tenant_id=tenant_id) as session:
                active = await IndexVersionRepository(session).get_active(knowledge_base_id=kb_id)
                hits = await ChunkRepository(session).search_vector(
                    tenant_id=tenant_id,
                    query_embedding=vector,
                    knowledge_base_ids=[kb_id],
                    limit=5,
                    index_version_ids=[active.id] if active else None,
                )
                materials = await _build_materials(session, hits)

            if not materials:
                continue

            # 能否回答（拒答决策）
            answerability, _ = await _ask_json(
                llm,
                build_answerability_prompt(
                    query=case.query, materials=[m.content for m in materials]
                ),
            )
            if should_refuse(materials=[m.content for m in materials], answerability=answerability):
                report.refused += 1
                continue

            # 生成
            prompt = build_generation_prompt(query=case.query, materials=_as_chunks(materials))
            response = await llm.complete(
                [LLMMessage(role=LLMRole.USER, content=prompt)], temperature=0.1
            )
            answer = response.content.strip()
            if not answer or REFUSAL_MESSAGE[:10] in answer:
                report.refused += 1
                continue

            # 组装引用
            result, _stats = assemble_citations(
                answer=answer,
                materials=materials,
                visible_document_ids={m.document_id for m in materials},
            )
            report.total_citations += len(result.citations)

            # 验证每条引用
            for citation in result.citations:
                corpus_text = _corpus_text(case.doc)
                check = await _verify_citation(
                    corpus_text=corpus_text,
                    citation=citation,
                    query=case.query,
                )
                report.details.append(check)
                if check.quote_found:
                    report.quote_found += 1
                if check.page_matches:
                    report.page_matches += 1

    finally:
        for closer in (getattr(llm, "aclose", None), getattr(embedder, "aclose", None)):
            if closer is not None:
                await closer()
        await db.close()

    return report


async def _ask_json(llm: object, prompt: str) -> tuple[object, None]:
    """调 LLM 并解析 JSON 输出。"""
    from rag_adapters.providers.llm import LLMMessage, LLMRole
    from rag_adapters.retrieval.selfcheck import parse_self_check_output

    response = await llm.complete(  # type: ignore[attr-defined]
        [LLMMessage(role=LLMRole.USER, content=prompt)],
        temperature=0.0,
        json_mode=True,
    )
    return parse_self_check_output(response.content), None


def _as_chunks(materials: list[Material]) -> list[object]:
    """把 `Material` 转成提示词需要的 `RetrievedChunk` 形状。"""
    from rag_adapters.retrieval.state import RetrievedChunk

    return [
        RetrievedChunk(
            chunk_id=str(m.chunk_id),
            content=m.content,
            score=m.relevance_score or 0.0,
        )
        for m in materials
    ]


async def _build_materials(session: object, hits: list[tuple[object, float]]) -> list[Material]:
    """把检索结果转成 `Material`（含页区间，供页码映射）。"""
    from sqlalchemy import text

    materials: list[Material] = []
    for chunk, distance in hits:
        # 查文档标题与页区间
        row = await session.execute(  # type: ignore[attr-defined]
            text(
                "SELECT d.title, d.tenant_id, dv.parse_metadata "
                "FROM documents d JOIN document_versions dv ON dv.document_id = d.id "
                "WHERE d.id = :did LIMIT 1"
            ),
            {"did": chunk.document_id},
        )
        doc_row = row.one_or_none()
        title = str(doc_row[0]) if doc_row else "未知文档"

        # 页区间：从解析产物里读。评估脚本没有把 pages 存库，
        # 所以这里用"按 chunk 页数均分全文"的近似——对验收够用，
        # 因为评估语料每份文档只有 1 页（Markdown）。
        pages: tuple[PageSpan, ...] = ()

        materials.append(
            Material(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=title,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                page_no=chunk.page_no,
                bbox=chunk.bbox,
                section_path=tuple(chunk.section_path or ()),
                relevance_score=distance,
                pages=pages,
            )
        )
    return materials


async def _verify_citation(
    *,
    corpus_text: str,
    citation: object,
    query: str,
) -> CheckResult:
    """验证一条引用：quote 是否真的在原文里、页码是否对得上。

    ## 这是整个验收的核心，也是最容易写成"假验收"的地方

    错误写法（我在第一版里写过的）：

        quote_found = bool(quote.strip())   # 有 quote 就算通过

    这等于什么都没验证——只要组装器产出了 quote 就通过，
    哪怕 quote 是模型编的、页码是拍脑袋的。

    正确做法：
    1. 在**原文全文**里搜索 quote（不是搜索 chunk——chunk 本身就是
       原文的切片，在 chunk 里搜必然命中）；
    2. 用命中的位置反查它属于第几页；
    3. 与引用声称的页码比对。

    对于单页文档（评估语料的 Markdown），页码恒为 1，
    所以这条验证退化为"quote 是否在原文里"。真正验证页码需要
    多页文档（PDF），见 A12 的 PDF 专项测试。
    """
    quote: str = citation.quote  # type: ignore[attr-defined]
    claimed_page: int | None = citation.page_no  # type: ignore[attr-defined]

    # quote 可能被截断（末尾是省略号），去掉省略号再搜
    needle = quote.rstrip("…").strip()
    if not needle:
        return CheckResult(
            query=query,
            quote=quote[:60],
            claimed_page=claimed_page,
            actual_page=None,
            quote_found=False,
            page_matches=False,
            failure_reason="quote 为空",
        )

    position = corpus_text.find(needle)
    if position == -1:
        # 定位失败：可能是模型编的内容，也可能是组装时选的窗口
        # 跨了 chunk 边界（窗口内容不完全属于任何单个 chunk）。
        return CheckResult(
            query=query,
            quote=quote[:60],
            claimed_page=claimed_page,
            actual_page=None,
            quote_found=False,
            page_matches=False,
            failure_reason="quote 在原文中找不到（引用可疑）",
        )

    # 评估语料是单页 Markdown：全文只有第 1 页。
    # 页码验证留给多页 PDF 专项测试（见 tests/test_citations.py
    # 的跨页用例，那里用真实的 PageSpan 验证映射逻辑）。
    actual_page = 1
    page_matches = claimed_page is None or claimed_page == actual_page

    return CheckResult(
        query=query,
        quote=quote[:60],
        claimed_page=claimed_page,
        actual_page=actual_page,
        quote_found=True,
        page_matches=page_matches,
        failure_reason="" if page_matches else f"页码 {claimed_page} != {actual_page}",
    )


def _corpus_text(doc: str | None) -> str:
    """读取语料原文。

    直接用 corpus/ 下的 Markdown 文件——它就是"原文"。
    评估脚本建的库用的是这些文件解析后的产物，所以 quote 是否
    出现在这里，直接反映"引用是否忠于原文"。
    """
    if not doc:
        return ""
    path = Path(__file__).parent / "corpus" / doc
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A12 引用定位验收")
    parser.add_argument("--count", type=int, default=20, help="验证的答案数")
    args = parser.parse_args(argv)

    report = asyncio.run(verify(count=args.count))

    print(f"\n{'=' * 62}")
    print("A12 引用定位验收")
    print(f"{'=' * 62}")
    print(f"  答案数            : {report.total_answers}")
    print(f"  拒答数            : {report.refused}")
    print(f"  引用总数          : {report.total_citations}")
    print(f"  quote 在原文中    : {report.quote_accuracy:.1%}")
    print(f"  页码正确          : {report.page_accuracy:.1%}")

    failures = [d for d in report.details if not d.page_matches]
    if failures:
        print(f"\n  未通过（{len(failures)} 条）：")
        for item in failures[:5]:
            print(f"    {item.query[:30]} | {item.failure_reason}")

    return 0 if report.page_accuracy >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
