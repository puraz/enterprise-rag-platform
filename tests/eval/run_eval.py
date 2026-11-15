"""检索质量评估：建库 + 跑黄金集 + 出指标报告。

## 为什么需要这个脚本（A11 的硬要求）

`docs/PROMPT.md` 的 A11 验收标准是：

> 每加一个节点用同一批 20+ query 做加前加后对比，**数据说话，不凭感觉**

没有这个脚本就无法回答"加 rerank 到底提升了多少"。凭感觉判断
"加了 rerank 应该更准"是最常见的自欺——项目里真正有效的节点
往往不是直觉认为的那个。

## 指标定义

- **Recall@k**：期望片段中有多少比例出现在前 k 个结果里。
  衡量"该召回的有没有召回到"，是检索最关键的下限指标。
- **MRR@k**（Mean Reciprocal Rank）：第一个命中结果的排名倒数的均值。
  衡量"正确答案排得靠不靠前"。Recall 相同但 MRR 不同，说明
  一个把答案排第 1、另一个排第 8——对 LLM 影响很大。
- **Hit@k**：至少命中一个期望片段的查询占比。
  对 unanswerable 类（无期望片段）不计入。
- **误报率**：unanswerable 查询返回了结果的比例。
  这条单独的指标很重要——检索系统"总能返回点什么"是危险的，
  它会诱导 LLM 用不相关材料编答案。

## 判定命中的方式

`chunk.content` 是否**包含**标注的 `relevant_snippets` 之一。
用原文片段而非"第几个 chunk"：换分块策略后 chunk 边界会变，
但"这段原文"不会变。

## 用法

    # 建库（首次或语料变更后）
    uv run python tests/eval/run_eval.py --setup

    # 跑基线
    uv run python tests/eval/run_eval.py --label baseline

    # 加节点后对比（结果存进 reports/，脚本自动打印与上次的差异）
    uv run python tests/eval/run_eval.py --label with-rerank

    # 只看报告
    uv run python tests/eval/run_eval.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

EVAL_DIR = Path(__file__).parent
CORPUS_DIR = EVAL_DIR / "corpus"
QUERIES_FILE = EVAL_DIR / "queries.jsonl"
REPORTS_DIR = EVAL_DIR / "reports"

# 评估用的固定租户/库标识。
#
# **用固定值而不是随机**：`--setup` 建一次库，后续多次评估复用同一批
# 向量。随机 ID 会导致每次评估都重建索引（bge-m3 算 30+ chunk 要几十秒），
# 且不同次评估的数据不可比。
EVAL_TENANT_SLUG = "eval-tenant"
EVAL_KB_NAME = "eval-kb"


@dataclass
class EvalCase:
    """一条评估用例。"""

    id: str
    query: str
    doc: str | None
    relevant_snippets: list[str]
    category: str


@dataclass
class CaseResult:
    """单条用例的结果。"""

    case: EvalCase
    retrieved: list[tuple[str, float]]  # (content, distance)
    first_hit_rank: int | None  # 1-based；None 表示未命中
    latency_ms: float


@dataclass
class Report:
    """一次评估的汇总。"""

    label: str
    k: int
    results: list[CaseResult]
    elapsed_s: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))

    def _answerable(self) -> list[CaseResult]:
        return [r for r in self.results if r.case.relevant_snippets]

    @property
    def recall(self) -> float:
        """期望片段被召回的比例（对所有 answerable 用例取均值）。

        多片段用例（multi_snippet）要求**全部**片段都出现才算这条完全命中——
        只召回一半仍会让 LLM 缺少关键信息（如"试用期三个月"召回但
        "工资八折"没召回）。
        """
        cases = self._answerable()
        if not cases:
            return 0.0
        scores: list[float] = []
        for result in cases:
            contents = [content for content, _ in result.retrieved]
            found = sum(
                1
                for snippet in result.case.relevant_snippets
                if any(snippet in content for content in contents)
            )
            scores.append(found / len(result.case.relevant_snippets))
        return sum(scores) / len(scores)

    @property
    def mrr(self) -> float:
        cases = self._answerable()
        if not cases:
            return 0.0
        return sum(1.0 / result.first_hit_rank for result in cases if result.first_hit_rank) / len(
            cases
        )

    @property
    def hit_rate(self) -> float:
        cases = self._answerable()
        if not cases:
            return 0.0
        return sum(1 for r in cases if r.first_hit_rank) / len(cases)

    @property
    def false_positive_rate(self) -> float:
        """unanswerable 查询仍返回结果的比例。

        检索系统"总能返回点什么"是危险的：会诱导 LLM 用不相关材料
        编出看起来有依据的答案。这个指标衡量"能不能说不知道"。
        """
        cases = [r for r in self.results if not r.case.relevant_snippets]
        if not cases:
            return 0.0
        return sum(1 for r in cases if r.retrieved) / len(cases)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)


def load_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in QUERIES_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            EvalCase(
                id=raw["id"],
                query=raw["query"],
                doc=raw.get("doc"),
                relevant_snippets=raw.get("relevant_snippets", []),
                category=raw.get("category", "direct"),
            )
        )
    return cases


# ---------------------------------------------------------------- 建库


async def setup_corpus(*, force: bool = False) -> tuple[uuid.UUID, uuid.UUID]:
    """把 corpus/ 下的文档建库（解析 → 切分 → 向量化）。

    返回 `(tenant_id, knowledge_base_id)`。

    幂等：已建过则直接返回现有 ID（除非 `--force`）。
    """
    from rag_core.db import Database
    from rag_core.settings import get_settings, load_env_files
    from rag_db.repositories import (
        KnowledgeBaseRepository,
    )

    load_env_files()
    settings = get_settings()
    db = Database(settings)

    try:
        # 事务边界要小心：
        # - `transaction_bypass_rls()` 内部是 `async with session.begin()`，
        #   脚本里再 commit 会报 "Can't operate on closed transaction"；
        # - `db.session()` 用应用角色（rag_app），受 RLS 约束，
        #   建库时会因"新行违背行级安全策略"失败（此时还没有 tenant 上下文）。
        #
        # 所以：**只读操作用 `db.session()`，写入用 `transaction(tenant_id=...)`**。
        # 新建租户本身必须绕过 RLS（此时租户还不存在，无从设置上下文），
        # 用短事务 `db.session()` 逐条提交。
        async with db.session() as session:
            # 找现有评估租户
            row = await session.execute(
                text("SELECT id FROM tenants WHERE slug = :s"), {"s": EVAL_TENANT_SLUG}
            )
            existing_tenant = row.scalar_one_or_none()

            if existing_tenant is not None and not force:
                row = await session.execute(
                    text("SELECT id FROM knowledge_bases WHERE tenant_id = :t AND name = :n"),
                    {"t": existing_tenant, "n": EVAL_KB_NAME},
                )
                kb = row.scalar_one_or_none()
                if kb is not None:
                    count = (
                        await session.execute(
                            text("SELECT count(*) FROM chunks WHERE tenant_id = :t"),
                            {"t": existing_tenant},
                        )
                    ).scalar_one()
                    if int(count) > 0:
                        print(f"复用现有评估库：tenant={existing_tenant} kb={kb} chunks={count}")
                        return existing_tenant, kb

            if existing_tenant is not None:
                # force：清掉旧数据
                for statement in (
                    "DELETE FROM chunk_embeddings WHERE tenant_id = :t",
                    "DELETE FROM chunks WHERE tenant_id = :t",
                    "DELETE FROM index_versions WHERE tenant_id = :t",
                    "DELETE FROM tasks WHERE tenant_id = :t",
                    "DELETE FROM document_versions WHERE tenant_id = :t",
                    "DELETE FROM documents WHERE tenant_id = :t",
                    "DELETE FROM knowledge_bases WHERE tenant_id = :t",
                    "DELETE FROM blobs WHERE tenant_id = :t",
                ):
                    await session.execute(text(statement), {"t": existing_tenant})
                await session.execute(
                    text("DELETE FROM tenants WHERE id = :t"), {"t": existing_tenant}
                )
                await session.commit()

            tenant_id = uuid.uuid4()
            # `users.email` 没有唯一约束，不能用 ON CONFLICT —— 先查后插。
            # （email 不是登录凭据，登录靠 identity_links 的 (issuer, subject)，
            #   所以建表时没给 email 加唯一约束是合理的。）
            row = await session.execute(
                text("SELECT id FROM users WHERE email = 'eval@eval.local' LIMIT 1")
            )
            existing_user = row.scalar_one_or_none()
            if existing_user is not None:
                user_id = existing_user
            else:
                user_id = uuid.uuid4()
                await session.execute(
                    text(
                        "INSERT INTO users (id, email, display_name, status, platform_role, "
                        "created_at, updated_at) VALUES (:id, 'eval@eval.local', 'eval', "
                        "'active', 'none', now(), now())"
                    ),
                    {"id": user_id},
                )
            await session.execute(
                text(
                    "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                    "VALUES (:id, :s, 'eval', 'active', now(), now())"
                ),
                {"id": tenant_id, "s": EVAL_TENANT_SLUG},
            )
            await session.commit()

        # 建知识库需要租户上下文（RLS 要求新行的 tenant_id 与上下文一致）
        async with db.transaction(tenant_id=tenant_id) as session:
            kb = await KnowledgeBaseRepository(session).create(
                tenant_id=tenant_id,
                name=EVAL_KB_NAME,
                created_by=user_id,
                chunking_strategy="recursive",
                chunk_size=500,
                chunk_overlap=80,
            )
            kb_id = kb.id

        # 解析 + 切分 + 向量化（走真实的 adapters 代码路径）
        await _ingest_corpus(db, tenant_id=tenant_id, kb_id=kb_id, user_id=user_id)
        return tenant_id, kb_id
    finally:
        await db.close()


async def _ingest_corpus(
    db: object, *, tenant_id: uuid.UUID, kb_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """逐文档：解析 → 切分 → 向量化 → 落库。

    直接调 adapters 与仓储，不走 worker（评估脚本不该依赖任务队列）。
    但**解析器/分块器/embedding 用的是同一套实现**，所以测的是真实链路。
    """
    from rag_adapters.chunkers.base import ChunkConfig, verify_chunks
    from rag_adapters.chunkers.factory import build_chunker
    from rag_adapters.parsers.artifact import BlockRecord, PageRecord, dump_parsed, load_parsed
    from rag_adapters.parsers.factory import parse_document
    from rag_adapters.providers.factory import build_embedding_provider
    from rag_contracts.enums import DocumentFormat
    from rag_db.repositories import BlobRepository, DocumentRepository
    from rag_db.repositories.chunk import ChunkRepository, IndexVersionRepository

    provider = build_embedding_provider(get_settings())
    documents = sorted(CORPUS_DIR.glob("*.md"))
    if not documents:
        raise SystemExit(f"corpus 为空：{CORPUS_DIR}")

    print(f"建库：{len(documents)} 个文档")
    try:
        for doc_path in documents:
            content = doc_path.read_text(encoding="utf-8")
            # 落成 blob（用本地存储）
            parsed_doc = parse_document(doc_path, format=DocumentFormat.MARKDOWN)
            artifact = dump_parsed(
                text=parsed_doc.text,
                blocks=[
                    BlockRecord(
                        text=b.text,
                        block_type=b.block_type,
                        char_start=b.char_start,
                        char_end=b.char_end,
                        page_no=b.page_no,
                        bbox=b.bbox,
                        level=b.level,
                    )
                    for b in parsed_doc.blocks
                ],
                pages=[
                    PageRecord(
                        page_no=p.page_no,
                        width=p.width,
                        height=p.height,
                        char_start=p.char_start,
                        char_end=p.char_end,
                        is_scanned=p.is_scanned,
                    )
                    for p in parsed_doc.pages
                ],
                parser_name=parsed_doc.parser_name,
                document_format=parsed_doc.format.value,
                has_scanned_pages=parsed_doc.has_scanned_pages,
            )
            artifact_parsed = load_parsed(artifact)

            chunker = build_chunker(
                "recursive",
                config=ChunkConfig(chunk_size=500, chunk_overlap=80, min_chunk_size=50),
            )
            drafts = chunker.chunk(artifact_parsed)
            problems = verify_chunks(drafts, artifact_parsed)
            if problems:
                raise SystemExit(f"{doc_path.name} 切分违反契约：{problems[0]}")

            # 生成向量
            texts = [d.embedding_input or d.content for d in drafts]
            result = await provider.embed_batch(texts)

            async with db.transaction(tenant_id=tenant_id) as session:  # type: ignore[attr-defined]
                import hashlib

                blob, _ = await BlobRepository(session).get_or_create(
                    tenant_id=tenant_id,
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    size_bytes=len(content.encode()),
                    storage_key=f"eval/{doc_path.name}",
                    storage_backend="local",
                    content_type="text/markdown",
                )
                doc = await DocumentRepository(session).create(
                    tenant_id=tenant_id,
                    knowledge_base_id=kb_id,
                    title=doc_path.name,
                    original_filename=doc_path.name,
                    format=DocumentFormat.MARKDOWN.value,
                    created_by=user_id,
                )
                repo = DocumentRepository(session)
                version_no = await repo.allocate_version_no(doc_id=doc.id)
                version = await repo.create_version(
                    tenant_id=tenant_id,
                    document_id=doc.id,
                    version_no=version_no,
                    blob_id=blob.id,
                    created_by=user_id,
                    chunking_strategy="recursive",
                    chunking_config={
                        "chunk_size": 500,
                        "chunk_overlap": 80,
                        "min_chunk_size": 50,
                    },
                )

                iv_repo = IndexVersionRepository(session)
                active = await iv_repo.get_active(knowledge_base_id=kb_id)
                if active is None:
                    active = await iv_repo.create(
                        tenant_id=tenant_id,
                        knowledge_base_id=kb_id,
                        version_tag=f"{provider.name}-eval",
                        embedding_model=provider.name,
                        embedding_dim=provider.capabilities.dimension,
                    )
                    await iv_repo.activate(knowledge_base_id=kb_id, version_id=active.id)
                    await session.flush()

                await ChunkRepository(session).bulk_create(
                    tenant_id=tenant_id,
                    knowledge_base_id=kb_id,
                    document_id=doc.id,
                    document_version_id=version.id,
                    chunks=[
                        {
                            "chunk_index": draft.chunk_index,
                            "content": draft.content,
                            "content_hash": draft.content_hash,
                            "chunk_type": draft.chunk_type,
                            "embedding_input": draft.embedding_input,
                            "chunk_level": draft.chunk_level,
                            "page_no": draft.page_no,
                            "char_start": draft.char_start,
                            "char_end": draft.char_end,
                            "bbox": draft.bbox,
                            "section_path": draft.section_path,
                            "chunk_metadata": draft.chunk_metadata,
                            "embedding": vector,
                            "embedding_model": provider.name,
                        }
                        for draft, vector in zip(drafts, result.vectors, strict=True)
                    ],
                    index_version_id=active.id,
                )

            print(f"  {doc_path.name}: {len(drafts)} chunk")
    finally:
        await provider.aclose()


def get_settings() -> object:
    from rag_core.settings import get_settings as _get

    return _get()


# ---------------------------------------------------------------- 跑评估


async def _resolve_eval_ids(db: object) -> tuple[uuid.UUID, uuid.UUID]:
    """从库中解析评估租户与知识库 ID。

    用一次 admin 连接（`transaction_bypass_rls`）按 slug 精确查单条，
    因为此刻还不知道 tenant_id（正是要查的东西）。
    这里只做**一次只读查询**，随后所有检索都在
    `transaction(tenant_id=...)` 里走 RLS——不是绕过隔离来跑业务查询。
    """
    async with db.transaction_bypass_rls() as session:  # type: ignore[attr-defined]
        row = await session.execute(
            text("SELECT id FROM tenants WHERE slug = :s"), {"s": EVAL_TENANT_SLUG}
        )
        tenant_id = row.scalar_one_or_none()
        if tenant_id is None:
            raise SystemExit("评估库未建，先跑 --setup")
        row = await session.execute(
            text("SELECT id FROM knowledge_bases WHERE tenant_id = :t AND name = :n"),
            {"t": tenant_id, "n": EVAL_KB_NAME},
        )
        kb_id = row.scalar_one_or_none()
        if kb_id is None:
            raise SystemExit("评估库未建，请先跑 --setup")
        return tenant_id, kb_id


@dataclass
class GraphFlags:
    """本次评估启用哪些检索节点。

    每个 flag 对应 `RetrievalConfig` 的一个开关。**分开传而不是传整份
    config**：命令行只暴露"加哪个节点"，避免一次改多个导致指标
    变化无法归因。
    """

    keyword: bool = False
    rerank: bool = False
    rewrite: bool = False


async def run_eval(*, label: str, k: int, flags: GraphFlags | None = None) -> Report:
    from rag_adapters.providers.embedding import EmbeddingTask
    from rag_adapters.providers.factory import build_embedding_provider
    from rag_core.db import Database
    from rag_core.settings import load_env_files
    from rag_db.repositories.chunk import ChunkRepository, IndexVersionRepository

    load_env_files()
    settings = get_settings()
    db = Database(settings)  # type: ignore[arg-type]
    provider = build_embedding_provider(settings)  # type: ignore[arg-type]

    cases = load_cases()
    results: list[CaseResult] = []

    try:
        # 取评估库。
        #
        # `tenants` 与 `knowledge_bases` 都受 RLS 约束，而此刻还不知道
        # tenant_id（正是要查的东西）。所以先用一次 admin 连接把 ID 读出来——
        # 这是合法的跨租户读：按 slug 精确匹配单条记录，且后续所有检索
        # 都在 `transaction(tenant_id=...)` 里走 RLS。
        tenant_id, kb_id = await _resolve_eval_ids(db)

        started = time.perf_counter()
        for case in cases:
            case_started = time.perf_counter()

            # 每次查询都新开一个租户事务（RLS 生效）。
            # 检索器闭包在事务内创建，确保所有查询都在同一上下文里。
            async with db.transaction(tenant_id=tenant_id) as session:
                active = await IndexVersionRepository(session).get_active(knowledge_base_id=kb_id)
                version_ids = [active.id] if active else None
                repo = ChunkRepository(session)
                query_vector = await provider.embed_one(case.query, task=EmbeddingTask.QUERY)

                vector_results = await repo.search_vector(
                    tenant_id=tenant_id,
                    query_embedding=query_vector,
                    knowledge_base_ids=[kb_id],
                    limit=k,
                    index_version_ids=version_ids,
                )
                retrieved = [(chunk.content, distance) for chunk, distance in vector_results]

                # 启用关键词检索时，走检索图的混合路径（RRF 融合）
                if flags is not None and flags.keyword:
                    fused = await _run_hybrid(
                        repo,
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        version_ids=version_ids,
                        query=case.query,
                        vector_results=vector_results,
                        limit=k,
                    )
                    if fused:
                        retrieved = fused

            latency = (time.perf_counter() - case_started) * 1000

            # 第一个命中位置（1-based）
            first_hit: int | None = None
            for rank, (content, _distance) in enumerate(retrieved, start=1):
                if any(snippet in content for snippet in case.relevant_snippets):
                    first_hit = rank
                    break

            results.append(
                CaseResult(
                    case=case,
                    retrieved=retrieved,
                    first_hit_rank=first_hit,
                    latency_ms=latency,
                )
            )
        elapsed = time.perf_counter() - started

        return Report(label=label, k=k, results=results, elapsed_s=elapsed)
    finally:
        await provider.aclose()
        await db.close()


# ---------------------------------------------------------------- 报告


async def _run_hybrid(
    repo: object,
    *,
    tenant_id: uuid.UUID,
    kb_id: uuid.UUID,
    version_ids: list[uuid.UUID] | None,
    query: str,
    vector_results: list[tuple[object, float]],
    limit: int,
) -> list[tuple[str, float]]:
    """混合检索：向量 + 关键词 → RRF 融合。

    **在评估里用仓储直接调，而不是装配完整 `RetrievalGraph`**：
    图的检索器是 Protocol，评估脚本里包一层闭包即可，不必为它
    造一套依赖注入。等图有多个条件分支（循环/自省）时再切换。

    关键词检索用纯关键词提取（去掉疑问词与停用词）——
    真实场景该交给分词器，但评估要的是"关键词路是否带来增益"这个
    结论，用最简单的实现即可（如果简单实现都有增益，复杂实现只会更好）。
    """
    from rag_adapters.retrieval.fusion import fuse_two

    keywords = _extract_keywords(query)
    if not keywords:
        return [(str(chunk.content), distance) for chunk, distance in vector_results]  # type: ignore[attr-defined]

    keyword_hits = await repo.search_keyword(  # type: ignore[attr-defined]
        tenant_id=tenant_id, keywords=keywords, knowledge_base_ids=[kb_id], limit=limit
    )
    vector_ids = [str(chunk.id) for chunk, _ in vector_results]
    keyword_ids = [str(chunk.id) for chunk, _ in keyword_hits]

    content_by_id = {str(chunk.id): chunk.content for chunk, _ in vector_results}
    content_by_id.update({str(chunk.id): chunk.content for chunk, _ in keyword_hits})
    distance_by_id = {str(chunk.id): distance for chunk, distance in vector_results}

    fused = fuse_two(vector_ids, keyword_ids)
    return [
        (content_by_id[item.item_id], distance_by_id.get(item.item_id, 0.0))
        for item in fused[:limit]
        if item.item_id in content_by_id
    ]


# 中文疑问词与常见停用词。去掉它们再做关键词检索，
# 否则 "什么"/"多少"/"怎么" 这些词会在所有文档里出现，污染 ts_rank。
_QUESTION_WORDS = (
    "多少",
    "什么",
    "怎么",
    "怎样",
    "如何",
    "哪些",
    "哪个",
    "可以",
    "能否",
    "是否",
    "需要",
    "时候",
    "多久",
    "几天",
    "吗",
    "呢",
    "的",
    "了",
    "有",
)


def _extract_keywords(query: str) -> list[str]:
    """从中文查询里提取关键词。

    **这不是好的分词器**——真正的中文分词需要 jieba / pg_bigm / zhparser。
    A10 已记录待办：当前 `search_keyword` 用的 `simple` 分词器不认中文。

    这里的近似做法：
    1. 去掉疑问词与停用词（否则"什么""多少"在所有文档里都出现，
       会让 ts_rank 全是噪声）；
    2. 按标点切成短语；
    3. 每个短语取 2~3 字滑窗（中文里 2 字词覆盖大部分实词）。

    **限定关键词数量上限（8 个）**：关键词之间是 OR 关系，
    词越多 OR 越宽、命中越多、排序越没意义，且 SQL 构造越慢。
    实测 12+ 个关键词会把延迟从 6ms 推到 650ms 而毫无收益。
    """
    cleaned = query
    for word in _QUESTION_WORDS:
        cleaned = cleaned.replace(word, " ")
    # 标点是天然短语边界
    for punct in "，。？！、；：,.?!;:":
        cleaned = cleaned.replace(punct, " ")

    phrases = [p for p in cleaned.split() if len(p) >= 2]

    pieces: list[str] = []
    for phrase in phrases:
        # 长短语整体也算一个关键词（命中时 ts_rank 权重更高）
        if len(phrase) <= 6:
            pieces.append(phrase)
        for size in (3, 2):
            for start in range(0, max(1, len(phrase) - size + 1)):
                piece = phrase[start : start + size]
                if len(piece) == size and piece not in pieces:
                    pieces.append(piece)

    # 去重并限制数量（少而精优于多而杂）
    return list(dict.fromkeys(pieces))[:8]


def print_report(report: Report) -> None:
    print(f"\n{'=' * 62}")
    print(f"检索评估报告 · {report.label}  (k={report.k})")
    print(f"{'=' * 62}")
    print(f"  Recall@{report.k}          : {report.recall:.1%}")
    print(f"  MRR@{report.k}             : {report.mrr:.3f}")
    print(f"  Hit@{report.k}             : {report.hit_rate:.1%}")
    print(f"  误报率（unanswerable）   : {report.false_positive_rate:.1%}")
    print(f"  平均检索延迟             : {report.avg_latency_ms:.1f}ms")
    print(f"  总耗时                   : {report.elapsed_s:.1f}s")

    # 分类别表现：能看出"哪类查询是弱项"
    by_category: dict[str, list[CaseResult]] = {}
    for result in report.results:
        by_category.setdefault(result.case.category, []).append(result)

    print("\n  按类别：")
    print(f"    {'类别':<16} {'条数':>4} {'命中':>6} {'MRR':>6}")
    for category, items in sorted(by_category.items()):
        answerable = [r for r in items if r.case.relevant_snippets]
        if answerable:
            hit = sum(1 for r in answerable if r.first_hit_rank) / len(answerable)
            mrr = sum(1.0 / r.first_hit_rank for r in answerable if r.first_hit_rank) / len(
                answerable
            )
            print(f"    {category:<16} {len(items):>4} {hit:>5.0%} {mrr:>6.3f}")
        else:
            print(f"    {category:<16} {len(items):>4} {'—':>6} {'—':>6}")

    # 未命中的用例：最有诊断价值
    misses = [r for r in report.results if r.case.relevant_snippets and not r.first_hit_rank]
    if misses:
        print(f"\n  未命中（{len(misses)} 条）：")
        for result in misses:
            print(f"    {result.case.id} [{result.case.category}] {result.case.query}")
            if result.retrieved:
                print(f"      最接近：{result.retrieved[0][0][:60]!r}")

    print(f"\n  {'=' * 60}")


def save_report(report: Report) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{report.timestamp}-{report.label}.json"
    payload = {
        "label": report.label,
        "k": report.k,
        "timestamp": report.timestamp,
        "elapsed_s": report.elapsed_s,
        "metrics": {
            "recall": report.recall,
            "mrr": report.mrr,
            "hit_rate": report.hit_rate,
            "false_positive_rate": report.false_positive_rate,
            "avg_latency_ms": report.avg_latency_ms,
        },
        "cases": [
            {
                "id": result.case.id,
                "query": result.case.query,
                "category": result.case.category,
                "first_hit_rank": result.first_hit_rank,
                "latency_ms": result.latency_ms,
                "retrieved_count": len(result.retrieved),
            }
            for result in report.results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def compare_with_previous(report: Report) -> None:
    """与上一次报告对比（同 k）。

    这是 A11 的核心工作方式：每加一个节点跑一次，看指标是涨还是跌。
    指标**下跌**同样重要——有些"优化"（如 aggressive 的 rerank）
    会伤害召回，必须能立刻发现。
    """
    reports = sorted(REPORTS_DIR.glob("*.json"))
    if len(reports) < 2:
        print("\n（无历史报告可对比）")
        return

    # 找与当前 label 不同的最近一份
    previous: dict[str, object] | None = None
    for path in reversed(reports):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("label") != report.label and data.get("k") == report.k:
            previous = data
            break
    if previous is None:
        print("\n（无同 k 的历史报告可对比）")
        return

    prev_metrics = previous["metrics"]
    assert isinstance(prev_metrics, dict)
    print(f"\n  对比上一次（{previous['label']}）：")
    pairs = (
        ("Recall", "recall", True),
        ("MRR", "mrr", True),
        ("Hit", "hit_rate", True),
        ("误报率", "false_positive_rate", False),
        ("延迟(ms)", "avg_latency_ms", False),
    )
    for label, key, higher_is_better in pairs:
        old = float(prev_metrics[key])  # type: ignore[index]
        new = {
            "recall": report.recall,
            "mrr": report.mrr,
            "hit_rate": report.hit_rate,
            "false_positive_rate": report.false_positive_rate,
            "avg_latency_ms": report.avg_latency_ms,
        }[key]
        delta = new - old
        if abs(delta) < 1e-9:
            mark = "="
        elif (delta > 0) == higher_is_better:
            mark = "↑ 改善"
        else:
            mark = "↓ 变差"
        if key == "avg_latency_ms":
            print(f"    {label:<12} {old:>8.1f} → {new:>8.1f}  {mark}")
        elif key == "mrr":
            print(f"    {label:<12} {old:>8.3f} → {new:>8.3f}  {mark}")
        else:
            print(f"    {label:<12} {old:>7.1%} → {new:>7.1%}  {mark}")


def list_reports() -> None:
    reports = sorted(REPORTS_DIR.glob("*.json"))
    if not reports:
        print("暂无报告")
        return
    print(f"{'时间':<22} {'label':<20} {'Recall':>8} {'MRR':>8} {'Hit':>8}")
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data["metrics"]
        print(
            f"{data['timestamp']:<22} {data['label']:<20} "
            f"{metrics['recall']:>7.1%} {metrics['mrr']:>8.3f} {metrics['hit_rate']:>7.1%}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索质量评估")
    parser.add_argument("--setup", action="store_true", help="建库（解析+切分+向量化）")
    parser.add_argument("--force", action="store_true", help="配合 --setup 重建")
    parser.add_argument("--label", default="baseline", help="本次评估的标签")
    parser.add_argument("--k", type=int, default=10, help="检索返回条数")
    parser.add_argument("--list", action="store_true", help="列出历史报告")
    parser.add_argument("--quiet", action="store_true", help="不打印明细，只存报告")
    parser.add_argument("--keyword", action="store_true", help="启用混合检索（RRF 融合）")
    parser.add_argument("--rerank", action="store_true", help="启用重排")
    parser.add_argument("--rewrite", action="store_true", help="启用查询改写")
    args = parser.parse_args(argv)

    if args.list:
        list_reports()
        return 0

    if args.setup:
        asyncio.run(setup_corpus(force=args.force))

    flags = GraphFlags(keyword=args.keyword, rerank=args.rerank, rewrite=args.rewrite)
    report = asyncio.run(run_eval(label=args.label, k=args.k, flags=flags))
    if not args.quiet:
        print_report(report)
    path = save_report(report)
    print(f"\n报告已保存：{path}")
    compare_with_previous(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
