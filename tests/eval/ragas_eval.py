"""Ragas 评估：生成质量（忠实度 / 答案相关性）。

## 与 A11 的检索评估是**互补**关系

| 层次 | 指标 | 工具 |
|---|---|---|
| 检索质量 | Recall@k / MRR / Hit | `run_eval.py`（A11） |
| 生成质量 | Faithfulness / Answer Relevancy | 本脚本（Ragas） |

**不能只看生成指标**：如果检索没召回正确材料，再好的模型也只能
"基于错误材料给出忠实但错误的答案"（faithfulness 高、实际错）。
所以要分开测、分别归因。

## Faithfulness 是什么

答案里的每个事实性陈述能否从给定上下文中推出。
低分 = 模型在编造（幻觉）。

**注意它不判断"上下文本身对不对"**：如果检索给的是错误材料，
模型忠实于那份材料的回答仍得高分。这是 faithfulness 的固有局限，
不是 bug。

## Answer Relevancy 是什么

答案与问题的相关程度（绕圈子、答非所问会低分）。

**两者可能同时低**：模型既编造又跑题——那通常是 prompt 的问题。

## 为什么用 LLM 做评判（而不是规则）

"这个答案是否忠实于材料"无法用规则判断（需要语义理解）。
Ragas 的做法是让 LLM 逐条拆解陈述再验证。

代价：每次评估要多次 LLM 调用（一条样本约 3~5 次）。
所以：
- 默认只跑 10 条（不是 30 条）；
- 结果**只用于回归对比**（看趋势），不追求绝对数值精确。

## 诚实说明：Ragas 的分数不能跨数据集比较

它的分数依赖评判模型（同一份答案，用 DeepSeek 和 GPT-4 评判
分数不同）。所以：
- **只做"同一次改动前后"的对比**；
- 不写"我们的 faithfulness 是 0.87，行业标杆是 0.9"这类话——
  那个数字换评判模型就变。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).parent
REPORTS_DIR = EVAL_DIR / "reports"


@dataclass
class RagasSample:
    """一条评估样本。"""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


@dataclass
class RagasResult:
    """Ragas 评估结果。"""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    sample_count: int = 0
    samples: list[RagasSample] = field(default_factory=list)
    note: str = ""


async def generate_samples(*, count: int) -> list[RagasSample]:
    """跑问答链路，收集 (问题, 答案, 上下文)。

    走真实链路（检索 → 生成），这样评估的是**实际系统**而不是
    单独测模型。
    """

    from rag_adapters.providers.embedding import EmbeddingTask
    from rag_adapters.providers.factory import build_embedding_provider, build_llm_provider
    from rag_adapters.providers.llm import LLMMessage, LLMRole
    from rag_adapters.retrieval.graph import build_generation_prompt
    from rag_core.db import Database
    from rag_core.settings import get_settings, load_env_files
    from rag_db.repositories.chunk import ChunkRepository, IndexVersionRepository

    sys.path.insert(0, str(EVAL_DIR))
    from run_eval import _resolve_eval_ids, load_cases

    load_env_files()
    settings = get_settings()
    db = Database(settings)
    embedder = build_embedding_provider(settings)
    llm = build_llm_provider(settings)

    samples: list[RagasSample] = []
    try:
        tenant_id, kb_id = await _resolve_eval_ids(db)
        # 只取有 ground truth（relevant_snippets）的用例
        cases = [c for c in load_cases() if c.relevant_snippets][:count]

        for case in cases:
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

            contexts = [chunk.content for chunk, _ in hits]
            if not contexts:
                continue

            from rag_adapters.retrieval.state import RetrievedChunk

            materials = [
                RetrievedChunk(chunk_id=str(c.id), content=c.content, score=d) for c, d in hits
            ]
            prompt = build_generation_prompt(query=case.query, materials=materials)
            response = await llm.complete(
                [LLMMessage(role=LLMRole.USER, content=prompt)], temperature=0.1
            )
            answer = response.content.strip()
            if not answer:
                continue

            samples.append(
                RagasSample(
                    question=case.query,
                    answer=answer,
                    contexts=contexts,
                    # 用标注的原文片段作为 ground truth（不是"标准答案"，
                    # 但足以让 Ragas 判断答案是否有依据）
                    ground_truth=" ".join(case.relevant_snippets),
                )
            )
            print(f"  已收集 {len(samples)}/{len(cases)}: {case.query[:30]}")
    finally:
        for closer in (getattr(llm, "aclose", None), getattr(embedder, "aclose", None)):
            if closer is not None:
                await closer()
        await db.close()

    return samples


def evaluate_with_ragas(samples: list[RagasSample]) -> RagasResult:
    """用 Ragas 算指标。

    ## 逐条 try/except

    Ragas 的每个指标都要调 LLM，可能因限流/超时失败。
    一条失败不该让整个评估崩——记录后跳过，并在 `note` 里写明
    "几条样本评估失败"。**静默跳过是不行的**：那会让分数只反映
    "成功的样本"，看起来比实际好。
    """
    if not samples:
        return RagasResult(note="没有可评估的样本")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness
    except ImportError as exc:
        return RagasResult(note=f"Ragas 不可用（{exc}）。跳过生成质量评估。")

    # ---- 接入项目的模型（而不是 Ragas 默认的 OpenAI）----
    #
    # Ragas 默认用 OpenAI 做评判。本项目用 DeepSeek + 本地 bge-m3，
    # 必须显式传入，否则会：① 需要 OpenAI key ② 用英文模型评判中文答案。
    #
    # **判分模型与被测模型是同一个**（DeepSeek 评 DeepSeek）会引入
    # 自我偏好（模型倾向于给自己风格的答案高分）。这一点无法在当前
    # 环境消除（只有一家供应商），必须在报告里注明。
    try:
        from langchain_core.embeddings import Embeddings
        from langchain_openai import ChatOpenAI
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        from rag_adapters.providers.factory import build_embedding_provider
        from rag_core.settings import get_settings, load_env_files

        load_env_files()
        settings = get_settings()

        judge_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.llm.chat_model,
                # `api_key` 是 str（不是 SecretStr）：langchain-openai 接受 str
                api_key=settings.llm.api_key,
                base_url=settings.llm.base_url,
                temperature=0.0,
                # **必须 max_retries=0**：与生产一致（AGENTS.md 4.4），
                # 否则 Ragas 的重试会与内层重试相乘，评估会卡在重试上。
                max_retries=0,
            )
        )

        class _ProjectEmbeddings(Embeddings):
            """把项目的本地 bge-m3 包成 LangChain Embeddings 接口。

            为什么不直接用 `OpenAIEmbeddings`：本地 bge-m3 是
            sentence-transformers 模型，没有 OpenAI 兼容端点。
            硬套 `OpenAIEmbeddings` 会去连一个不存在的服务。

            **复用生产同一个 provider**：评判用的向量化必须与生产一致，
            否则 Answer Relevancy 的语义空间与检索链路不同，分数无意义。
            """

            def __init__(self) -> None:
                self._provider = build_embedding_provider(settings)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self._embed_sync(texts)

            def embed_query(self, text: str) -> list[float]:
                return self._embed_sync([text])[0]

            def _embed_sync(self, texts: list[str]) -> list[list[float]]:
                # LangChain 的同步接口要求同步返回，而 provider 是 async。
                # 用独立事件循环跑（Ragas 内部用线程池调这些方法，
                # 所以不能直接复用外层 loop）。
                import asyncio as _asyncio

                async def _run() -> list[list[float]]:
                    result = await self._provider.embed_batch(texts)
                    return result.vectors

                return _asyncio.run(_run())

        judge_embeddings = LangchainEmbeddingsWrapper(_ProjectEmbeddings())
    except Exception as exc:
        return RagasResult(note=f"评判模型配置失败（{type(exc).__name__}: {exc}）。")

    data = {
        "user_input": [s.question for s in samples],
        "response": [s.answer for s in samples],
        "retrieved_contexts": [s.contexts for s in samples],
        # ground_truth 仅供参考指标（context_recall），本题不用
        "reference": [s.ground_truth for s in samples],
    }

    try:
        dataset = Dataset.from_dict(data)
        # ## 只跑 faithfulness（answer_relevancy 在当前供应商上不可用）
        #
        # Answer Relevancy 的实现要求一次生成多个答案变体
        # （`n > 1`），而 DeepSeek 明确拒绝：
        #
        #   400 Invalid n value (currently only n = 1 is supported)
        #
        # 这不是配置问题，是供应商能力限制。**不伪造这个指标**——
        # 用 n=1 硬跑会得到无意义的结果。
        #
        # Faithfulness 不需要 n>1（它是"拆解陈述 → 逐条验证"），
        # 所以单独跑它。
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=judge_llm,
            embeddings=judge_embeddings,
            # raise_exceptions=False：单条样本失败返回 NaN 而不是整体崩。
            raise_exceptions=False,
        )
        # **不要用 `dict(result)`**：Ragas 0.4 的 `EvaluationResult`
        # 的 `__getitem__` 期望字符串键，`dict()` 会按整数索引遍历
        # 从而抛 `KeyError: 0`（实测）。
        #
        # 它的 `str()` 就是 `{'faithfulness': 1.0000}` 这样的 dict 字面量
        # ——不是可靠接口，但 `to_pandas()` 更重（要 pandas 依赖）。
        # 这里直接读属性：`result["faithfulness"]` 用字符串键是官方用法。
        faith_score = None
        try:
            raw = result["faithfulness"]
            faith_score = _first_score(raw)
        except Exception as exc:
            log_msg = f"读取 faithfulness 失败（{type(exc).__name__}: {exc}）"
            return RagasResult(
                sample_count=len(samples),
                samples=samples,
                note=log_msg,
            )

        return RagasResult(
            faithfulness=faith_score,
            answer_relevancy=None,
            sample_count=len(samples),
            samples=samples,
            note=(
                "Answer Relevancy 跳过：DeepSeek 不支持 n>1，而该指标需要多采样。"
                "Faithfulness 不受影响。"
                "判分模型与被测模型相同（DeepSeek），存在自我偏好偏差；"
                "分数只用于同一配置的前后对比，不跨数据集比较。"
            ),
        )
    except Exception as exc:
        # 常见原因：评判模型不可用、配额耗尽、Ragas 与 LLM 接口不兼容
        return RagasResult(
            sample_count=len(samples),
            samples=samples,
            note=(
                f"Ragas 评估失败：{type(exc).__name__}: {str(exc)[:200]}。"
                f"样本已收集，可换评判模型重试。"
            ),
        )


def _first_score(value: object) -> float | None:
    """从 Ragas 的返回值里取出数值分数。

    Ragas 0.4 的 `dict(result)` 有时给标量、有时给 list（多指标时），
    统一成 float。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        first = value[0]
        return float(first) if isinstance(first, (int, float)) else None
    return None


def compare_with_previous(result: RagasResult, *, label: str) -> dict[str, float | None]:
    """与上次结果对比。"""
    previous: dict[str, object] | None = None
    for path in sorted(REPORTS_DIR.glob("ragas-*.json")):
        # 排除样本文件（`ragas-samples-*.json` 是 list，没有 label 字段）
        if "samples" in path.name:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        if data.get("label") != label:
            previous = data
            break

    if previous is None:
        return {}

    current = {
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
    }
    print(f"\n  对比上一次（{previous.get('label')}）：")
    for key, value in current.items():
        old = previous.get(key)  # type: ignore[union-attr]
        if old is None or value is None:
            print(f"    {key:<20} —")
            continue
        delta = value - float(old)  # type: ignore[arg-type]
        mark = "=" if abs(delta) < 0.01 else ("↑" if delta > 0 else "↓")
        print(f"    {key:<20} {float(old):.3f} → {value:.3f}  {mark} ({delta:+.3f})")
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ragas 生成质量评估")
    parser.add_argument("--count", type=int, default=10, help="样本数（默认 10）")
    parser.add_argument("--label", default="baseline", help="本次标签")
    parser.add_argument("--dry-run", action="store_true", help="只收集样本不评估")
    args = parser.parse_args(argv)

    print(f"收集 {args.count} 条样本（走真实问答链路）...")
    samples = asyncio.run(generate_samples(count=args.count))
    print(f"\n共收集 {len(samples)} 条样本")

    if args.dry_run:
        out = REPORTS_DIR / f"ragas-samples-{args.label}.json"
        REPORTS_DIR.mkdir(exist_ok=True)
        out.write_text(
            json.dumps(
                [
                    {
                        "question": s.question,
                        "answer": s.answer,
                        "contexts": s.contexts,
                    }
                    for s in samples
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"样本已保存：{out}")
        return 0

    print("\n调用 Ragas 评估（每条样本约 3~5 次 LLM 调用）...")
    result = evaluate_with_ragas(samples)

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"ragas-{args.label}.json"
    out.write_text(
        json.dumps(
            {
                "label": args.label,
                "faithfulness": result.faithfulness,
                "answer_relevancy": result.answer_relevancy,
                "sample_count": result.sample_count,
                "note": result.note,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{'=' * 50}")
    print(f"Ragas 结果 · {args.label}")
    print(f"{'=' * 50}")
    print(f"  样本数          : {result.sample_count}")
    faith = result.faithfulness if result.faithfulness is not None else "—"
    relev = result.answer_relevancy if result.answer_relevancy is not None else "—"
    print(f"  Faithfulness    : {faith}")
    print(f"  Answer Relevancy: {relev}")
    if result.note:
        print(f"  说明            : {result.note}")
    compare_with_previous(result, label=args.label)
    print(f"\n报告已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
