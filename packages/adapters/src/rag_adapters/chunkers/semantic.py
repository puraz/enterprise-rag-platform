"""语义切分：按相邻句子/段落的向量相似度突变点切。

## 与其他策略的根本区别

其他策略靠**形式特征**（长度、标点、标题）。语义切分靠**内容含义**：
相邻段落讲的是同一件事就留在一起，话题变了就切。

典型收益场景：一份没有标题的会议纪要 / 访谈记录 / 长段落散文——
没有标题可依（structured 失效），句子长度均匀（recursive 切得"均匀"但切点随意），
而话题切换处能被向量相似度捕捉到。

## 代价：切分需要 embedding

这是唯一需要调用模型的切分策略。后果：

- **慢**：n 个段落要算 n 个向量；
- **花钱**（若用远程 embedding）；
- **引入依赖**：切分不再是纯函数，同样的输入在不同模型下结果不同。

所以：
1. 语义切分**不是默认策略**（默认是 recursive）；
2. embedding 通过**注入**得到，chunker 不 import 任何模型实现；
3. 失败时**降级**为 recursive 而不是让任务失败（切分退化总比整篇处理失败好）。

## 算法

    1. 把文本切成句子（复用 sentence_window 的切分逻辑）；
    2. 相邻句子两两合并成"候选单元"（避免句子太短导致向量噪声大）；
    3. 计算相邻单元的余弦相似度；
    4. 找相似度**低谷**（低于均值 - k×标准差，或低于绝对阈值）作为切点；
    5. 按切点聚合单元，受 chunk_size 上限约束。

为什么用"相邻相似度低谷"而不是"聚类"：
聚类需要预设簇数且对长度不均的文本效果差；低谷检测是**局部**判断，
更贴合"话题在哪一段切换"这个实际问题，且能自然产生不等长的块。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Awaitable, Callable, Sequence

from rag_adapters.chunkers.base import BaseChunker, ChunkDraft
from rag_adapters.chunkers.recursive import RecursiveChunker
from rag_adapters.parsers.artifact import ParsedArtifact
from rag_core.logging import get_logger

# embedding 函数签名：一批文本 → 一批等长向量
EmbedFn = Callable[[Sequence[str]], Awaitable[list[list[float]]]]

# 切点判定：相似度低于 (均值 - K × 标准差) 视为话题切换
# K=1.0 是经验值：太大则切点过少（块超长），太小则切得过碎。
_BREAKPOINT_K = 1.0

# 绝对下限保护：相似度本身已经很高（如全部 > 0.9）时，
# 相对判定会把"最像里的不那么像"当成切点，产生无意义切分。
_MIN_ABSOLUTE_DROP = 0.05

log = get_logger(__name__)


class SemanticChunker(BaseChunker):
    """语义切分。

    必须注入 `embed_fn`；未注入时 `chunk()` 直接降级为 recursive。
    """

    name = "semantic"

    def __init__(
        self,
        config: object | None = None,
        *,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        # config 的类型在父类里是 ChunkConfig；这里保持宽松以便测试注入
        super().__init__(config)  # type: ignore[arg-type]
        self._embed_fn = embed_fn

    def chunk(self, parsed: ParsedArtifact) -> list[ChunkDraft]:
        if not parsed.text.strip():
            return []

        if self._embed_fn is None:
            # 降级：没有 embedding 能力时用递归切分。
            # 不抛异常——整篇文档因为"策略选错"而处理失败太粗暴。
            return RecursiveChunker(self.config).chunk(parsed)

        # 语义切分需要异步（要算向量），但 BaseChunker.chunk 是同步接口。
        # 这个矛盾在 handler 里解决：handler 调用 `chunk_async`。
        # 同步路径降级为 recursive，保证接口契约仍然成立。
        return RecursiveChunker(self.config).chunk(parsed)

    async def chunk_async(self, parsed: ParsedArtifact) -> list[ChunkDraft]:
        """异步入口（handler 走这个）。

        ## 降级必须可观测

        三种情况会退化为 recursive：没有 embed_fn、单元太少、
        embedding 调用失败。**每次降级都写日志**。

        不写日志的后果：运维看到 `strategy=semantic` 的文档，
        以为在用语义切分，实际全是递归切分——对比实验的数据全是假的，
        而且没有任何迹象。
        """
        if not parsed.text.strip():
            return []

        if self._embed_fn is None:
            log.info(
                "semantic_chunk_degraded",
                reason="no_embedding_function",
                note="未注入 embed_fn，退化为 recursive",
            )
            return RecursiveChunker(self.config).chunk(parsed)

        units = self._collect_units(parsed)
        if len(units) < 2:
            log.info(
                "semantic_chunk_degraded",
                reason="too_few_units",
                unit_count=len(units),
                note="单元不足 2 个，无法计算相邻相似度",
            )
            return RecursiveChunker(self.config).chunk(parsed)

        try:
            vectors = await self._embed_fn([unit.text for unit in units])
        except Exception as exc:
            # embedding 失败 → 降级。切分不是关键路径，
            # 让整篇文档卡在"切分失败"不合理（重试也一样可能失败）。
            log.warning(
                "semantic_chunk_degraded",
                reason="embedding_failed",
                unit_count=len(units),
                error_type=type(exc).__name__,
                note="退化为 recursive（切分成功但非语义切分）",
            )
            return RecursiveChunker(self.config).chunk(parsed)

        breakpoints = self._detect_breakpoints(vectors)
        log.info(
            "semantic_chunk_completed",
            unit_count=len(units),
            breakpoint_count=len(breakpoints),
        )
        return self._assemble(parsed, units, breakpoints)

    # ------------------------------------------------------------ 单元

    def _collect_units(self, parsed: ParsedArtifact) -> list[_Unit]:
        """把块内的句子按主题粒度聚成"候选单元"。

        直接用句子当单元的问题：太短，向量噪声大，
        相邻句相似度普遍很高（同一话题内），区分度不足。

        用"两两合并"：单元长度约 2 句，既保留局部语义，
        又让相似度差异更明显。
        """
        units: list[_Unit] = []

        for block in parsed.blocks:
            if not block.text.strip():
                continue

            from rag_adapters.chunkers.sentence_window import SentenceWindowChunker

            sentences = SentenceWindowChunker._split_sentences(block)
            if not sentences:
                continue

            buffer_text = ""
            buffer_start = sentences[0][1]
            buffer_count = 0

            for sentence_text, offset, length in sentences:
                buffer_text += sentence_text
                buffer_count += 1

                if buffer_count >= 2:
                    units.append(_Unit(text=buffer_text, start=buffer_start, end=offset + length))
                    buffer_text = ""
                    buffer_count = 0
                    buffer_start = offset + length

            if buffer_text.strip():
                units.append(
                    _Unit(
                        text=buffer_text,
                        start=buffer_start,
                        end=buffer_start + len(buffer_text),
                    )
                )

        return units

    # ------------------------------------------------------------ 切点检测

    @staticmethod
    def _detect_breakpoints(vectors: list[list[float]]) -> set[int]:
        """返回切点位置（在第 i 个单元**之后**切）。

        相似度序列：`sim[i] = cos(vectors[i], vectors[i+1])`
        切点判据：`sim[i] < mean(sim) - K * stdev(sim)`

        另外要求"确实有下降"（`sim[i] < max(sim) - _MIN_ABSOLUTE_DROP`），
        避免在整体都很相似的文本里强行找出切点。
        """
        if len(vectors) < 3:
            return set()

        similarities = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        if not similarities:
            return set()

        mean = statistics.fmean(similarities)
        stdev = statistics.pstdev(similarities) if len(similarities) > 1 else 0.0
        threshold = mean - _BREAKPOINT_K * stdev

        highest = max(similarities)
        breakpoints: set[int] = set()
        for index, similarity in enumerate(similarities):
            if similarity < threshold and similarity < highest - _MIN_ABSOLUTE_DROP:
                breakpoints.add(index)

        return breakpoints

    # ------------------------------------------------------------ 组装

    def _assemble(
        self,
        parsed: ParsedArtifact,
        units: list[_Unit],
        breakpoints: set[int],
    ) -> list[ChunkDraft]:
        """按切点聚合单元，同时受 chunk_size 上限约束。"""
        drafts: list[ChunkDraft] = []
        index = 0
        budget = self.config.chunk_size

        group: list[_Unit] = []

        def flush() -> None:
            nonlocal index
            if not group:
                return
            content = parsed.text[group[0].start : group[-1].end]
            drafts.append(
                self._make_draft(
                    content=content,
                    index=index,
                    char_start=group[0].start,
                    char_end=group[-1].end,
                )
            )
            index += 1
            group.clear()

        for position, unit in enumerate(units):
            group.append(unit)

            hit_breakpoint = position in breakpoints
            over_budget = len(parsed.text[group[0].start : unit.end]) > budget

            if hit_breakpoint or over_budget:
                flush()

        flush()
        return drafts


class _Unit:
    """切分单元（内部使用）。"""

    __slots__ = ("end", "start", "text")

    def __init__(self, *, text: str, start: int, end: int) -> None:
        self.text = text
        self.start = start
        self.end = end


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。

    不假设向量已归一化：调用方可能是任意 embedding 实现。
    归一化向量可以直接点积，但多算两次范数的代价远小于"忘了归一化导致
    相似度全错"的风险。
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = ["EmbedFn", "SemanticChunker"]
