"""分块器抽象：契约 + 公共数据结构 + 位置信息传播。

## 分块是检索质量的源头

向量检索召回不到的内容，后面所有环节（RRF、rerank、LLM）都救不回来。
而分块做错的方式非常隐蔽：

- **切断了句子/表格**：chunk 读起来是半句话，embedding 语义漂移；
- **丢了位置信息**：答案里能看到引用，但点不进去（高亮失效）；
- **父块与子块对不上**：small-to-big 回填时拿到错文档的内容；
- **重叠区域算错**：同一段文字出现在两个 chunk 里，检索结果重复。

所以本层的设计原则是：**凡是能从输入推出来的信息，必须精确传递到输出**，
不允许"大概对"。

## 契约（所有策略必须遵守）

1. **不产生空块**：`content.strip()` 为空的一律丢弃。
2. **不切断受保护类型**：`TABLE` / `CODE` / `FORMULA` 整块保留。
   表格被切开会让行列错位；代码被切开会失去语义。
3. **位置信息成对**：`char_start`/`char_end` 必须同有同无，
   且 `char_end > char_start`（数据库有 CHECK 约束兜底）。
4. **顺序单调**：`chunk_index` 从 0 递增，且 `char_start` 不倒退。
   否则"上一段/下一段"导航会乱跳。
5. **可复现**：同样的输入 + 同样的配置 → 完全相同的输出。
   分块结果散落着随机性的话，出了问题无法复现、无法对比策略优劣。
6. **不越界**：`char_start`/`char_end` 必须落在源文本范围内。
   `verify_chunks()` 会校验这一点。

## 为什么位置信息用"源文本偏移"而不是"chunk 内偏移"

`chunks.char_start` 是**相对于整篇文档纯文本**的偏移。
这样前端拿到引用后，直接在原文档文本上切 `[char_start:char_end]`
就能高亮，不需要知道分块是怎么切的（也不需要重建分块边界）。

代价是分块器必须一直带着一个"当前在源文本中的游标"。
本层用 `_Cursor` 显式管理它，而不是让每个策略自己算——
自己算必然出现某处漏加分隔符长度，导致偏移整体错位。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_adapters.parsers.artifact import BlockRecord, ParsedArtifact

# 整块保留、不参与按长度切分的类型。
# 理由是语义完整性：表格切开行列错位、代码切开失去结构。
PROTECTED_BLOCK_TYPES = frozenset({"table", "code", "formula"})


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """切分参数。

    **必须与 `document_versions.chunking_config` 快照一致**：
    版本上存了这个 config，A10 回滚到旧版本时要用当初的参数，
    而不是用库的当前配置（否则"回滚"会得到完全不同的 chunk）。
    """

    chunk_size: int = 800
    """目标长度（字符数）。

    为什么不按 token：
    - token 计数依赖具体模型的分词器（bge-m3 与 DeepSeek 不同）；
    - 分块发生在向量化之前，此时还不知道会用哪个 embedding 模型；
    - 字符数是稳定、可复现的度量。token 数在入库时单独统计。
    """

    chunk_overlap: int = 100
    """相邻块的重叠字符数。

    为什么需要重叠：一句话被切断在两块之间时，两边都缺半个语义。
    重叠让跨块边界的语义在两块里都完整。

    为什么不能太大：重叠部分会被重复向量化（成本 ×N），
    且检索时同一段内容可能占满多个结果位（挤掉其它相关内容）。
    """

    min_chunk_size: int = 50
    """最小块长度。短于这个的块会被合并到前一块（若能合并）。

    太短的块（如一个孤立的"注："）embedding 信息量极低，
    但会在检索结果里占一个位置。
    """

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须为正")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能为负")
        if self.chunk_overlap >= self.chunk_size:
            # 重叠 >= 块长会让切分无法前进（每步移动量 <= 0）→ 死循环。
            # 数据库 CHECK 约束也拦这个，但这里要更早失败（构造时就报）。
            raise ValueError("chunk_overlap 必须小于 chunk_size，否则切分无法推进")


@dataclass(slots=True)
class ChunkDraft:
    """分块产出（尚未落库）。

    与 `rag_db.models.Chunk` 的字段一一对应，但**不含 id / tenant 等
    数据库关注的东西**：分块器不应该知道租户、知识库、文档 ID——
    它只负责"把这份解析结果切成块"。归属由调用方（handler）填。
    """

    content: str
    chunk_index: int
    chunk_type: str = "text"
    char_start: int | None = None
    char_end: int | None = None
    page_no: int | None = None
    bbox: dict[str, Any] | None = None
    section_path: list[str] = field(default_factory=list)
    embedding_input: str | None = None
    parent_index: int | None = None
    """父块在**本批次结果里的下标**（不是数据库 ID）。

    数据库自引用要等父块先入库拿到 ID 才能填。用下标中转：
    chunker 不知道也不该知道数据库 ID，由 handler 落库后回填。
    """

    chunk_level: int = 0
    chunk_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """内容 SHA-256，用于去重与缓存 key。

        用 content 而不是 embedding_input 算：embedding_input 含上下文前缀
        （如章节路径），同样的正文在不同章节下 hash 会不同——
        那样就失去了"内容去重"的意义。
        """
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Coverage:
    """分块覆盖统计。

    ## 只统计"会被检索的块"

    **父块（`chunk_level > 0`）不计入**：父子策略里父块是子块的超集，
    把父块计入会让重叠率恒为 100%（父块覆盖了自己所有子块），
    覆盖率也恒为 100%。父块不参与检索（不向量化、检索按 level=0 过滤），
    所以"检索覆盖"只看叶子块才对应真实效果。

    ## 只统计"非空白字符"

    块之间的分隔符（`\n\n`）不属于任何 chunk 是**正常的**：
    它们是解析器插入的结构标记，本身没有检索价值。
    把空白算进分母会让每份文档都显示 97%~99%，
    真正的缺口（某段正文完全没被覆盖）就被淹没。
    """

    covered_chars: int
    total_chars: int
    overlap_chars: int

    @property
    def coverage_ratio(self) -> float:
        """非空白字符中被至少一个 chunk 覆盖的比例。

        **低于 100% 是 bug**：有内容永远检索不到，且没有任何报错。
        """
        return self.covered_chars / self.total_chars if self.total_chars else 0.0

    @property
    def overlap_ratio(self) -> float:
        """非空白字符中被多个 chunk 重复覆盖的比例。

        适度的重叠是期望的（跨块边界的语义完整性）；
        过高则是浪费（重复向量化的成本 + 检索结果重复占位）。
        """
        return self.overlap_chars / self.total_chars if self.total_chars else 0.0


def compute_coverage(text: str, drafts: Sequence[ChunkDraft]) -> Coverage:
    """计算分块对原文的覆盖与重叠（口径见 `Coverage` 的说明）。"""
    text_length = len(text)
    if text_length == 0:
        return Coverage(0, 0, 0)

    hits = [0] * text_length
    for draft in drafts:
        if draft.chunk_level > 0:
            continue  # 父块不参与检索，不计入覆盖
        if draft.char_start is None or draft.char_end is None:
            continue
        start = max(0, draft.char_start)
        end = min(text_length, draft.char_end)
        for position in range(start, end):
            hits[position] += 1

    # 分母：非空白字符（块间分隔符不属于任何 chunk 是正常的）
    meaningful = [index for index, char in enumerate(text) if not char.isspace()]
    if not meaningful:
        return Coverage(0, 0, 0)

    covered = sum(1 for index in meaningful if hits[index] >= 1)
    overlap = sum(hits[index] - 1 for index in meaningful if hits[index] >= 2)
    return Coverage(covered_chars=covered, total_chars=len(meaningful), overlap_chars=overlap)


class BaseChunker(ABC):
    """分块器契约。

    实现约定（违反会导致检索质量静默下降，不报错）：
    - 不产生空块；
    - 位置信息成对且不越界；
    - 受保护类型整块保留；
    - 确定性（不依赖时间/随机/哈希顺序）。
    """

    name: str = "base"
    """策略名。必须与 `ChunkingStrategy` 枚举值一致。"""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    @abstractmethod
    def chunk(self, parsed: ParsedArtifact) -> list[ChunkDraft]:
        """把解析产物切成块。

        Args:
            parsed: 含全文、块列表、页列表。位置信息全在里面。

        Returns:
            按原文顺序排列的块。
        """

    # ------------------------------------------------------------ 公共工具

    def _make_draft(
        self,
        *,
        content: str,
        index: int,
        char_start: int,
        char_end: int,
        source_block: BlockRecord | None = None,
        section_path: Sequence[str] | None = None,
        embedding_input: str | None = None,
        parent_index: int | None = None,
        chunk_level: int = 0,
        chunk_type: str = "text",
        chunk_metadata: dict[str, Any] | None = None,
        include_bbox: bool = True,
    ) -> ChunkDraft:
        """构造 draft，并统一处理位置与类型的默认值。

        集中在父类做，是为了让各策略不会"忘了填某字段"——
        漏填 bbox 的表现是"高亮时好时坏"，极难定位。

        Args:
            include_bbox: 是否从 source_block 继承 bbox。

                `fixed` 策略传 False：它的切点是任意字符位置
                （窗口边界附近的分隔符），不对应任何版面框。
                冒用起点块的 bbox 会让前端框选错区域——高亮整个段落
                而实际内容只有一行。
        """
        bbox = None
        if include_bbox and source_block is not None and source_block.bbox is not None:
            # tuple → dict：`chunks.bbox` 是 JSONB，存 dict 便于前端直接取键
            x0, y0, x1, y1 = source_block.bbox
            bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

        return ChunkDraft(
            content=content,
            chunk_index=index,
            chunk_type=chunk_type,
            char_start=char_start,
            char_end=char_end,
            page_no=source_block.page_no if source_block is not None else None,
            bbox=bbox,
            section_path=list(section_path or []),
            embedding_input=embedding_input,
            parent_index=parent_index,
            chunk_level=chunk_level,
            chunk_metadata=dict(chunk_metadata or {}),
        )

    def _section_prefix(self, section_path: Sequence[str]) -> str:
        """把章节路径拼成 embedding 前缀。

        为什么给 embedding_input 加前缀而 content 不加：
        - 向量需要"这段文字属于哪一章"的上下文才能区分
          "报销额度"在"差旅"和"招待"两章下的不同含义；
        - 但引用展示要高亮**原文**，加前缀会让高亮区间对不上。
        """
        if not section_path:
            return ""
        return " > ".join(section_path) + "\n"


def verify_chunks(
    drafts: Sequence[ChunkDraft],
    parsed: ParsedArtifact,
) -> list[str]:
    """校验分块结果是否满足契约。返回问题列表（空表示通过）。

    在切分任务里、落库之前调用。**不通过必须让任务失败**：
    违反契约的 chunk 会一路影响到检索与引用，而且后续环节无法修正。
    """
    problems: list[str] = []
    text_length = len(parsed.text)

    previous_start = -1
    for index, draft in enumerate(drafts):
        if not draft.content.strip():
            problems.append(f"chunk[{index}] 内容为空")

        if draft.chunk_index != index:
            problems.append(f"chunk[{index}] 的 chunk_index={draft.chunk_index} 与下标不一致")

        # 位置必须成对
        if (draft.char_start is None) != (draft.char_end is None):
            problems.append(f"chunk[{index}] 的 char_start/char_end 未成对出现")
        elif draft.char_start is not None and draft.char_end is not None:
            if draft.char_end <= draft.char_start:
                problems.append(
                    f"chunk[{index}] char_end({draft.char_end}) <= char_start({draft.char_start})"
                )
            elif draft.char_end > text_length:
                problems.append(
                    f"chunk[{index}] char_end({draft.char_end}) 超出文本长度({text_length})"
                )
            elif draft.char_start < previous_start:
                # 允许相等（父子策略里父块与子块可以同起点），不允许倒退
                problems.append(
                    f"chunk[{index}] char_start({draft.char_start}) 早于前一块({previous_start})"
                )
            else:
                # 最关键的校验：切出来的区间必须真的是这段文字。
                # 偏移算错 = 引用高亮错位，而这个错误不会在答案里体现。
                actual = parsed.text[draft.char_start : draft.char_end]
                if actual != draft.content:
                    problems.append(
                        f"chunk[{index}] 内容与源文本切片不一致："
                        f"期望 {draft.content[:30]!r}，实际 {actual[:30]!r}"
                    )
                previous_start = max(previous_start, draft.char_start)

    return problems


__all__ = [
    "PROTECTED_BLOCK_TYPES",
    "BaseChunker",
    "ChunkConfig",
    "ChunkDraft",
    "Coverage",
    "compute_coverage",
    "verify_chunks",
]
