"""递归字符切分（生产默认）。

## 思路

按分隔符优先级递归下降：

    段落 → 换行 → 句子 → 逗号 → 单字

先用最高优先级的分隔符把文本切成段；某段仍超长就用下一级分隔符继续切。
这样得到的块**尽量落在语义边界上**，同时长度受控。

这是 LangChain `RecursiveCharacterTextSplitter` 的同一思路，但这里自己实现
（约 200 行）而不是引入依赖，理由：
- 需要精确的**源文本偏移**，第三方实现只给文本不给偏移；
- 需要与 `ParsedArtifact` 的块结构对齐（保留 page/bbox/section）；
- 需要可控的合并规则（短块合并、受保护类型跳过）。

## 与固定滑窗的区别

| | fixed | recursive |
|---|---|---|
| 切点 | 窗口末端附近任意分隔符 | 按优先级逐级下降 |
| 短块 | 不合并 | 合并到相邻块 |
| 位置 | 窗口边界 | 原始块边界 |

recursive 的块长度**不是严格等于 chunk_size**——它在"不超长"的前提下
尽量用大的语义单元。这让块长度方差更大，但语义更完整。

## 关键实现细节：偏移如何保持正确

递归过程中我们不断把文本切成子串。**不能**对子串做 `str.split` 然后
丢掉位置——那样每层都要重新计算偏移，必然出错。

做法：递归函数始终携带 `base_offset`（子串在源文本中的起点），
每次切分时记录每个片段相对子串的起点，加上 `base_offset` 就是源文本偏移。

这个不变量（`text[base:base+len(part)] == part`）由 `verify_chunks` 校验。
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_adapters.chunkers.base import PROTECTED_BLOCK_TYPES, BaseChunker, ChunkDraft
from rag_adapters.parsers.artifact import ParsedArtifact

# 分隔符优先级（高 → 低）。
# 中文文档里"。"比"."更常见，所以两个都放；顺序影响切点选择。
_SEPARATORS: tuple[str, ...] = (
    "\n\n",  # 段落
    "\n",  # 行
    "。",  # 中文句末
    "！",
    "？",
    "；",
    ". ",  # 英文句末（带空格，避免切开小数点/缩写）
    "! ",
    "? ",
    "; ",
    "，",  # 中文逗号（最后手段，比硬切好）
    ", ",
    " ",  # 空格（英文词边界）
    "",  # 单字符硬切
)


@dataclass(slots=True)
class _Piece:
    """切分过程中的片段（带源文本偏移）。"""

    text: str
    start: int
    end: int


class RecursiveChunker(BaseChunker):
    """递归字符切分。"""

    name = "recursive"

    def chunk(self, parsed: ParsedArtifact) -> list[ChunkDraft]:
        if not parsed.text.strip():
            return []

        drafts: list[ChunkDraft] = []
        index = 0

        # 逐块处理，而不是把全文当一个字符串：
        # 块边界（page/bbox/标题层级）本身就是最好的语义边界，
        # 跨块合并会丢掉位置信息（一个 chunk 横跨两页，bbox 该填哪个？）。
        for block in parsed.blocks:
            if not block.text.strip():
                continue

            if block.block_type in PROTECTED_BLOCK_TYPES:
                # 表格/代码/公式整块保留。切开会行列错位或失去语义。
                # 即使超长也不切——超长表格是"必须接受"的代价，
                # 真正的解法是在检索侧做截断/分页，而不是破坏结构。
                drafts.append(
                    self._make_draft(
                        content=block.text,
                        index=index,
                        char_start=block.char_start,
                        char_end=block.char_end,
                        source_block=block,
                        chunk_type=block.block_type,
                    )
                )
                index += 1
                continue

            pieces = self._split_recursive(
                text=block.text,
                base_offset=block.char_start,
                separator_index=0,
            )

            # 短片段合并：孤立的一两行（如"注："）信息量太低，
            # 单独成块会在检索结果里占位但没有价值。
            merged = self._merge_short_pieces(pieces)

            for piece in merged:
                if not piece.text.strip():
                    continue
                drafts.append(
                    self._make_draft(
                        content=piece.text,
                        index=index,
                        char_start=piece.start,
                        char_end=piece.end,
                        source_block=block,
                        chunk_type=block.block_type,
                    )
                )
                index += 1

        # 跨块合并孤立短块。
        #
        # 为什么必须做：标题常常是独立的 block（"3.1 差旅报销" 只有 8 字符）。
        # 块内合并管不到它，结果是一个 8 字符的 chunk 进入向量库——
        # 它的向量几乎不携带语义，但在检索时仍会占用一个结果位，
        # 把真正有内容的块挤下去。
        #
        # 只在**同一页且都是普通正文**时合并：跨页合并会让 page_no/bbox
        # 无法表达（一个块横跨两页，引用高亮该跳哪一页？）。
        drafts = self._merge_across_blocks(drafts, parsed)

        # 合并后下标会变，重新编号保证 chunk_index 从 0 连续
        for position, draft in enumerate(drafts):
            draft.chunk_index = position

        return drafts

    def _merge_across_blocks(
        self, drafts: list[ChunkDraft], parsed: ParsedArtifact
    ) -> list[ChunkDraft]:
        """把过短的孤立块并入相邻块（同页、同类型、不超预算）。

        从后往前扫：并入"前一块"后不需要回头重新判断，单次遍历完成。
        第一块太短时并入后一块（前面没有可并对象）。
        """
        if len(drafts) < 2:
            return drafts

        min_size = self.config.min_chunk_size
        budget = self.config.chunk_size

        def mergeable(left: ChunkDraft, right: ChunkDraft) -> bool:
            """两块能否合并成一个合法 chunk。"""
            if left.chunk_level != 0 or right.chunk_level != 0:
                return False  # 父/子块由 parent_child 策略管理，不在这里动
            if left.page_no != right.page_no:
                return False  # 跨页 → page_no/bbox 无法表达
            if (
                left.chunk_type in PROTECTED_BLOCK_TYPES
                or right.chunk_type in PROTECTED_BLOCK_TYPES
            ):
                # 表格/代码/公式不能并进正文：并进去之后这个 chunk 的类型
                # 无法表达（既不是表格也不是纯正文），检索时按类型过滤会出错。
                return False
            if left.char_end is None or right.char_start is None:
                return False
            if right.char_start < left.char_end:
                return False  # 区间重叠（父子块/异常输入），不合并
            # 中间只允许是分隔符/空白。
            #
            # **不能要求严格连续**（`left.char_end == right.char_start`）：
            # 解析器在块之间插入了 "\n\n"，严格连续会让绝大多数块永远
            # 无法合并——短标题块就永远孤立。
            #
            # 用源文本切片作为新 content，分隔符自然包含在内，
            # 偏移与内容仍然严格一致（verify_chunks 校验的就是这一点）。
            gap = parsed.text[left.char_end : right.char_start]
            if gap.strip():
                return False  # 中间夹着真实内容 → 合并会丢字
            return len(left.content) + len(gap) + len(right.content) <= budget

        def merged_chunk(left: ChunkDraft, right: ChunkDraft) -> ChunkDraft:
            """把两块合并成一个。

            合并后的类型取 `text`：无论是 "标题+正文" 还是 "列表+正文"，
            内容主体都是正文。保留 left 的位置起点与 right 的终点。
            """
            return ChunkDraft(
                content=parsed.text[left.char_start : right.char_end],
                chunk_index=left.chunk_index,
                chunk_type="text",
                char_start=left.char_start,
                char_end=right.char_end,
                page_no=left.page_no,
                bbox=left.bbox,
                section_path=left.section_path,
                chunk_level=0,
            )

        result: list[ChunkDraft] = []
        pending = list(drafts)

        # 第一遍：向后合并短块。
        #
        # 为什么两遍：短块可能出现在两类位置——
        #   1. 上一块是表格/代码（受保护，不能并）→ 只能向后并；
        #   2. 前面没块（批次首块）→ 只能向后并。
        # 完全靠"向前合并"会让这两类短块永远孤立。
        forward: list[ChunkDraft] = []
        position = 0
        while position < len(pending):
            current = pending[position]
            if (
                len(current.content) < min_size
                and position + 1 < len(pending)
                and mergeable(current, pending[position + 1])
            ):
                # 与下一块合并，结果继续参与判断（可能还能继续并）
                pending[position + 1] = merged_chunk(current, pending[position + 1])
                position += 1
                continue
            forward.append(current)
            position += 1

        # 第二遍：向前合并仍过短的块（此时向后已无对象）
        for draft in forward:
            if result and len(draft.content) < min_size and mergeable(result[-1], draft):
                result[-1] = merged_chunk(result[-1], draft)
            else:
                result.append(draft)

        return result

    # ------------------------------------------------------------ 递归核心

    def _split_recursive(
        self,
        *,
        text: str,
        base_offset: int,
        separator_index: int,
    ) -> list[_Piece]:
        """递归切分。

        Args:
            text: 待切分的子串。
            base_offset: 该子串在源文本中的起点（偏移基准）。
            separator_index: 当前使用的分隔符下标。

        不变量：`text` 就是 `源文本[base_offset : base_offset+len(text)]`。
        """
        budget = self.config.chunk_size

        # 够短：不再切
        if len(text) <= budget:
            return [_Piece(text=text, start=base_offset, end=base_offset + len(text))]

        # 分隔符用尽（最后一档是 ""）：只能硬切
        separator = _SEPARATORS[min(separator_index, len(_SEPARATORS) - 1)]
        if separator == "":
            return self._hard_split(text=text, base_offset=base_offset)

        segments = self._split_keep_offsets(text=text, separator=separator, base_offset=base_offset)

        # 分隔符切不动（整段里没有这个分隔符）：换下一级
        if len(segments) <= 1:
            return self._split_recursive(
                text=text,
                base_offset=base_offset,
                separator_index=separator_index + 1,
            )

        results: list[_Piece] = []
        for segment in segments:
            if len(segment.text) <= budget:
                results.append(segment)
            else:
                # 该片段仍超长：用下一级分隔符继续切（递归下降）
                results.extend(
                    self._split_recursive(
                        text=segment.text,
                        base_offset=segment.start,
                        separator_index=separator_index + 1,
                    )
                )
        return results

    @staticmethod
    def _split_keep_offsets(*, text: str, separator: str, base_offset: int) -> list[_Piece]:
        """按分隔符切分，并保留每段的源文本偏移。

        **切点保留在分隔符之后**：让每段自带结尾分隔符（换行/句号）。
        否则下一段开头会是纯文本，读起来缺标点。
        """
        pieces: list[_Piece] = []
        cursor = 0
        length = len(text)

        while cursor < length:
            found = text.find(separator, cursor)
            if found == -1:
                tail = text[cursor:]
                if tail:
                    pieces.append(
                        _Piece(
                            text=tail,
                            start=base_offset + cursor,
                            end=base_offset + length,
                        )
                    )
                break

            end = found + len(separator)
            piece_text = text[cursor:end]
            pieces.append(
                _Piece(
                    text=piece_text,
                    start=base_offset + cursor,
                    end=base_offset + end,
                )
            )
            cursor = end

        return pieces

    def _hard_split(self, *, text: str, base_offset: int) -> list[_Piece]:
        """无分隔符可用时的硬切（按预算等分）。"""
        budget = self.config.chunk_size
        pieces: list[_Piece] = []
        for start in range(0, len(text), budget):
            end = min(start + budget, len(text))
            pieces.append(
                _Piece(
                    text=text[start:end],
                    start=base_offset + start,
                    end=base_offset + end,
                )
            )
        return pieces

    # ------------------------------------------------------------ 合并

    def _merge_short_pieces(self, pieces: list[_Piece]) -> list[_Piece]:
        """把过短的片段合并到前一个（或后一个）片段。

        规则：
        - 从后往前扫，短片段优先并入前一个（保持原文顺序）；
        - 合并后不能超过 `chunk_size`（否则本来合格的块会超长）；
        - 第一个片段太短则并入后一个（前面没有可并的对象）。

        为什么从后往前：这样"并入前一个"之后不需要重新处理前面的片段，
        单次遍历即可。从前往后合并会让后面的短片段反复触发前移。
        """
        if not pieces:
            return []

        min_size = self.config.min_chunk_size
        budget = self.config.chunk_size

        # 先处理"首个片段太短"的情况（前面没有可并对象）
        if (
            len(pieces) > 1
            and len(pieces[0].text) < min_size
            and len(pieces[0].text) + len(pieces[1].text) <= budget
        ):
            merged_first = _Piece(
                text=pieces[0].text + pieces[1].text,
                start=pieces[0].start,
                end=pieces[1].end,
            )
            pieces = [merged_first, *pieces[2:]]

        result: list[_Piece] = [pieces[0]]
        for piece in pieces[1:]:
            previous = result[-1]
            can_merge = (
                len(piece.text) < min_size and len(previous.text) + len(piece.text) <= budget
            )
            if can_merge:
                result[-1] = _Piece(
                    text=previous.text + piece.text,
                    start=previous.start,
                    end=piece.end,
                )
            else:
                result.append(piece)

        return result


__all__ = ["RecursiveChunker"]
