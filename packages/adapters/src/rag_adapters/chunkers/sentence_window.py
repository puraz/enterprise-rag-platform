"""句子窗口切分。

## 与父子策略的相似与不同

两者都是"小块检索、大块回填"，但机制不同：

| | parent_child | sentence_window |
|---|---|---|
| 检索单元 | 子块（递归切分的产物） | **单个句子** |
| 回填内容 | 父块（预先切好的大块） | **动态取前后 N 句** |
| 存储 | 父+子都入库 | 只存句子；窗口在检索时拼接 |
| 适合 | 结构化文档 | 连续散文、问答、对话记录 |

`sentence_window` 的优势：窗口大小可以在**查询时**决定。
同一个句子，简单问题取前后 1 句就够，复杂问题取前后 3 句——
不需要重新切分、不需要重建索引。

代价：必须保留"句子 → 相邻句子"的关系。本实现用
`chunk_metadata = {"window_id": ..., "sentence_index": ...}` 表达，
检索时按 `window_id` 取相邻句子。

## 为什么不用 `parent_chunk_id`

父块是**静态**的（切分时定死）。句子窗口的回填是**动态**的
（按查询决定范围）。硬用 `parent_chunk_id` 会让窗口大小锁死在切分时，
失去这个策略的核心价值。

## 句子切分的中文难点

中文没有空格分词，句末标点是唯一可靠边界：`。！？；`
但有几个陷阱：
- **引号包裹**：`他说："好的。"然后离开。` —— 引号内的句号不是句子边界；
- **省略号**：`……` 常常是停顿而非句末；
- **小数点/编号**：`1.5 倍`、`3.2 节` 里的点不是句末。

本实现用正则按标点切分，并**保留标点到句中**（切在标点之后），
对小数的处理靠"标点后必须跟空白或中文"这个条件过滤掉 `1.5`。
"""

from __future__ import annotations

import re

from rag_adapters.chunkers.base import BaseChunker, ChunkDraft
from rag_adapters.parsers.artifact import ParsedArtifact

# 句子边界：中文句末标点 + 英文句末（后面必须跟空白，避免切开 1.5 / 3.2 节）
# 用捕获组保留标点，切分后再拼回前一句
_SENTENCE_PATTERN = re.compile(r"(?<=[。！？；!?;])\s*|(?<=[.])\s+(?=[A-Z\u4e00-\u9fff])")

# 每个句子窗口默认引用前后各几句（存进 metadata，检索时可覆盖）
_DEFAULT_WINDOW_SIZE = 1


class SentenceWindowChunker(BaseChunker):
    """句子级切分 + 窗口元数据。"""

    name = "sentence_window"

    def chunk(self, parsed: ParsedArtifact) -> list[ChunkDraft]:
        if not parsed.text.strip():
            return []

        drafts: list[ChunkDraft] = []
        index = 0
        # 窗口 ID：同一"段落组"内的句子共享，检索回填时按它取邻句。
        # 用块序号而非全局序号：跨块（如跨页）的句子不该被当成同一窗口，
        # 否则回填会把"上一页结尾"和"下一页开头"拼在一起。
        for block_index, block in enumerate(parsed.blocks):
            if not block.text.strip():
                continue

            sentences = self._split_sentences(block)
            if not sentences:
                continue

            window_id = f"b{block_index}"

            for sentence_index, (sentence, offset, length) in enumerate(sentences):
                # 短句合并：孤立的一两个字的句子（"是。"）单独成块
                # 没有任何检索价值，反而会污染结果。合并到前一句。
                if drafts and len(sentence.strip()) < self.config.min_chunk_size:
                    previous = drafts[-1]
                    if previous.chunk_metadata.get("window_id") == window_id:
                        # 合并后仍要保证 content 与源文本区间一致
                        new_end = offset + length
                        start = previous.char_start or offset
                        previous.content = parsed.text[start:new_end]
                        previous.char_end = new_end
                        previous.chunk_metadata["sentence_count"] = (
                            int(previous.chunk_metadata.get("sentence_count", 1)) + 1
                        )
                        continue

                # 单句超过 chunk_size 时，退化为按预算硬切：
                # 一个 3000 字的句子（无标点）必须切，否则向量覆盖太泛。
                if len(sentence) > self.config.chunk_size:
                    for piece_start in range(0, len(sentence), self.config.chunk_size):
                        piece = sentence[piece_start : piece_start + self.config.chunk_size]
                        if not piece.strip():
                            continue
                        drafts.append(
                            self._make_draft(
                                content=piece,
                                index=index,
                                char_start=offset + piece_start,
                                char_end=offset + piece_start + len(piece),
                                source_block=block,
                                chunk_metadata={
                                    "window_id": window_id,
                                    "sentence_index": sentence_index,
                                    "sentence_count": 1,
                                    "window_size": _DEFAULT_WINDOW_SIZE,
                                    "split_long_sentence": True,
                                },
                            )
                        )
                        index += 1
                    continue

                drafts.append(
                    self._make_draft(
                        content=sentence,
                        index=index,
                        char_start=offset,
                        char_end=offset + length,
                        source_block=block,
                        chunk_metadata={
                            # 检索侧按 window_id + sentence_index 取前后 N 句
                            "window_id": window_id,
                            "sentence_index": sentence_index,
                            "sentence_count": 1,
                            "window_size": _DEFAULT_WINDOW_SIZE,
                        },
                    )
                )
                index += 1

        # 跨块合并孤立短句。
        #
        # 为什么需要：块内合并只能处理"同一 block 内的短句"。
        # 但**块本身就是短句**的情况很常见（标题块"3.2 业务招待"只有 8 字符），
        # 而它前后可能是受保护类型（表格）或批次边界，块内合并管不到。
        #
        # 与块内合并的区别：这里**要丢弃 window_id**（两块来自不同窗口），
        # 所以合并后的块不能再靠 window_id 回填补齐——它已经是完整上下文了。
        drafts = self._merge_cross_block_short(drafts, parsed)
        for position, draft in enumerate(drafts):
            draft.chunk_index = position

        return drafts

    def _merge_cross_block_short(
        self, drafts: list[ChunkDraft], parsed: ParsedArtifact
    ) -> list[ChunkDraft]:
        """把过短的孤立块并入相邻块（同页、不跨受保护类型）。

        合并后清空 window 元数据：这个块现在自带上下文，
        检索侧不该再按窗口扩展（否则会把邻近的无关句子也拉进来）。
        """
        if len(drafts) < 2:
            return drafts

        min_size = self.config.min_chunk_size
        budget = self.config.chunk_size

        def mergeable(left: ChunkDraft, right: ChunkDraft) -> bool:
            if left.page_no != right.page_no:
                return False
            if left.char_end is None or right.char_start is None:
                return False
            if right.char_start < left.char_end:
                return False
            if parsed.text[left.char_end : right.char_start].strip():
                return False
            return len(left.content) + len(right.content) <= budget

        def merged(left: ChunkDraft, right: ChunkDraft) -> ChunkDraft:
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
                # 合并块横跨多个句子/窗口：不再是"单个句子窗口"
                chunk_metadata={"merged": True, "sentence_count": 0},
            )

        # 两遍：先向后（首块无前驱、前块是受保护类型时只能向后）
        forward: list[ChunkDraft] = []
        position = 0
        pending = list(drafts)
        while position < len(pending):
            current = pending[position]
            if (
                len(current.content) < min_size
                and position + 1 < len(pending)
                and mergeable(current, pending[position + 1])
            ):
                pending[position + 1] = merged(current, pending[position + 1])
                position += 1
                continue
            forward.append(current)
            position += 1

        # 再向前
        result: list[ChunkDraft] = []
        for draft in forward:
            if result and len(draft.content) < min_size and mergeable(result[-1], draft):
                result[-1] = merged(result[-1], draft)
            else:
                result.append(draft)
        return result

    # ------------------------------------------------------------ 句子切分

    @staticmethod
    def _split_sentences(block: object) -> list[tuple[str, int, int]]:
        """把块切成句子，返回 `(句子文本, 源文本偏移, 长度)`。

        **基于块的 `text` 与 `char_start`**：句子的绝对偏移
        = 块偏移 + 句内相对偏移。

        尾部没有标点的残句也要保留（文档常以无标点的列表结尾）。
        """
        text = getattr(block, "text", "")
        base = getattr(block, "char_start", 0)

        sentences: list[tuple[str, int, int]] = []
        cursor = 0

        for match in _SENTENCE_PATTERN.finditer(text):
            boundary = match.end()
            if boundary <= cursor:
                continue
            sentence = text[cursor:boundary]
            if sentence.strip():
                # 保留前导空白在偏移里（否则相邻句子之间会漏字符，
                # 让 verify_chunks 的区间校验失败）
                sentences.append((sentence, base + cursor, len(sentence)))
            cursor = boundary

        remainder = text[cursor:]
        if remainder.strip():
            sentences.append((remainder, base + cursor, len(remainder)))

        return sentences


__all__ = ["SentenceWindowChunker"]
