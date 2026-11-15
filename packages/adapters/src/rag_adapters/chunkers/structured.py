"""标题层级结构化切分。

## 与递归切分的本质区别

递归切分只看**长度与分隔符**，不知道文档结构。
结构化切分看**标题树**：一块内容隶属于哪个章节，是文档自带的语义。

这带来两个递归切分做不到的能力：

1. **`section_path` 完整**：chunk 知道自己属于"第三章 > 3.2 报销流程"。
   引用展示能给出面包屑，用户能判断"这段话的适用上下文是什么"。
2. **章节不混切**：两块内容分属不同章节时绝不合并，哪怕都很短。
   递归切分会把"第三章末尾一句"和"第四章开头一句"合并成一个 chunk，
   检索时这个块同时"像"两个主题，反而两边都不够准。

## embedding_input 加前缀

结构化策略的独有价值在这里：只把**正文**送去向量化，会把
"额度为 5000 元"变成一个脱离语境的孤句。加上章节前缀：

    "第三章 差旅管理 > 3.2 报销流程
     额度为 5000 元。"

向量就能区分"差旅报销额度"与"招待报销额度"——两者正文可能完全相同。

**但 `content` 不加前缀**：引用要高亮原文，加了前缀高亮区间就对不上。

## 位置信息：只从源文本切片，绝不拼接

合并多个块时，**不能**写 `"\\n\\n".join(block.text)`。
块之间的分隔符由解析器决定（PDF 用 `"\\n\\n"`，纯文本按行、CSV 按批），
拼接出来的字符串与源文本区间**不保证一致**。

不一致的后果：`text[char_start:char_end] != content`，
前端按偏移高亮会显示错误的文字。这类 bug 在答案层面完全看不出来。

所以本实现统一用 `parsed.text[first.char_start : last.char_end]` 取正文，
由构造保证 `content` 就是那段源文本。

## 标题检测不可靠时的降级

如果整篇没有一个标题块（比如纯文本 TXT），`section_stack` 为空，
所有内容归入同一"章节"，行为退化为"按块边界合并到预算内"——
接近 recursive 但严格按块边界，不跨块切分。
不抛异常：策略要能跑任何输入。
"""

from __future__ import annotations

from rag_adapters.chunkers.base import PROTECTED_BLOCK_TYPES, BaseChunker, ChunkDraft
from rag_adapters.parsers.artifact import BlockRecord, ParsedArtifact


class StructuredChunker(BaseChunker):
    """按标题层级切分。"""

    name = "structured"

    def chunk(self, parsed: ParsedArtifact) -> list[ChunkDraft]:
        if not parsed.text.strip():
            return []

        drafts: list[ChunkDraft] = []
        index = 0

        # 当前章节路径（按 level 维护的栈）与本章节已积累的块
        section_stack: list[tuple[int, str]] = []
        pending: list[BlockRecord] = []

        def flush() -> None:
            nonlocal index
            if not pending:
                return
            section_path = [title for _, title in section_stack]
            for draft in self._emit_section(
                parsed=parsed,
                blocks=pending,
                section_path=section_path,
                start_index=index,
            ):
                drafts.append(draft)
                index += 1
            pending.clear()

        for block in parsed.blocks:
            if not block.text.strip():
                continue

            is_heading = block.block_type == "title" or block.level > 0

            if is_heading:
                # 新章节开始：先把上一节的内容落下来
                flush()

                level = max(block.level, 1)
                # 弹出层级 >= 当前标题的旧章节（它们是同级或更深的兄弟/子节点）
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, block.text.strip()))

                # 标题本身也算内容：它承载的关键词对检索有用
                pending.append(block)
                continue

            pending.append(block)

        flush()
        # 孤立标题合并：一级标题后直接跟二级标题时，前者会单独成块
        return self._merge_orphan_headings(drafts, parsed)

    def _emit_section(
        self,
        *,
        parsed: ParsedArtifact,
        blocks: list[BlockRecord],
        section_path: list[str],
        start_index: int,
    ) -> list[ChunkDraft]:
        """把一个章节内的块包装成 chunk。

        合并规则：
        - 受保护类型（表格/代码/公式）**单独成块**，不与正文混。
          混在一起时，表格的字符区间与正文不连续，偏移无法表达；
          而且检索命中时拿不到完整表格。
        - 正文块在不超过 `chunk_size` 的前提下尽量合并。
          同章节内连续正文合并是安全的：语义同源。
        """
        drafts: list[ChunkDraft] = []
        index = start_index
        budget = self.config.chunk_size

        buffer: list[BlockRecord] = []

        def flush_buffer() -> None:
            nonlocal index
            if not buffer:
                return
            content = self._slice(parsed, buffer)
            drafts.append(
                self._make_draft(
                    content=content,
                    index=index,
                    char_start=buffer[0].char_start,
                    char_end=buffer[-1].char_end,
                    source_block=buffer[0],
                    section_path=section_path,
                    embedding_input=self._contextual_input(content, section_path),
                )
            )
            index += 1
            buffer.clear()

        for block in blocks:
            if block.block_type in PROTECTED_BLOCK_TYPES:
                flush_buffer()
                drafts.append(
                    self._make_draft(
                        content=block.text,
                        index=index,
                        char_start=block.char_start,
                        char_end=block.char_end,
                        source_block=block,
                        section_path=section_path,
                        chunk_type=block.block_type,
                        # 表格也要章节上下文：一张"600/400/300"的表格
                        # 脱离章节就无法判断是住宿标准还是招待标准。
                        embedding_input=self._contextual_input(block.text, section_path),
                    )
                )
                index += 1
                continue

            if buffer:
                prospective = self._slice(parsed, [*buffer, block])
                if len(prospective) > budget:
                    flush_buffer()

            buffer.append(block)

            # 单块就超预算：立刻落下，避免缓冲区无限增长
            if len(self._slice(parsed, buffer)) >= budget:
                flush_buffer()

        flush_buffer()
        return drafts

    def _merge_orphan_headings(
        self, drafts: list[ChunkDraft], parsed: ParsedArtifact
    ) -> list[ChunkDraft]:
        """把"只含标题"的孤立块并入下一块。

        ## 为什么需要

        文档常有这种结构：

            # 公司费用管理制度     ← 一级标题，后面直接跟二级标题
            ## 1. 差旅报销
            正文...

        一级标题单独成块只有 10 个字符。它的向量几乎没有信息量，
        但在检索时会占一个结果位，把真正有内容的块挤下去。

        ## 为什么并到**下一块**而不是上一块

        标题是它**后面**内容的统领。并到上一块会让上一章的正文
        突然挂上一个不相关的标题。并到下一块才符合语义。

        ## 为什么只处理"纯标题块"

        有正文的块要保留原样（正文本身有价值，不需要额外处理）。
        只有"整块就是一个标题"才需要特殊照顾。
        """
        if len(drafts) < 2:
            return drafts

        min_size = self.config.min_chunk_size
        result: list[ChunkDraft] = []
        position = 0
        pending = list(drafts)

        while position < len(pending):
            current = pending[position]
            is_orphan_heading = (
                len(current.content.strip()) < min_size
                and current.chunk_level == 0
                and position + 1 < len(pending)
                and pending[position + 1].page_no == current.page_no
                and current.char_end is not None
                and pending[position + 1].char_start is not None
            )

            if is_orphan_heading:
                following = pending[position + 1]
                # 中间只允许空白分隔符（否则会丢字）
                gap = parsed.text[current.char_end : following.char_start]
                merged_length = len(current.content) + len(gap) + len(following.content)
                if not gap.strip() and merged_length <= self.config.chunk_size:
                    pending[position + 1] = ChunkDraft(
                        content=parsed.text[current.char_start : following.char_end],
                        chunk_index=current.chunk_index,
                        chunk_type=following.chunk_type,
                        char_start=current.char_start,
                        char_end=following.char_end,
                        page_no=current.page_no,
                        bbox=current.bbox,
                        # 章节路径取**下一块**的：标题属于它统领的章节
                        section_path=following.section_path,
                        embedding_input=following.embedding_input,
                        chunk_level=0,
                    )
                    position += 1
                    continue

            result.append(current)
            position += 1

        for order, draft in enumerate(result):
            draft.chunk_index = order

        return result

    def _contextual_input(self, content: str, section_path: list[str]) -> str | None:
        """构造 embedding 输入：章节前缀 + 正文。

        **必须去掉与正文重复的部分**：章节标题块本身就在 content 里
        （它承载检索关键词），如果前缀又写一遍完整路径，就会出现
        "第三章 差旅管理\\n第三章 差旅管理\\n\\n正文"——同一句话两次。

        重复的实际代价：向量被这段重复文字轻微带偏，且在预算有限的
        上下文里白占 token。做法是从路径**尾部**去掉 content 已经以之
        开头的元素。
        """
        if not section_path:
            return None

        # 从尾部剥离：content 以 "3.1 差旅报销" 开头时，路径里的
        # 最后一段就不必再加前缀
        effective = list(section_path)
        stripped = content.lstrip()
        while effective and stripped.startswith(effective[-1]):
            stripped = stripped[len(effective[-1]) :].lstrip()
            effective.pop()

        if not effective:
            # 正文已经把整个路径都覆盖了（短文档/标题即全文）：不需要前缀
            return None

        return self._section_prefix(effective) + content

    @staticmethod
    def _slice(parsed: ParsedArtifact, blocks: list[BlockRecord]) -> str:
        """取连续块在**源文本**中的区间。

        这是本实现唯一正确的拼接方式（见模块 docstring）。

        边界情况：块之间可能有分隔符（如 `"\\n\\n"`），
        直接切片会把分隔符也包含进来。这是**期望行为**——
        它们属于源文本，且保证 `text[char_start:char_end] == content`。
        """
        return parsed.text[blocks[0].char_start : blocks[-1].char_end]


__all__ = ["StructuredChunker"]
