"""固定长度滑窗切分（baseline）。

## 这是最笨的策略，但必须存在

不做任何语义判断，纯按 `chunk_size` 开窗、`chunk_overlap` 步进。
它的价值是**对照组**：如果结构化/语义策略在黄金集上打不过它，
说明那些策略的额外复杂度没有回报，应该回退。

很多项目一上来就上语义切分，却从没测过 baseline——
结果无法回答"这个复杂度到底带来了多少收益"。

## 为什么按段落边界回退

纯按字符切会把句子劈成两半（"报销额度为" / "5000 元"）。
这里在窗口末尾向前找一个合适的分隔符（段落 > 换行 > 句号），
只有找不到时才硬切。

这是"用最小代价消除最明显的伤害"，复杂度极低。
真正的语义判断留给 recursive/semantic。

## 位置信息的处理

窗口在**源文本**上滑动，`char_start`/`char_end` 直接就是窗口边界。
但回退到分隔符后，实际结束位置会提前 —— 必须用**调整后**的边界，
否则 `text[start:end]` 切出来的内容与 `content` 不一致
（`verify_chunks` 会抓到）。
"""

from __future__ import annotations

from rag_adapters.chunkers.base import BaseChunker, ChunkDraft
from rag_adapters.parsers.artifact import BlockRecord, ParsedArtifact

# 回退查找分隔符的优先级。越靠前越"干净"。
# 段落 > 行 > 句末 > 逗号/分号 > 空白
_BREAK_POINTS = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", ";", "，", ",", " ")


class FixedChunker(BaseChunker):
    """固定滑窗。

    参数解释：
    - `chunk_size`：窗口宽度（字符）；
    - `chunk_overlap`：相邻窗口重叠量；
    - 步进 = `chunk_size - chunk_overlap`，必须 > 0（`ChunkConfig` 已校验）。
    """

    name = "fixed"

    def chunk(self, parsed: ParsedArtifact) -> list[ChunkDraft]:
        text = parsed.text
        if not text.strip():
            return []

        size = self.config.chunk_size
        drafts: list[ChunkDraft] = []

        cursor = 0
        index = 0
        text_length = len(text)

        while cursor < text_length:
            raw_end = min(cursor + size, text_length)
            end = raw_end

            if raw_end < text_length:
                # 在窗口后 1/4 范围内向前找分隔符，避免硬切。
                # 只往前找（不往后）是为了保证步进仍然前进、不死循环。
                search_floor = cursor + (size * 3 // 4)
                end = self._find_break(text, floor=search_floor, ceiling=raw_end)

            content = text[cursor:end]
            if content.strip():
                # 位置信息：按**窗口起点**所在的块决定 page_no。
                #
                # 为什么不用终点所在的块：一个窗口可能横跨两页，
                # 只有起点页是确定的。用终点会让"从第 1 页开始的块"
                # 显示成第 2 页——引用跳转会偏一页。
                #
                # **故意不给 bbox**：滑窗边界是任意字符位置，没有对应的
                # 版面框。冒用起点块的 bbox 会让前端在原文上框错区域
                # （比如高亮整个段落而实际内容只有一行）。
                # 前端应据 `positioning=char_offset` 用字符区间高亮。
                anchor = self._block_at(parsed, cursor)
                drafts.append(
                    self._make_draft(
                        content=content,
                        index=index,
                        char_start=cursor,
                        char_end=end,
                        source_block=anchor,
                        # 不继承 bbox：滑窗边界没有对应的版面框
                        include_bbox=False,
                        chunk_metadata={"positioning": "char_offset"},
                    )
                )
                index += 1

            if end >= text_length:
                break

            # 从"实际结束位置"回退 overlap 作为下一个窗口起点。
            # 用 end 而不是 raw_end：否则调整过的尾部会被下一个窗口
            # 完整重复一遍，产生大段冗余。
            cursor = max(end - self.config.chunk_overlap, cursor + 1)

        return drafts

    @staticmethod
    def _block_at(parsed: ParsedArtifact, offset: int) -> BlockRecord | None:
        """找包含指定字符偏移的块（用于取页码）。

        块按 char_start 升序排列。块数通常几千以内，线性扫描够快，
        不必为此引入二分查找的复杂度。
        """
        for block in parsed.blocks:
            if block.char_start <= offset < block.char_end:
                return block
            if block.char_start > offset:
                break

        # 落在块之间的分隔符上：用前一个块（分隔符跟在它后面）
        fallback: BlockRecord | None = None
        for block in parsed.blocks:
            if block.char_end <= offset:
                fallback = block
            else:
                break
        return fallback

    @staticmethod
    def _find_break(text: str, *, floor: int, ceiling: int) -> int:
        """在 `[floor, ceiling]` 里从后向前找最靠后的分隔符，返回切点。

        找不到就返回 `ceiling`（硬切）。
        """
        best = ceiling
        for separator in _BREAK_POINTS:
            position = text.rfind(separator, floor, ceiling)
            if position != -1:
                # 切在分隔符**之后**（保留分隔符属于本块，
                # 这样下一块开头不会是一个孤零零的换行）
                candidate = position + len(separator)
                if candidate < best:
                    best = candidate
        return best


__all__ = ["FixedChunker"]
