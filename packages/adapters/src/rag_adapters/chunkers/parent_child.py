"""父子文档切分（small-to-big）。

## 解决什么矛盾

检索精度与上下文完整性是**冲突**的：

- 块小 → 向量只表达一个局部语义，匹配精准；但命中后送给 LLM 的上下文太少，
  模型看不出前因后果，容易答非所问。
- 块大 → 上下文完整；但向量是整块的"平均语义"，局部细节被稀释，
  检索时反而匹配不到具体问题。

父子策略两头都要：

    父块（大，~2×chunk_size）  ← 只用于回填上下文，不参与向量检索
      ├── 子块（小）           ← 送向量库检索
      ├── 子块
      └── 子块

命中子块 → 按 `parent_chunk_id` 取父块 → 把父块内容喂给 LLM。

## 数据库怎么表达

`chunks` 表有自引用外键 `parent_chunk_id` 与 `chunk_level`：

- `chunk_level = 0`：叶子块（参与检索）；
- `chunk_level = 1`：父块（**不参与检索**，只为回填）。

检索侧必须加 `chunk_level = 0` 过滤，否则父子块会同时出现在结果里，
同一段内容占两个位置（浪费一半上下文预算，还让用户看到重复引用）。

**注意**：`ChunkRepository.search_vector` 目前没有这个过滤条件——
A11 接入检索时必须补上。这是本策略引入的新约束，记在这里防止遗漏。

## 父块不送向量化

父块内容 = 若干子块的拼接。把父块也向量化意味着同一份文字算两遍
embedding（成本翻倍），而它根本不参与检索。
所以父块的 `embedding` 保持 NULL。

## 父块与子块的位置关系

父块的 `char_start/char_end` 覆盖它所有子块的区间；
子块各自在自己的区间。因此**块的 char_start 会重复/重叠**——
`verify_chunks` 允许"不倒退"，不允许"倒退"，父子同起点是合法的。
"""

from __future__ import annotations

from rag_adapters.chunkers.base import PROTECTED_BLOCK_TYPES, BaseChunker, ChunkDraft
from rag_adapters.chunkers.recursive import RecursiveChunker
from rag_adapters.parsers.artifact import ParsedArtifact

# 父块目标长度 = 子块预算 × 该倍率。
# 取 3 是基于"回填后不超过常见 LLM 上下文的 1/4"的经验值：
# 子块 800 字符 → 父块约 2400 字符；一次回填 3 个父块约 7K 字符，
# 给答案与其它上下文留足空间。
_PARENT_SIZE_RATIO = 3


class ParentChildChunker(BaseChunker):
    """父子切分。

    内部用 `RecursiveChunker` 生成子块（不重复实现切分逻辑），
    然后把相邻子块按预算聚合成父块。
    """

    name = "parent_child"

    def chunk(self, parsed: ParsedArtifact) -> list[ChunkDraft]:
        if not parsed.text.strip():
            return []

        # ---- 1. 先用递归策略切出子块 ----
        # **显式传 config**：子块的预算来自本策略的配置，
        # 不能依赖 RecursiveChunker 的默认值（那样改配置会不生效）。
        child_chunker = RecursiveChunker(self.config)
        children = child_chunker.chunk(parsed)

        if not children:
            return []

        # ---- 2. 把子块按父块预算分组 ----
        parent_budget = self.config.chunk_size * _PARENT_SIZE_RATIO
        groups: list[list[ChunkDraft]] = []
        current: list[ChunkDraft] = []
        current_length = 0

        for child in children:
            # 受保护类型（表格/代码）不参与聚合：父块跨"表格 + 正文"时，
            # 回填的上下文会把表格和无关正文混在一起，反而干扰理解。
            if child.chunk_type in PROTECTED_BLOCK_TYPES:
                if current:
                    groups.append(current)
                    current, current_length = [], 0
                groups.append([child])
                continue

            if current and current_length + len(child.content) > parent_budget:
                groups.append(current)
                current, current_length = [], 0

            current.append(child)
            current_length += len(child.content)

        if current:
            groups.append(current)

        # ---- 3. 生成父块 + 子块（父块在前，便于落库时先拿 ID）----
        drafts: list[ChunkDraft] = []
        index = 0
        for group in groups:
            if len(group) == 1:
                # 单子块成组的场景（超长表格、或文档本身就小）：
                # 不必造父块。父子结构是为了"小检索大回填"，
                # 只有一个子块时父块=子块，造出来只是重复存储。
                child = group[0]
                drafts.append(
                    ChunkDraft(
                        content=child.content,
                        chunk_index=index,
                        chunk_type=child.chunk_type,
                        char_start=child.char_start,
                        char_end=child.char_end,
                        page_no=child.page_no,
                        bbox=child.bbox,
                        section_path=child.section_path,
                        embedding_input=child.embedding_input,
                        chunk_level=0,
                    )
                )
                index += 1
                continue

            parent_index = index
            parent_content = self._join_children(parsed, group)
            parent = ChunkDraft(
                content=parent_content,
                chunk_index=parent_index,
                chunk_type="text",
                char_start=group[0].char_start,
                char_end=group[-1].char_end,
                page_no=group[0].page_no,
                bbox=None,  # 父块跨多块/多页时 bbox 无意义，留空让前端用区间高亮
                section_path=group[0].section_path,
                chunk_level=1,
                chunk_metadata={"child_count": len(group)},
            )
            drafts.append(parent)
            index += 1

            for child in group:
                drafts.append(
                    ChunkDraft(
                        content=child.content,
                        chunk_index=index,
                        chunk_type=child.chunk_type,
                        char_start=child.char_start,
                        char_end=child.char_end,
                        page_no=child.page_no,
                        bbox=child.bbox,
                        section_path=child.section_path,
                        embedding_input=child.embedding_input,
                        parent_index=parent_index,
                        # 叶子块：参与检索
                        chunk_level=0,
                    )
                )
                index += 1

        return drafts

    @staticmethod
    def _join_children(parsed: ParsedArtifact, group: list[ChunkDraft]) -> str:
        """父块正文 = 从首个子块起点到末个子块终点的源文本区间。

        与 structured 同样的理由：只从源文本切片，绝不拼接。
        子块来自 recursive，它们之间可能存在被丢弃的短片段，
        用切片才能覆盖完整原文。
        """
        start = group[0].char_start
        end = group[-1].char_end
        if start is None or end is None:
            # 理论上 recursive 总给位置；缺位置时退化为拼接
            return "".join(child.content for child in group)
        return parsed.text[start:end]


__all__ = ["ParentChildChunker"]
