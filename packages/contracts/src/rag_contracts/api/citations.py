"""引用契约：答案里的每个"来源"用什么形状表达。

## 为什么引用是契约而不是内部对象

同一个"引用"要满足三个消费者的不同需求：

| 消费者 | 需要什么 |
|---|---|
| 前端 | 点击跳到原文 → 页码、字符区间、高亮文本 |
| 管理端审计 | 核查依据 → 文档 ID、chunk ID、相关性分数 |
| 后端校验 | 防止越权引用 → 文档 ID 必须在可见集内 |

如果各端自己拼一个形状，会出现"前端要 `page`、后端给 `page_no`"
这类漂移——而且只在联调时暴露。所以引用的形状由契约定义。

## 字段命名遵循 AGENTS.md 4.1

- `snake_case`、主键 `id`、时间 `*_at`、布尔 `is_*`/`has_*`；
- 页码字段叫 `page_no`（从 1 开始），不是 `page`（含糊）也不是 `page_index`（0-based 暗示）。

## 为什么引用要带 `quote`（原文片段）

前端高亮有两个信息来源：
1. **坐标**（page_no + bbox / char 区间）——精确，但依赖解析质量；
2. **原文片段**——模糊，但**永远能用**。

只给坐标时，如果解析器没给 bbox（DOCX/XLSX）或页面旋转导致坐标偏移，
前端就完全无法高亮。带上 `quote` 让前端可以"坐标优先，失败则全文搜"。

`quote` 也是**人类审计的锚点**：审计员看到"引用说 YES，
但原文这句其实说的是 NO"时，才能发现引用错配。

## 为什么 `quote` 要限长

引用片段会进响应体，也可能进日志。不限制长度的话，
一份引用可能携带整个 chunk（几百字），10 条引用就是几千字——
响应体膨胀，且 `quote` 可能包含不该出现在日志里的内容。

## 置信度（`relevance_score`）为什么用"距离"而不是"相似度"

底层检索产出的是余弦距离（越小越近）。**不要在契约层转换**：
转换会让"这个分数到底是距离还是相似度"变得需要推理，
而两者语义相反，搞混就是排序反了。

契约里统一用 `relevance_score` + 明确说明"**0 最相关、2 最不相关**"，
并在字段描述里写清方向。前端展示时可以自己转成"相似度 %"。
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 引用片段的长度上限。超出会被截断（保留开头——开头通常包含关键信息）。
MAX_QUOTE_LENGTH = 300


class Citation(BaseModel):
    """一条引用：答案里的某个结论来自原文的哪个位置。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(
        ge=1,
        description="引用编号（从 1 开始）。答案正文里的 `[1]` 对应这个字段。",
    )
    document_id: uuid.UUID = Field(description="来源文档 ID")
    document_title: str = Field(description="文档标题（前端展示用，避免再查一次库）")
    chunk_id: uuid.UUID = Field(description="来源 chunk ID")
    chunk_index: int = Field(
        ge=0,
        description="chunk 在文档版本内的序号。引用可能跨多个 chunk 时按最小序号记。",
    )

    # ---- 定位信息（前端高亮用）----
    page_no: int | None = Field(
        default=None,
        ge=1,
        description="页码（**从 1 开始**，与用户看到的页码一致）。非分页格式为 null。",
    )
    char_start: int | None = Field(
        default=None, ge=0, description="在文档纯文本中的起始字符偏移（含）"
    )
    char_end: int | None = Field(
        default=None, ge=0, description="在文档纯文本中的结束字符偏移（不含）"
    )
    bbox: dict[str, float] | None = Field(
        default=None,
        description=(
            "版面坐标 {x0,y0,x1,y1}，单位 PDF point。"
            "只有 PDF 类画布式文档有；前端据此在原图上框选。"
        ),
    )
    section_path: list[str] = Field(
        default_factory=list,
        description="章节路径（如 ['第三章', '3.2 报销流程']），用于面包屑展示",
    )

    # ---- 内容与元信息 ----
    quote: str = Field(
        max_length=MAX_QUOTE_LENGTH,
        description=(
            "引用处的原文片段（截断到 300 字符）。"
            "坐标缺失或不准时，前端可据此在全文里搜索定位；"
            "也是人工审计「引用是否错配」的锚点。"
        ),
    )
    relevance_score: float | None = Field(
        default=None,
        description=(
            "相关性分数，**余弦距离语义：0 最相关，2 最不相关**（越小越好）。"
            "不在此处转成相似度——两者语义相反，转换点越多越容易把排序搞反。"
        ),
    )

    @field_validator("char_start", "char_end")
    @classmethod
    def _non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("字符偏移不能为负")
        return value

    def has_position(self) -> bool:
        """是否有可用于定位的信息。

        `quote` 不算"定位信息"——它是兜底手段（全文搜索），
        在没有坐标时可用，但不如坐标精确。
        """
        return self.page_no is not None or (
            self.char_start is not None and self.char_end is not None
        )


class AnswerWithCitations(BaseModel):
    """带引用的答案。

    这是**问答接口的成功响应体**（`POST /api/v1/chat` 之类）。

    为什么 `citations` 可能是空列表：
    - 模型拒答（"根据现有资料无法回答"）；
    - 答案不需要引用（如"你好"这类寒暄）。

    前端不能假设 `citations` 非空。**拒答时 `refused=True`**，
    让前端能区分"拒答"与"回答了但没有引用"——这两者的 UI 处理完全不同。
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="答案正文。其中的 `[n]` 对应 citations 里的 index。")
    refused: bool = Field(
        default=False,
        description=(
            "是否为拒答（材料不足，没有编造）。"
            "**与「citations 为空」不等价**：答案可以没有引用但不是拒答。"
        ),
    )
    citations: list[Citation] = Field(default_factory=list, description="引用列表，按 index 升序")
    degraded: bool = Field(
        default=False,
        description=(
            "是否走了降级路径（如 reranker 挂→只用 RRF、主模型挂→备用模型）。"
            "降级时前端应给用户明确提示，不能悄悄降级。"
        ),
    )

    @field_validator("citations")
    @classmethod
    def _citations_sorted_and_unique(cls, value: list[Citation]) -> list[Citation]:
        """引用必须按 index 升序且不重复。

        不校验的话，前端按 index 找引用会拿到错的条目
        （比如 `[2]` 指向了第 3 条引用），而且不会有任何报错。
        """
        indexes = [citation.index for citation in value]
        if indexes != sorted(indexes):
            raise ValueError("citations 必须按 index 升序")
        if len(set(indexes)) != len(indexes):
            raise ValueError("citations 的 index 不能重复")
        if indexes and indexes[0] != 1:
            raise ValueError("citations 的 index 必须从 1 开始连续")
        if indexes and indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("citations 的 index 必须连续（不允许跳号）")
        return value


__all__ = ["MAX_QUOTE_LENGTH", "AnswerWithCitations", "Citation"]
