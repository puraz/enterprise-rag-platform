"""引用定位：把"答案引用了第 N 条材料"变成"原文的哪一页、哪个区间"。

## 为什么这件事不简单

材料（chunk）→ 原文位置看起来是直接拷贝：
`chunk.page_no` / `char_start` / `char_end` 就在数据库里。

但有三个真实问题：

### 1. chunk 可能跨页

一个 chunk 从第 2 页末尾延续到第 3 页开头。`chunk.page_no` 只记了
**起始页**。如果答案引用的是这个 chunk 后半段（第 3 页的内容），
前端跳到第 2 页会让用户找不到。

解决：用 `pages` 表（char_start/char_end → page_no）把 chunk 的
**字符区间**映射成页码。区间跨页时，取答案引用的片段所在的页。

### 2. 答案引用的不是整个 chunk，而是 chunk 里的一小段

LLM 说"住宿上限 600 元 [1]"，实际只用了 chunk 里的一句话。
整个 chunk 高亮 500 字，用户找不到重点。

解决：**在 chunk 内定位答案片段**。方法是从答案里提取关键词，
在 chunk 里找最匹配的窗口。

### 3. 模型可能记错来源

答案里的 `[1]` 说的内容其实来自 `[2]`。这时强行定位会得到
"引用了但找不到匹配文本"——**这本身就是有价值的信号**（说明引用错配）。

解决：定位失败时返回 `located=False`，让上层决定
（降级为整个 chunk 高亮 / 标记引用可疑 / 拒答）。

## 算法选择：为什么不用精确子串匹配

答案通常**不包含**原文原句——LLM 会改写、压缩、合并。
"每晚不超过 600 元" vs 原文"一线城市每晚不超过 600 元，二线城市…"

所以要用**关键词覆盖度**：把答案片段切成关键词（中文按 2-3 字切），
在 chunk 上滑动窗口，找覆盖关键词最多的窗口。

这与 A11 关键词检索用的是同一类启发式——**不是好的中文分词，
但在没有分词器时是最合理的近似**，且行为可解释、可测试。

## 为什么不引入向量相似度定位

技术上可以用 embedding 在 chunk 内找语义最相似的句子。
但那需要额外一次模型调用（每次引用一次），延迟与成本都不可接受。

关键词覆盖度在"引用定位"这个任务上够用：我们要找的是
"答案提到的那句话在哪"，而答案与原文的**用词重合度**通常很高
（专有名词、数字、条款几乎必然重复）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageSpan:
    """一页在文档纯文本中的字符区间（对应 `document_versions` 的页信息）。"""

    page_no: int
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class LocationResult:
    """定位结果。"""

    located: bool
    """是否找到了可靠的位置。False 表示需要降级（整 chunk 高亮或标记可疑）。"""

    char_start: int | None = None
    char_end: int | None = None
    page_no: int | None = None
    quote: str = ""
    """用于高亮/审计的原文片段。定位失败时是整个 chunk 的开头。"""

    reason: str = ""
    """定位失败的原因（进日志，便于排查"为什么引用高亮不准"）。"""


# 中文停用词与无信息量的高频词。
#
# 去掉它们的原因：这些词在任何文档里都大量出现，用来做定位
# 会让"窗口覆盖度"对所有位置都接近满分，失去区分度。
_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "是",
        "在",
        "和",
        "与",
        "或",
        "等",
        "及",
        "为",
        "对",
        "从",
        "到",
        "按",
        "由",
        "以",
        "之",
        "其",
        "该",
        "此",
        "这",
        "那",
        "有",
        "无",
        "不",
        "也",
        "都",
        "就",
        "而",
        "但",
        "则",
        "若",
        "如",
        "可以",
        "需要",
        "应当",
        "必须",
        "根据",
        "进行",
        "相关",
        "上述",
        "如下",
    }
)

# 英文停用词。**必须单独列出**：中文停用词表管不到英文，
# 而英文里 "must"/"be"/"within" 这类词与中文的"的""了"等价——
# 它们在每个 chunk 里都出现，匹配上它们完全不说明"定位对了"。
#
# 实测过的误判：答案"claims must be filed within 30 days" 与
# chunk "travel request must be approved before booking" 只共享
# "must" 与 "be"，却被判为定位成功——因为它们是仅有的两个
# "在本 chunk 内出现次数 ≤2" 的 token（短 chunk 里所有词都稀有）。
_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "need",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "without",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "as",
        "so",
        "such",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "i",
        "my",
        "not",
        "no",
        "yes",
        "all",
        "any",
        "some",
        "more",
        "most",
        "other",
        "within",
        "about",
        "into",
        "over",
        "under",
        "between",
        "through",
        "before",
        "after",
        "during",
        "above",
        "below",
        "up",
        "down",
        "out",
        "also",
        "just",
        "only",
        "very",
        "same",
        "each",
        "every",
        "both",
        "policy",
        "policies",
        "state",
        "states",
        "state's",
        "according",
    }
)


# 窗口大小：定位窗口至少这么大，避免"只匹配到 2 个字"就当成命中。
_MIN_WINDOW = 20
# 窗口最多这么大：超过这个长度的"引用片段"对高亮没有意义（用户看不完）。
_MAX_WINDOW = 200


def _window_size(chunk_length: int, fragment_length: int) -> int:
    """按 chunk 与片段的实际长度决定窗口大小。

    ## 为什么不能固定 _MAX_WINDOW

    固定窗口在短 chunk 上会退化为"整块"：chunk 只有 60 字时，
    任何锚点的窗口都是全文 → 得分相同 → 返回 (0, 60) →
    高亮整块、页码永远取起始页。

    实测症状：跨页 chunk 的引用总是显示起始页（用户翻错页），
    因为"定位"实际上没有定位。

    规则：窗口至少覆盖片段的 1.5 倍（留上下文），但不超过 chunk 长度。
    """
    desired = max(_MIN_WINDOW, int(fragment_length * 1.5))
    return min(_MAX_WINDOW, max(_MIN_WINDOW, min(desired, chunk_length)))


def _tokenize(text: str) -> list[str]:
    """把文本切成定位用的 token。

    **保留数字与英文串整体**（"600"、"security@example.com"、"8888"）：
    这些是定位最有价值的锚点——它们几乎不会在文档里重复出现。

    中文按 2~3 字滑窗切：没有分词器时的合理近似。
    单字不切（区分度太低，"的""在"到处都是）。
    """
    tokens: list[str] = []

    # 先提取英文/数字串（整体保留，不拆）
    for match in re.finditer(r"[A-Za-z0-9@._\-]+", text):
        token = match.group(0)
        if len(token) < 2:
            continue
        lowered = token.lower()
        # 中文停用词表也可能含英文（"可以"这类没有），主要看英文表
        if lowered in _ENGLISH_STOPWORDS or lowered in _STOPWORDS:
            continue
        # 纯数字短串（1~9）区分度太低，跳过；"30""600"这类保留
        if token.isdigit() and len(token) < 2:
            continue
        tokens.append(token)

    # 再处理中文：去掉非中文、英文、数字的字符后按 2~3 字滑窗
    chinese_only = re.sub(r"[^\u4e00-\u9fff]", " ", text)
    for segment in chinese_only.split():
        # 去掉停用词（按最长匹配优先，避免"可以将"被切成"可以"+"将"）
        for word in sorted(_STOPWORDS, key=len, reverse=True):
            segment = segment.replace(word, " ")
        for piece in segment.split():
            if len(piece) < 2:
                continue
            # 2 字与 3 字都切，短片段更容易命中、长片段更精确
            for size in (2, 3):
                for start in range(0, max(1, len(piece) - size + 1)):
                    window = piece[start : start + size]
                    if len(window) == size:
                        tokens.append(window)

    # 去重但保留顺序（顺序影响后续的"首选 token"）
    return list(dict.fromkeys(tokens))


def locate_in_chunk(
    *,
    answer_fragment: str,
    chunk_text: str,
    chunk_char_start: int,
) -> tuple[int, int] | None:
    """在 chunk 内定位答案片段，返回相对 chunk 的 `(start, end)`。

    Args:
        answer_fragment: 答案里对应这条引用的文字（可能被 LLM 改写）。
        chunk_text: chunk 的正文。
        chunk_char_start: chunk 在文档全文中的起始偏移（仅用于长度校验，未使用）。

    Returns:
        相对偏移区间；定位失败返回 None。

    ## 算法

    1. 提取答案片段的关键词（数字/英文整串优先，中文 2~3 字滑窗）；
    2. 在 chunk 上以每个关键词的出现位置为候选窗口起点；
    3. 对每个候选窗口计算"覆盖了多少不同关键词"；
    4. 取覆盖度最高的窗口；覆盖度低于阈值则判失败。

    **为什么以关键词出现位置为起点**：文本是连续的，答案提到的内容
    在原文里也连续。以关键词为锚点能快速锁定候选区，避免 O(n²) 扫描
    所有可能窗口。
    """
    if not answer_fragment.strip() or not chunk_text.strip():
        return None

    tokens = _tokenize(answer_fragment)
    if not tokens:
        return None

    chunk_length = len(chunk_text)
    window_size = _window_size(chunk_length, len(answer_fragment))

    # 每个 token 在 chunk 里的所有出现位置
    positions: dict[str, list[int]] = {}
    for token in tokens:
        found: list[int] = []
        start = 0
        while True:
            index = chunk_text.find(token, start)
            if index == -1:
                break
            found.append(index)
            start = index + 1
        if found:
            positions[token] = found

    if not positions:
        return None

    # 粗筛：答案里至少要有 20% 的 token 在 chunk 里出现过。
    #
    # 这是"这段答案与该 chunk 是否扯得上关系"的快速判断。
    # 阈值低是因为答案经过改写会引入原文没有的词，要求高覆盖
    # 等于要求答案几乎照抄原文。
    #
    # 真正的定位判据是后面的"稀有 token 命中"，这条只是提前排除
    # "几乎完全无关"的情况（省掉后续窗口计算）。
    coverage = len(positions) / len(tokens)
    if coverage < 0.2:
        return None

    # 用"最稀有"的 token 作为锚点：它出现次数最少，定位最准。
    # 例如答案里有"600"（chunk 里出现 1 次）和"城市"（出现 8 次），
    # 用"600"作为窗口中心能直接锁定位置。
    anchor_token = min(positions, key=lambda token: len(positions[token]))
    anchor_positions = positions[anchor_token]

    best_start = 0
    best_end = min(window_size, chunk_length)
    best_score = -1.0

    for anchor in anchor_positions:
        # 窗口以锚点为中心向两侧扩展
        window_start = max(0, anchor - window_size // 2)
        window_end = min(chunk_length, window_start + window_size)
        # 靠近末尾时向前补足窗口长度（保证窗口不小于 _MIN_WINDOW）
        if window_end - window_start < _MIN_WINDOW:
            window_start = max(0, window_end - _MIN_WINDOW)

        window_text = chunk_text[window_start:window_end]

        # 窗口得分：覆盖的 token 数（用不同 token 计数，不重复计）
        covered = sum(1 for token in tokens if token in window_text)
        score = covered / len(tokens)

        if score > best_score:
            best_score = score
            best_start = window_start
            best_end = window_end

    # ## 最终判据：**稀有 token 命中数**
    #
    # 走过的弯路（记录下来避免重犯）：
    #
    # 1. "匹配 token 数 / 答案 token 数" —— 答案必然含原文没有的词
    #    （"发邮件" vs "报告渠道"），这些词永远匹配不上，把正常的
    #    改写判定为定位失败（实测两条用例全挂）。
    # 2. "IDF 加权覆盖率"（分母含所有 token）—— 不匹配的 token
    #    仍占分母权重，8 个不匹配词就把 2 个高价值匹配词淹没了。
    #
    # 正确的判据：**只看稀有 token 是否命中**。
    #
    # 定义"稀有" = 在 chunk 内出现 ≤ 2 次的 token（含数字、英文串、
    # 以及"三线""600"这类具体词）。这类 token 一旦命中，就足以
    # 锁定位置——因为它们不会在文档里到处出现。
    #
    # 而"城市""不超过"这类高频词即使命中也不提供定位信息，
    # 把它们计入判据只会稀释信号。
    rare_tokens = {
        token
        for token, occurrences in positions.items()
        if len(occurrences) <= 2 and len(token) >= 2
    }
    if not rare_tokens:
        # 所有匹配上的 token 都是高频词：无法可靠定位。
        # 典型场景：答案与 chunk 只共享"公司""规定"这类通用词。
        return None

    # 至少要有 1 个稀有 token 落在最佳窗口内
    best_window_text = chunk_text[best_start:best_end]
    rare_hits = sum(1 for token in rare_tokens if token in best_window_text)
    if rare_hits == 0:
        return None

    # 收紧窗口：去掉首尾空白，让高亮区间更贴内容
    while best_start < best_end and chunk_text[best_start].isspace():
        best_start += 1
    while best_end > best_start and chunk_text[best_end - 1].isspace():
        best_end -= 1

    if best_end <= best_start:
        return None

    return best_start, best_end


def page_for_offset(pages: Sequence[PageSpan], offset: int) -> int | None:
    """把文档全文中的字符偏移映射成页码。

    Args:
        pages: 按 page_no 升序的页区间。
        offset: 文档全文中的字符偏移。

    Returns:
        页码（1-based）；无法映射时返回 None。

    ## 为什么必须用"字符偏移 → 页"而不是直接用 chunk.page_no

    `chunk.page_no` 只记录**起始页**。一个跨页 chunk（第 2 页末尾
    延续到第 3 页）如果答案引用了后半段，用起始页会让用户翻错页。

    用偏移映射能精确定位到"这段文字在第几页"。
    """
    for page in pages:
        if page.char_start <= offset < page.char_end:
            return page.page_no
    # 边界情况：偏移正好落在页末尾（char_end 是开区间）
    for page in pages:
        if offset == page.char_end:
            return page.page_no
    return None


def build_quote(text: str, *, start: int, end: int, max_length: int) -> str:
    """从窗口区间生成引用片段（截断到上限）。

    截断保留**开头**：开头通常包含关键信息（"一线城市…600 元"），
    截尾比截头更不容易丢重点。
    """
    fragment = text[start:end].strip()
    if len(fragment) <= max_length:
        return fragment
    # 截断处补省略号，让读者知道被截断了
    return fragment[: max_length - 1] + "…"


__all__ = [
    "LocationResult",
    "PageSpan",
    "build_quote",
    "locate_in_chunk",
    "page_for_offset",
]
