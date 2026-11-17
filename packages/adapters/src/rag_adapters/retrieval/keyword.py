"""中文关键词检索的文本处理：分词、停用词、关键词提取。

## 为什么用 jieba，而不是数据库扩展

中文没有空格，PostgreSQL 内置的 `simple` 分词器会把整句当成一个 token
（`to_tsvector('simple', '二线城市住宿标准')` 只有一个词项），
所以 `to_tsquery('simple', '二线')` 永远匹配不到。ADR 0003 因此得到
"关键词路零命中、混合检索零收益"的错误结论——不是方法无效，是前置条件缺失。

可选方案与实际取舍：

| 方案 | 结论 |
|---|---|
| `zhparser` / `pg_bigm` | 当前 PG 18 未提供，需要编译 SCWS/扩展；运维成本高 |
| `pg_trgm` 子串匹配 | 可用，但只是"包含"匹配：没有词项、没有词频、中文短词用不上索引 |
| ParadeDB `pg_search` | 真 BM25，但需要安装 Rust 扩展；本机/多数托管 PG 不支持 |
| **jieba 预分词 + 内置 FTS** | 纯 Python、零扩展；按空格切词后即词项索引 + ts_rank |

本项目选择最后一种：

- **写入时**：`tokenize_for_index(content)` 把正文切成空格分隔的词，
  存进 `chunks.search_text`；数据库用生成列
  `to_tsvector('simple', search_text)` 建 GIN 索引。
- **查询时**：`extract_keywords(query)` 用同一套分词器提取关键词，
  构造 `to_tsquery('simple', '词1 | 词2')`，按 `ts_rank` 排序。

两端必须用**同一个分词器**：查询切出"住宿"，索引里却是"住宿标准"，
仍然匹配不上——这是中文 FTS 最常见的静默失败。

## 为什么不用 jieba 的默认词性/自定义词典

当前只需要"把句子切成实词"。词性过滤（去掉动词/副词）会误伤
"报销""缴纳"这类关键动作词，而自定义词典需要业务词典维护流程。
等有真实查询日志、能统计漏召回后再加，不提前引入维护负担。

## 首次调用的初始化成本

jieba 首次 `cut` 会加载前缀词典（约 0.5~1s）。调用方（API 的检索闭包）
通过 `extract_keywords_async` 把它放进线程池，避免阻塞事件循环；
worker 侧在入库时同样用 `to_thread` 批量处理。
"""

from __future__ import annotations

import asyncio
import logging
import re
import warnings
from collections.abc import Iterable

# jieba 0.42 在 Python 3.12 下会报 SyntaxWarning（正则里的 `\.`）。
# 只屏蔽该类警告，不影响其它警告；同时把它的初始化日志降到 WARNING，
# 否则首次检索会在访问日志里插入一行“Building prefix dict”。
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import jieba

jieba.setLogLevel(logging.WARNING)

# 查询与索引共用的停用词。
#
# 这些词在几乎所有文档里都出现，留着只会制造"命中"假象并稀释 ts_rank。
# **只做整词匹配**：不做字符级替换（早期实现把「有」全局替换成空格，
# 会把「有效期」变成「效期」——静默破坏关键词，且很难发现）。
_STOPWORDS = frozenset(
    {
        # 中文虚词/代词
        "的",
        "了",
        "是",
        "在",
        "和",
        "与",
        "及",
        "或",
        "而",
        "也",
        "就",
        "都",
        "很",
        "更",
        "最",
        "可",
        "能",
        "会",
        "要",
        "有",
        "对",
        "从",
        "到",
        "为",
        "以",
        "于",
        "之",
        "其",
        "这",
        "那",
        "等",
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "这个",
        "那个",
        "这些",
        "那些",
        # 疑问/指令词
        "什么",
        "多少",
        "怎么",
        "怎样",
        "如何",
        "哪些",
        "哪个",
        "为什么",
        "为何",
        "是否",
        "能否",
        "可以",
        "需要",
        "多久",
        "几天",
        "时候",
        "请问",
        "请",
        "帮",
        "帮我",
        "告诉",
        "告诉我",
        "吗",
        "呢",
        "吧",
        # 英文停用词
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "what",
        "how",
        "why",
        "when",
        "which",
        "who",
        "do",
        "does",
        "did",
        "be",
        "been",
        "with",
        "by",
        "at",
        "as",
        "it",
    }
)

# 有意义的字符：中文、字母、数字。纯标点/空白直接丢弃。
_MEANINGFUL_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

# 英文/数字标识（错误码、型号、分机）。jieba 有时会把 `E1024`
# 拆成 `E` + `1024`，所以额外用正则兜一层，整体保留。
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_MAX_TOKEN_LENGTH = 32

# 单字停用词。只用于剥离词首/词尾（不为 1 时不动）：
# jieba 的 HMM 会把["最多能"]当成一个词，但正文里写的是"最多可"。
# 把尾部的"能"剥掉后，“最多”才能在索引与查询两侧稳定对齐。
_SINGLE_CHAR_STOPWORDS = (
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "及",
    "或",
    "而",
    "也",
    "就",
    "都",
    "很",
    "更",
    "最",
    "可",
    "能",
    "会",
    "要",
    "有",
    "对",
    "从",
    "到",
    "为",
    "以",
    "于",
    "之",
    "其",
    "这",
    "那",
    "等",
    "吗",
    "呢",
    "吧",
)


def _strip_stopword_affixes(token: str) -> str:
    """剥离词尾的单字停用词（每次至少保留 2 字）。

    **只剥词尾，不剥词首**：剥词首会把「有效期」变成「效期」、
    把「最多」变成「多」——这是静默破坏，比不剥更糟。

    为什么需要词尾剥离：jieba 的 HMM 会把「最多能」合成一个词，
    而正文里写的是「最多可」；剥掉尾部「能」后两侧才能对齐。
    """
    changed = True
    while changed:
        changed = False
        for word in _SINGLE_CHAR_STOPWORDS:
            if token.endswith(word) and len(token) - 1 >= 2:
                token = token[:-1]
                changed = True
                break
    return token


def _is_meaningful(token: str) -> bool:
    """是否是可用于检索的词项。

    规则：
    - 必须含中文或字母数字（丢弃标点/空白）；
    - 不在停用词表里；
    - 中文单字不丢（「税」「率」有意义），但停用词里的单字已排除；
    - 超长 token 截断到 32 字符（防止异常输入撑大索引）。
    """
    if not token or token in _STOPWORDS:
        return False
    if not _MEANINGFUL_PATTERN.search(token):
        return False
    return len(token) <= _MAX_TOKEN_LENGTH


def _normalize(token: str) -> str:
    """英文小写化；中文保持原样。

    查询与索引必须用同样的归一化，否则 `E1024` 与 `e1024` 互不匹配。
    """
    return token.lower() if token.isascii() else token


def _cut(text: str) -> list[str]:
    """分词并过滤。

    `cut_for_search` 会额外产出长词的子词（"一线城市" → 一线/城市/一线城市），
    对召回有利：用户问"一线"时也能命中"一线城市"。
    """
    tokens: list[str] = []
    for raw in jieba.cut_for_search(text):
        token = _strip_stopword_affixes(_normalize(raw))
        if _is_meaningful(token):
            tokens.append(token)
    return tokens


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def tokenize_for_index(text: str) -> str:
    """把正文切成空格分隔的词项，供 PostgreSQL `simple` 分词器建索引。

    **必须与 `extract_keywords` 用同一个分词器与同一套停用词**：
    写入时切成"住宿标准"，查询时切成"住宿"，才能互相匹配。

    返回值直接写进 `chunks.search_text`；生成列会把它转成 tsvector。
    空输入返回空串（调用方入库时用空串，不写入 NULL）。
    """
    if not text:
        return ""
    return " ".join(_dedupe(_cut(text)))


def extract_keywords(query: str, *, max_keywords: int = 8) -> list[str]:
    """从查询里提取关键词（去重、去停用词、数量封顶）。

    Args:
        query: 用户问题（原始或独立化后均可）。
        max_keywords: 上限，默认 8。关键词之间是 OR 关系，
            词越多匹配越宽、`ts_rank` 越没有区分度，且 SQL 越慢。
            实测 12+ 个关键词把延迟从 6ms 推到 650ms 而召回无提升。

    Returns:
        关键词列表（按词长从长到短，长的更具体）。空列表表示没有可用词项，
        调用方应跳过关键词检索路。
    """
    if max_keywords <= 0 or not query.strip():
        return []

    tokens = _cut(query)

    # 英文/数字标识整体优先（精确匹配的价值最高，且 jieba 可能拆开）
    identifiers = [
        _normalize(match.group(0).strip("._-")) for match in _IDENTIFIER_PATTERN.finditer(query)
    ]
    candidates = _dedupe([token for token in identifiers if _is_meaningful(token)] + tokens)

    # 按词长降序：长词更具体（"住宿标准"优于"住宿"），
    # 截断/抽样时先保留长词。同长度保持原顺序（稳定）。
    ranked = sorted(enumerate(candidates), key=lambda pair: (-len(pair[1]), pair[0]))
    return [token for _index, token in ranked[:max_keywords]]


async def extract_keywords_async(query: str, *, max_keywords: int = 8) -> list[str]:
    """`extract_keywords` 的异步版本。

    jieba 首次调用会同步加载词典（约 0.5~1s）。在事件循环里直接调用
    会让**所有并发请求**一起卡住；放到线程池后只阻塞调用方自己的协程。
    """
    return await asyncio.to_thread(extract_keywords, query, max_keywords=max_keywords)


__all__ = ["extract_keywords", "extract_keywords_async", "tokenize_for_index"]
