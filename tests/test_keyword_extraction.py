"""中文关键词提取与预分词测试。

## 为什么这些测试重要

中文 FTS 的失败是**静默**的：分词器不一致 / 停用词误伤，表现都是
"关键词路 0 命中"，查询仍然 200、向量路仍然有结果，只有对比两路
命中数才能发现（ADR 0003 踩过）。

这里把三类约束钉死：

1. 查询与索引用同一分词器（同一输入切出同一组词）；
2. 停用词只做整词/词尾处理，**不做字符级替换**（「有效期」不能被切成「效期」）；
3. 英文/数字标识整体保留且统一小写。
"""

from __future__ import annotations

import pytest

from rag_adapters.retrieval.keyword import (
    extract_keywords,
    extract_keywords_async,
    tokenize_for_index,
)
from rag_core.settings import RetrievalSettings


def test_extracts_chinese_content_words() -> None:
    """只保留实词：疑问词/虚词被丢弃，住宿/报销等保留。"""
    keywords = extract_keywords("二线城市出差住宿每晚最多能报销多少？")

    for expected in ("二线", "城市", "出差", "住宿", "报销"):
        assert expected in keywords
    for stopword in ("多少", "能", "的", "吗"):
        assert stopword not in keywords


def test_does_not_destructively_replace_characters() -> None:
    """**不能做字符级替换**。

    早期实现把「有」全局替换成空格，于是「有效期」变成「效期」——
    关键词看起来还在，实际已经不是用户的词了。
    """
    keywords = extract_keywords("有效期是多久")

    assert "有效期" in keywords
    # 「效期」可以是 jieba 的子词（召回优化），但「有效期」不能被弄丢
    assert keywords[0] == "有效期", f"长词应优先保留，实际：{keywords}"


def test_keeps_identifiers_whole_and_lowercased() -> None:
    """错误码/型号必须整体保留（向量路对 E1024/E1025 几乎不可分）。"""
    keywords = extract_keywords("错误码 E1024 是什么意思")

    assert "e1024" in keywords


def test_empty_query_returns_empty() -> None:
    assert extract_keywords("") == []
    assert extract_keywords("   ") == []
    assert extract_keywords("，。？！") == []


def test_respects_max_keywords_and_prefers_long_terms() -> None:
    keywords = extract_keywords("公司差旅住宿标准报销流程", max_keywords=3)

    assert len(keywords) == 3
    # 按词长降序（长的更具体）
    assert keywords == sorted(keywords, key=len, reverse=True)


@pytest.mark.asyncio
async def test_async_version_matches_sync() -> None:
    query = "二线城市住宿标准"
    assert await extract_keywords_async(query) == extract_keywords(query)


def test_index_text_is_space_separated_and_normalized() -> None:
    """预分词结果必须是空格分隔（`simple` 分词器靠空格切词）。"""
    text = tokenize_for_index("二线城市出差住宿每晚最多可报销 400 元")

    tokens = text.split(" ")
    assert all(tokens), f"出现空 token：{tokens!r}"
    assert "住宿" in tokens
    assert "400" in tokens
    # 停用词不应进入索引
    assert "的" not in tokens


def test_query_and_index_share_tokens() -> None:
    """**跨端一致性**：查询切出的词必须能在索引文本里找到。

    这条断言直接对应"混合检索零命中"的根因：两端分词不一致。
    """
    content = "二线城市出差住宿每晚最多可报销 400 元"
    index_tokens = set(tokenize_for_index(content).split())

    for keyword in extract_keywords("二线城市住宿标准是多少"):
        if keyword in content:
            assert keyword in index_tokens, (
                f"关键词 {keyword!r} 在正文里出现，却不在预分词索引里——查询与索引分词不一致"
            )


def test_index_is_deterministic() -> None:
    """同一文本重复分词结果必须一致（否则索引与查询会随机错位）。"""
    text = "差旅住宿标准与报销流程"
    assert tokenize_for_index(text) == tokenize_for_index(text)


# ---------------------------------------------------------------- 配置校验


def test_retrieval_settings_rejects_impossible_limits() -> None:
    """final_limit 不能超过两路召回总数（否则永远凑不够材料）。"""
    with pytest.raises(ValueError, match="final_limit"):
        RetrievalSettings(vector_limit=1, keyword_limit=1, final_limit=5)


def test_self_check_refuse_requires_self_check_enabled() -> None:
    off = RetrievalSettings(use_self_check=False, self_check_action="refuse")
    on = RetrievalSettings(use_self_check=True, self_check_action="refuse")

    assert off.self_check_refuse is False
    assert on.self_check_refuse is True
