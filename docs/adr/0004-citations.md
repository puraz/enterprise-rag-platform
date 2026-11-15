# A12 引用对齐：定位算法与"诚实报告失败"

> 日期：2026-11-15　环境：bge-m3 + DeepSeek + PyMuPDF
> 复现：`uv run python tests/eval/verify_citations.py --count 20`

## 1. 验收结果

| 指标 | 结果 |
|---|---|
| 答案数 | 20 |
| 引用总数 | 21 |
| quote 在原文中 | **100%** |
| 页码正确 | **100%** |
| 多页 PDF 页码专项 | **4/4 正确** |

## 2. 引用契约（`AnswerWithCitations` / `Citation`）

字段设计的三条依据：

1. **`page_no` 而不是 `page`**：`page` 含糊（页码？页面对象？），
   `page_index` 暗示 0-based。`page_no` + `ge=1` 明确"从 1 开始"。
2. **必须带 `quote`**：只给坐标时，若解析器没给 bbox（DOCX/XLSX）
   或页面旋转导致坐标偏移，前端完全无法高亮。`quote` 让前端能
   "坐标优先，失败则全文搜索"，也是人工审计"引用是否错配"的锚点。
3. **`relevance_score` 保持"距离"语义**（0 最相关）：不在契约层
   转成相似度。两者语义相反，转换点越多越容易把排序搞反。

`citations` 的 `index` 校验：**必须从 1 开始连续、不跳号、不重复**。
跳号会让前端按 index 找引用时拿到错条目，且不报错。

## 3. 定位算法：从"稀有 token 命中"到"窗口"

答案通常**不包含**原文原句（LLM 会改写），所以不能用精确子串匹配。

算法：提取答案的关键词 → 在 chunk 上以关键词位置为中心开窗 →
找覆盖关键词最多的窗口。

### 三次迭代（每次都是被测试打回来的）

**v1：固定窗口 200 字符 + 覆盖率阈值**
失败：短 chunk（60 字）上窗口永远是全文，定位退化为"整块"，
页码永远是起始页。实测跨页 chunk 的引用总显示起始页。

**v2：自适应窗口 + token 覆盖率**
失败：答案必然含原文没有的词（"发邮件" vs "报告渠道"），
这些词拉低覆盖率，正常改写被判为定位失败（两条测试全挂）。

**v3：稀有 token 命中（最终方案）**
- 窗口大小按 `max(20, 片段长度 × 1.5) min chunk长度` 自适应；
- 判据只看**稀有 token**（chunk 内出现 ≤2 次）是否落在窗口内；
- 高频词（"城市""不超过"）即使命中也不提供定位信息，不计入。

### 一个关键补充：英文停用词

"稀有"按 chunk 内频次定义有致命缺陷：**短 chunk 里所有词都稀有**。

实测误判：答案"claims must be filed within 30 days" 与 chunk
"travel request must be approved before booking" 只共享
"must" 与 "be"，它们在该 chunk 里各出现 1 次 → 判为稀有命中
→ 定位成功 → 引用跳到完全错误的页。

修复：加英文停用词表（`must`/`be`/`the`/`policy`/`within` 等 80+ 词），
这类通用词不参与定位判据。修复后同一用例返回 None（正确）。

## 4. 最重要的一条：定位失败时**不给页码**

最初的设计是"定位失败就用整个 chunk 的位置"（含 `page_no`），
理由是"至少有东西可高亮"。

实测证明这是错的。4 页 PDF、每页一个 chunk 的场景：

| 引用内容 | 实际页 | 原设计的报出页 |
|---|---|---|
| 第 3 页的"30 days" | 3 | **1** ✗ |
| 第 4 页的"VP approval" | 4 | **1** ✗ |

用户会被送到第 1 页，在一个明显不相关的位置找依据——
**比"无法跳转"更糟**。

现在的行为：

```python
LocationResult(
    located=False,      # 告诉上层这是降级结果
    page_no=None,       # 无法确定在哪一页
    char_start=...,     # chunk 的区间仍是真实信息
    quote=...,          # 整块内容，前端可退化为"显示原文"
)
```

**宁可说"我不知道在哪一页"，也不要给一个错的页码。**

## 5. 引用错配的检测价值

定位失败（`locate_failed`）不只是"降级信号"，它是**引用错配的探测器**：

- 模型说"这个结论来自 [1]"，但 [1] 里根本没有相关内容；
- 上下文压缩/父块回填引入了不属于该引用的内容。

统计 `locate_success_rate` 能反映"模型引用是否靠谱"。
A12 的 20 条验收里 `locate_failed` 出现过 2 次（10%），
都发生在 LLM 大幅改写的场景——值得持续观察但不构成阻塞。

## 6. 跨页 chunk 的页码映射

`chunk.page_no` 只记录**起始页**。跨页 chunk 若答案引用后半段，
用起始页会让用户翻错页。

解法：用 `pages` 的字符区间把"定位到的偏移"映射成页码：

```python
page_no = page_for_offset(pages, absolute_start) or material.page_no
```

`pages` 来自解析产物的 `PageInfo`（A8 已落盘）。
实测跨页用例页码正确（见 `test_multi_page_citation_uses_content_page_not_chunk_start`）。

## 7. 遗留

1. **`pages` 在生产链路里没传到引用组装**：`Material.pages` 字段存在，
   但评估脚本里是空的（评估语料是单页 Markdown），worker 侧写入的
   chunk 也没把 pages 带出来。A13/A14 接入问答接口时需要从
   `document_versions.parse_metadata` 读页信息并传入。
2. **`bbox` 未参与定位**：目前只给引用带上 chunk 的 bbox（整个块的框），
   不是"引用那句"的精确框。精确框需要解析器给 span 级 bbox，
   PyMuPDF 能做到，但当前解析器只给 block 级——留给后续优化。
3. **引用越界只丢弃不报错**：模型引用 `[5]` 但只有 3 条材料时，
   丢弃该引用、保留答案。管理端应能看到这个计数（`AssemblyStats.out_of_range`）。
