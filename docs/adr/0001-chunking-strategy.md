# A9 分块策略选型结论

> 日期：2026-11-15　文档：`/tmp/handbook.md`（员工手册，1090 字符、5 章、7 段）
> 参数：`chunk_size=400`、`chunk_overlap=50`、`min_chunk_size=50`
> 复现：`uv run python scripts/chunk_preview.py --markdown <doc> --all-strategies --size 400 --overlap 50 --out cmp.html`

## 1. 量化对比

| 策略 | 块数 | 最短 | P90 | 中位 | 覆盖率 | 重叠率 | 校验 |
|---|---|---|---|---|---|---|---|
| `fixed` | 4 | 203 | 380 | 328 | 100% | 13.9% | OK |
| `recursive` | 15 | 54 | 92 | 66 | 100% | 0% | OK |
| `structured` | 5 | 160 | 264 | 218 | 100% | 0% | OK |
| `sentence_window` | 15 | 54 | 92 | 66 | 100% | 0% | OK |
| `parent_child` | 16 | 54 | 104 | 70 | 100% | 0% | OK |
| `semantic` | 15 | 54 | 92 | 66 | 100% | 0% | **降级** |

`semantic` 在预览脚本里没有注入 embedding，**降级为 recursive**——
数据与 recursive 完全一致是降级的证据，不是"两种策略效果相同"。
真实语义切分的效果需要带模型跑（`SemanticChunker(embed_fn=...)`），
本步骤未做，原因见第 4 节。

## 2. 结论

**默认策略：`recursive`**。理由：

- 与 `fixed` 相比：块长度受控（P90 92 vs 380）、无重叠浪费（0% vs 13.9%）、
  切点落在句子/段落边界。`fixed` 的 380 字符 P90 说明它把整章塞进一块，
  且 13.9% 重叠是纯浪费（重复向量化 + 检索结果占位）。
- 与 `structured` 相比：`structured` 的块更完整（中位 218），
  但依赖可靠的标题检测。PDF 的标题靠字号推测，纯文本没有标题时
  `structured` 退化为"按块合并"，优势消失。
  **`recursive` 不依赖任何结构假设，对任意输入都给出合理结果。**

**`structured` 作为可选策略**，用于：文档有可靠章节结构（Markdown/HTML、
排版规范的 PDF），且用户需要 `section_path` 面包屑与"章节内不混切"。

**`parent_child` 用于检索精度优先场景**：命中子块后回填父块，
适合"问题很具体、但答案需要上下文"的场景（如条款查询）。
代价是存储翻倍（父+子）且检索侧必须过滤 `chunk_level=0`。

**`sentence_window` 用于连续散文/问答记录**：窗口大小可在查询时动态决定，
不需要重建索引。代价是必须保留句子邻接关系（用 `window_meta` 表达，不是 `parent_chunk_id`）。

**`fixed` 只作对照基线**，不用于生产。

## 3. 为什么不做"哪个策略 recall 最高"的结论

本步骤**没有做检索质量对比**，原因是缺少必要条件：

- 没有黄金集（`tests/eval` 是 A15 的事）；
- 语义策略没有接 embedding（会退化为 recursive，测了也没意义）；
- 检索链路（A11）还不存在，无法计算 recall@k / MRR。

**用分块的形式指标（覆盖率、重叠率、长度分布）做选型是诚实的下限**：
这些指标能排除明显有问题的策略（有内容没覆盖、重叠浪费翻倍），
但无法证明"某种切法的召回更高"。后者必须在 A11 有了检索链路、
A15 有了黄金集之后才能测。

**不编造 recall 数字**——那比没有结论更糟，因为它会误导后续决策。

## 4. 已知局限

1. **`semantic` 未经真实模型验证**：实现完成（相似度突变点检测 + 降级），
   但没在真实 embedding 下跑过。需要在 A11 接入检索后补测，
   否则无法判断"语义切分是否值得它的额外成本（每段要算向量）"。
2. **`sentence_window` 的窗口扩展未实现**：chunk 上已带
   `window_id`/`sentence_index`/`window_size` 元数据，
   但"检索时按它取前后 N 句"属于 A11 的检索逻辑。
3. **中文句子切分依赖标点**：`。！？；` 为主。无标点的长句会退化为按
   `chunk_size` 硬切（已在 `sentence_window` 里处理，标记
   `split_long_sentence=true`）。
4. **PDF 标题检测靠字号**（`_guess_level`），不保证准确。
   标题识别错误会让 `structured` 的 `section_path` 出错，
   但不会导致切分失败（只是面包屑不准）。

## 5. 验收记录

- 单测 45 项（`tests/test_chunkers.py`）：契约参数化覆盖全部 6 策略、
  位置一致性、确定性、边界输入、产物格式往返。
- 集成测试 6 项（`tests/test_worker_chunk_handler.py`）：端到端落库、
  幂等、父子关系回填、快照参数优先、产物缺失/损坏的永久失败。
- 真实 worker 端到端：PDF（5 页）→ 解析 → 切分 5 块，
  页码 1-5 正确、`char_start`/`char_end` 连续、`embedding` 全 NULL、
  文档状态推进到 `embedding`。
