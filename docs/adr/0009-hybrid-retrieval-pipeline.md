# A17 混合检索、重排与生成后自省：把检索图接进生产

> 日期：2026-11-17　环境：PostgreSQL 18.6 + pgvector 0.8.7 + bge-m3 + DeepSeek
> 复现：`uv run pytest tests/test_keyword_extraction.py tests/test_reranker_provider.py`
> 评估：`uv run python tests/eval/run_eval.py --label hybrid-fts --keyword`
> 回填：`uv run python scripts/backfill_search_text.py`

## 1. 这次补的缺口

A11 实现了 `RetrievalGraph`（向量/关键词/RRF/rerank/自省），但：

1. **生产链路没接**：`ChatService._retrieve` 手写了一条纯向量检索，
   关键词、融合、重排全部没跑；
2. **关键词路对中文不可用**：`to_tsvector('simple', content)` 不切中文，
   查询词永远匹配不到（ADR 0003 的"混合检索零收益"结论由此而来）；
3. **reranker 是 `NoOp`**：接口在，实现没有；
4. **生成后自省没有调用点**：提示词写好了，`ChatService` 只做生成前的
   `answerability`。

本次把四件事补成一条可配置、可降级、可观测的链路。

## 2. 最终链路

```
query
  │
  ├─ (多轮) rewrite: history + query → standalone_query        [LLM, 可关]
  │
  ├─ vector: bge-m3 → chunk_embeddings(HNSW) → 取限 → chunk 过滤   [必跑]
  │     第一段：旁表 top-(limit×4)
  │     第二段：chunks + is_current + level=0 + ACL + kb
  │
  ├─ keyword: jieba 提词 → to_tsquery(simple) → ts_rank           [可关]
  │     索引：chunks.search_vector（search_text 的生成列，GIN）
  │
  ├─ RRF 融合（只用排名，k=60）                                    [关键词开时]
  │
  ├─ rerank（cross-encoder，只重排前 N 条）                        [可关]
  │     失败 → 全 0 分 → 稳定排序保留 RRF 顺序（降级链）
  │
  └─ final_limit=5 → 回填 Material（位置信息/标题/向量距离）
        │
        ├─ answerability（生成前）→ 决定拒答
        ├─ 生成（SSE）
        └─ support（生成后）→ 拦幻觉：flag / refuse
```

## 3. 关键词检索：为什么是 jieba + 内置 FTS

中文 FTS 的根因不是"没有 BM25"，是"没有分词"。可选方案与实际取舍：

| 方案 | 结论 |
|---|---|
| `zhparser` / `pg_bigm` | 当前 PG 18 未提供；编译 SCWS/扩展，运维成本高 |
| `pg_trgm` + `ILIKE '%词%'` | 可用，但只有"包含"语义：无词频、无相关性排序，中文 2 字词用不上 trigram 索引 |
| ParadeDB `pg_search`（BM25） | 能力最合适，但需安装 Rust 扩展；本机/多数托管 PG 不支持 |
| **jieba 预分词 + `simple` FTS** | 纯 Python 依赖、零扩展；按词项索引 + `ts_rank` |

落地方式：

- **写入侧**（worker `chunks` handler）：
  `tokenize_for_index(content)` → `chunks.search_text`（空格分隔词项）。
- **DDL**：生成列
  `search_vector = to_tsvector('simple', coalesce(nullif(search_text,''), content))`，
  GIN 索引 `ix_chunks_search_vector`。
- **查询侧**：`extract_keywords(query)` 用**同一个 jieba 分词器**，
  构造 `to_tsquery('simple', '''词1'' | ''词2''')`，按 `ts_rank` 降序。

两个必须遵守的约束：

1. **查询与索引必须同一分词器、同一停用词**。切分不一致的后果是
   "关键词路永远 0 命中"，且不报错。
2. **空 `search_text` 回退到 `content`**：直接 SQL 插入的历史/测试数据
   英文仍可检索；**中文历史数据必须跑回填脚本**，否则整句一个 token。

`ts_rank` 不是完整 BM25（没有 IDF），但已包含词频与长度归一化。
真正的 IDF/BM25 需要扩展或应用层全局统计，留作后续；
只有用真实查询日志证明"词频排序不够"时才值得引入。

## 4. 为什么检索图改回顺序执行

原实现用 `asyncio.gather` 并行跑向量与关键词。生产里两路共用**同一个
`AsyncSession`**（请求级事务，RLS 上下文在 `set_config` 里），
而 `AsyncSession` 不是并发安全的：两个协程同时发 SQL 会抛
`InvalidRequestError` / greenlet 错误。

真正并行需要两套连接，并保证两个会话都设置了租户上下文；
而两路查询都是毫秒级，真正的延迟在 embedding 与 LLM。因此改成顺序执行，
并在 `RetrievalGraph._retrieve` 的 docstring 里写清"不是遗漏，是被会话安全否掉"。

## 5. 融合：RRF 仍然只用排名

两路分数量纲不同（余弦距离 0~2；`ts_rank` 依赖词频与文档长度），
加权需要按结果集归一化，而归一化会让同一 chunk 在不同查询里的权重
不可解释。RRF 只用排名，免疫量纲与分数漂移，只有一个 k。

**每个来源的原始命中数必须单独记**（`retrieval_meta.source_counts`）：
融合后一个条目可能来自任一路，`len(candidates)` 看不出"关键词路到底
有没有命中"——这正是 ADR 0003 无法归因的原因。

## 6. 重排：降级链在调用方，不在工厂

- 模型：`LocalBGEReranker`（sentence-transformers `CrossEncoder`），
  惰性加载、`to_thread` 推理、按 `max_pairs` 分批。
- 只重排前 `rerank_top_n` 条；尾部保留但不评分（丢弃会伤召回）。
- **分数用原始 logits，不做 sigmoid**：重排只关心顺序，sigmoid 是单调
  变换；保留原分数才能观察模型置信度漂移。
- **推理失败 → 返回全 0 分**：图里是稳定排序，全 0 等于保留 RRF 顺序。
  抛异常会让"排序退化"变成"整个问答失败"。
- `use_rerank=true` 但 `RERANKER__PROVIDER=none`：装配期显式降级并告警，
  不跑一次"全 0 分排序"假装开了。

## 7. 生成后自省：与拒答是两个判断

| 判断 | 时机 | 问题 | 用途 |
|---|---|---|---|
| `answerability` | 生成前 | 材料**能否**回答 | 决定拒答 |
| `support` | 生成后 | 答案的事实**是否有材料依据** | 拦幻觉 |

`support` 失败的处理：

- `self_check_action=flag`（默认）：保留答案，`done.degraded=true`、
  `retrieval_meta.self_check` 记录、指标 `rag_self_check_total` 计数。
  先观察误报率，不直接替换答案。
- `self_check_action=refuse`：库里内容替换为拒答文案，`done.refused=true`、
  `citations=[]`。**流式已发出的 token 无法收回**，因此 `done` 必须是
  前端权威状态（这与现有 `refused` 契约一致）。

解析失败（`raw_output == ""`）**不触发拒答**，只记 `result=error`：
一次 JSON 格式抖动不该让用户拿不到任何答案。

## 8. 引用与分数方向

`Citation.relevance_score` 的契约是**余弦距离（0 最相关）**。
RRF 分与 rerank 分都是"越大越相关"，不能写进引用。因此检索闭包用
`vector_distance_by_id` 单独留存向量距离；关键词独有的 chunk 没有距离，
传 `None`（"不知道"比"方向相反"好）。

## 9. 配置与默认值

| 配置 | 默认 | 说明 |
|---|---|---|
| `RAG_RETRIEVAL__USE_KEYWORD` | `true` | 关键词路（jieba + FTS） |
| `RAG_RETRIEVAL__USE_REWRITE` | `true` | 多轮独立化改写 |
| `RAG_RETRIEVAL__USE_RERANK` | `false` | CPU 密集；先在评估里验证 |
| `RAG_RETRIEVAL__USE_SELF_CHECK` | `false` | 多一次 LLM 调用 |
| `RAG_RETRIEVAL__SELF_CHECK_ACTION` | `flag` | 观察期不替换答案 |
| `RAG_RETRIEVAL__VECTOR_LIMIT` | 20 | 融合前召回 |
| `RAG_RETRIEVAL__KEYWORD_LIMIT` | 20 | 融合前召回 |
| `RAG_RETRIEVAL__FINAL_LIMIT` | 5 | 拼进 prompt 的条数 |
| `RAG_RETRIEVAL__MAX_KEYWORDS` | 8 | OR 链长度上限 |
| `RAG_RERANKER__PROVIDER` | `none` | `local_bge` 可选 |
| `RAG_RERANKER__MODEL_NAME` | `BAAI/bge-reranker-base` | CPU 可接受 |

## 10. 验证与遗留

已验证（单测，无需外部依赖）：

- jieba 提词：停用词、疑问词、英文/数字标识、空输入、数量上限；
- reranker：顺序对应、空输入、分批、长度不符报错；
- 检索图：配置与实现一致性、RRF、重排尾部保留。

集成验证（需真实 PG，迁移后）：

- 中文关键词检索命中 `二线/城市/住宿`，且 400 元 chunk 排在 600 元前；
- ACL 在关键词路同样生效；
- 历史数据跑 `scripts/backfill_search_text.py` 后命中。

遗留：

1. **真 BM25/IDF**：`ts_rank` 无 IDF，跨文档的罕见词权重不足。
   等有真实查询日志与黄金集后再评估是否引入 `pg_search`。
2. **评估语料仍太小**（ADR 0003 第 4 节）：37 chunk 上所有策略满分，
   无法比较。混合检索的增益必须用 A15 的真实业务语料验证。
3. **重排模型未做量化/批内并发**：CPU 上 20 条候选约数百毫秒，
   若成为瓶颈再考虑 ONNX/量化；不提前优化。
4. **中文自定义词典**：业务术语（如产品名）可能被 jieba 切错。
   需要时用 `jieba.load_userdict`，但要有词典维护流程，否则会变成
   "某人本地加词、线上没加"的不可复现问题。
