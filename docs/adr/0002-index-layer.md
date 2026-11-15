# A10 索引层：旁表方案与 HNSW 实测结论

> 日期：2026-11-15　环境：PostgreSQL 18.6 + pgvector 0.8.7，本机 NVMe SSD
> 复现：`scripts/drill_index_rebuild.py`、`scripts/bench_hnsw.py`

## 1. 为什么向量要放旁表（而不是 chunks 单列）

初始设计是 `chunks.embedding` 单列 + `chunks.index_version_id` 标记归属，
"重建索引"时就地覆盖向量并改版本号。

`scripts/drill_index_rebuild.py` 演练证明这个方案**根本错误**：

```
回填时把 chunk 的 index_version_id 从 V1 改成 V2
  → 该 chunk 立刻从 V1 的查询结果里消失
  → 回填完成时 V1 一个 chunk 都没有
  → 既有停机窗口（回填期间结果逐渐变少），又无法回滚（旧向量已被覆盖）
```

**根因：一行只能存一个向量。** 这不是"写代码更小心"能绕过的，
必须改物理结构。

改为旁表：

```
chunks                  内容（一份）
  └── chunk_embeddings  向量（每个索引版本一份）
```

回填 V2 时**新增**旁表行，旧行完全不动 → 零停机；切换是一条 UPDATE；
回滚也只需改回去（旧行还在）。

### 演练结果（真实 PG）

| 指标 | 100 chunk | 500 chunk |
|---|---|---|
| 回填耗时 | 0.22s | 1.82s |
| 回填期间查询次数 | 10 | 57 |
| 查询错误 | 0 | 0 |
| **查询返回空（停机）** | **0** | **0** |
| 未完成时切换被拒绝 | ✅ | ✅ |
| 切换耗时 | 3.9ms | 4.6ms |
| 回滚耗时 | 3.8ms | 4.7ms |
| 旧版本向量保留（可回滚） | ✅ | ✅ |

代价：回填期间向量存储翻倍；检索多一次按主键的 JOIN。这是零停机的必需成本。

## 2. HNSW 索引**不能**用部分索引（最重要的一条）

原索引定义：

```sql
CREATE INDEX ix_chunk_embeddings_hnsw ON chunk_embeddings
  USING hnsw (embedding vector_cosine_ops)
  WITH (m=16, ef_construction=64)
  WHERE embedding IS NOT NULL;      -- ← 这一行让索引完全失效
```

`embedding` 列本身是 NOT NULL，所以这个谓词恒真，**看起来无害**。
实测（20000 行 × 1024 维）：

| 索引形式 | 查询 | 计划 | 耗时 |
|---|---|---|---|
| 完整 HNSW | `ORDER BY embedding <=> ?` | Index Scan | **4.2ms** |
| 部分 HNSW | 同上 | Seq Scan + Sort | 131ms |
| 部分 HNSW | 加 `WHERE embedding IS NOT NULL` | Seq Scan + Sort | 131ms |
| 部分 HNSW | `enable_seqscan=off` 强制 | **仍然 Seq Scan** | 131ms |

即：**"部分索引"这个形式本身就使 HNSW 无法服务于 `ORDER BY 向量` 排序**，
补上对应谓词无效，关闭顺序扫描也无效。

代价 30 倍延迟，**不报任何错**。只有 EXPLAIN 能发现。
已在迁移 `f04725f245ac` 中改为完整索引。

## 3. `random_page_cost` 默认值会压掉向量索引

PostgreSQL 默认 `random_page_cost = 4.0`，这是**机械硬盘**时代的估算。
SSD 上这个假设过时，后果是 HNSW 的"随机页读取"被高估，
planner 选择 Seq Scan + Sort。

实测（20000 行）：

| random_page_cost | 计划 | 耗时 |
|---|---|---|
| 4.0（默认） | Seq Scan + Sort | 191ms |
| 1.1（SSD 推荐） | Index Scan using HNSW | **7.6ms** |

已在 `DatabaseSettings.random_page_cost` 默认设为 1.1，
并通过 `server_settings` 注入每个连接。机械硬盘环境应显式设回 4.0。

## 4. 检索必须两段式（JOIN 会破坏索引）

直觉写法是一条 JOIN：

```sql
SELECT c.* FROM chunks c JOIN chunk_embeddings ce ON ce.chunk_id = c.id
WHERE c.tenant_id = ? AND ...
ORDER BY ce.embedding <=> ? LIMIT 10
```

实测 **282ms**，计划是 Hash Join + Seq Scan × 2 + Sort——HNSW 完全没用上。

原因：`ORDER BY` 的距离项来自 JOIN 右表，planner 无法把它下推成
"先走索引取 top-N 再 JOIN"，只能全表扫完再排序。
`enable_seqscan=off` 也救不回来。

改为两段式后 **2.0ms**（快 140 倍）：

```python
# 第一段：只在旁表上走 HNSW，多取 limit × 4 个候选
stage_one = select(ChunkEmbedding.chunk_id, distance)
             .where(tenant_id == ...)          # 旁表上的租户过滤（可下推到索引扫描）
             .order_by(distance).limit(limit * 4)

# 第二段：按 ID 取内容，在此应用 ACL / 知识库过滤
stage_two = select(Chunk).where(Chunk.id.in_(ids), access_filter, ...)
```

`candidate_multiplier = 4` 存在的原因：第一段在旁表上无法应用
chunk 级过滤（ACL、知识库、is_current），必须多取一些才能凑够 limit。

**已知限制**：若 ACL 过滤极严（只允许 1% 文档），4 倍可能不够，
会返回少于 limit 的结果。要彻底解决需把 ACL 条件下推到第一段
（那就要 JOIN chunks，代价是失去索引）。这个取舍留给 A11 按实测决定。

## 5. 召回率测不出来（诚实记录）

`bench_hnsw.py` 用伪随机单位向量测召回，得到 35%~59% 的"低召回"。
**这不是索引缺陷，是测试数据的性质**：

1024 维随机单位向量之间距离高度集中——实测 top-10 与 top-40 的
距离差只有 **0.017**，边界几乎无区分度。任何近似索引都无法
精确复现这种排序。

真实语义向量的邻域结构清晰得多（同类文档距离明显小于异类），
召回率才有意义。**所以本步骤没有给出"ef_search 该设多少"的结论**——
那必须等 A15 用真实文档 + 黄金集测。

`bench_hnsw.py` 保留为**索引可用性验证工具**（确认索引被 planner 选中、
测量单查询延迟），不用于召回率评估。

## 6. 遗留事项

1. **`chunks.embedding` 列尚未删除**：A3 遗留，迁移已把数据复制到旁表，
   但列保留（expand-contract 的 contract 阶段留到下一个版本）。
   当前代码不读不写它。
2. **ACL 过滤与索引的取舍**：见第 4 节的已知限制，A11 实测后决定。
3. **大规模回填分批**：`IndexRebuilder.backfill` 已按 batch 提交，
   但 `hnsw.ef_search` / `maintenance_work_mem` 在百万级数据上需重新调参。
4. **`CREATE INDEX CONCURRENTLY`**：迁移注释里写了生产环境要做，
   本步骤库为空故直接用普通 CREATE INDEX。
