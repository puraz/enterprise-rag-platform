# B4 知识库与文档接口：上传之后那一步去哪了

> 日期：2026-11-16
> 复现：`uv run pytest tests/test_documents_api.py tests/test_acl_retrieval.py -m integration -q`

## 1. 补的是一条断掉的链，不是缺一个页面

课程把 B5 写成"知识库与上传"，读起来像是纯前端工作。探索后发现
后端缺的不是"接口层的 CRUD"，而是**上传链路的最后一环**：

    A6（上传）做完 → Blob 落对象存储
    A8（解析）做完 → 拿着 document_version_id 去下载 blob

两边都对，**中间没人把 blob 变成 document**。表现：

- `POST /uploads/{id}/complete` 返回 200 + `blob_id`，看起来成功；
- `documents` 表里没有任何新行；
- worker 每 2 秒轮询 `tasks` 表，那里也没有新任务；
- 用户看到的是"上传成功但文档列表是空的"，**没有任何报错**。

`apps/` 下 grep `enqueue`，只有两处：parse handler 入队 chunk、
chunk handler 入队 embed。`PARSE_DOCUMENT` 的**唯一生产者缺失**。

这类缺陷的特征：每个组件单独看都正确、测试也都通过，
错的是组装。所以 B4 的测试刻意断言"三张表都写对了"，
而不是只测某个 service 方法。

## 2. 为什么拆成 `POST /documents` 而不是塞进 `complete`

两条上游路径在 complete 之后**必须是同一条**：

    # 首次上传
    POST /uploads → PUT parts × N → POST /uploads/{id}/complete → {blob_id}
                                                                      │
    # 秒传（内容命中）                                                  │
    POST /uploads → {deduplicated: true, blob_id} ─────────────────────┤
                                                                      ▼
                                                          POST /documents

秒传命中时没有 `upload_id`、没有分片、没有可"完成"的会话。
如果建文档逻辑塞进 `complete`，秒传就得另写一份——两份代码
迟早不一致，而差异会在最不显眼的地方（比如某一版忘了设
`is_current=true`，文档永远检索不到）。

分开的第二个好处是职责清晰：
- `complete` 只管**存储层**（合并分片、魔数校验、整体 hash、落 blob）；
- `POST /documents` 只管**业务层**（校验密级、建文档版本、入队）。

## 3. `deduplicated_blob` 不能用 `ref_count` 判断

第一版写的是：

```python
new_ref_count = await self._blobs.increment_ref(blob_id=blob.id)
deduplicated = new_ref_count > 1
```

测试立刻抓到：**第一次建文档就返回 `deduplicated=true`**。

原因：`UploadService.complete()` 已经 `increment_ref` 过一次
（那次表达的是"上传完成持有这个 blob"）。所以第一次建文档后
计数是 2，`> 1` 恒真。

`ref_count` 是个**混合计数器**：上传会话持有 + 每个文档引用，
数值本身没有业务含义，不能用来回答"这是不是重复上传"。

改成从文档侧数：

```sql
SELECT count(DISTINCT document_id) FROM document_versions
JOIN documents ON ... WHERE blob_id = :b AND deleted_at IS NULL
```

计数 > 1 才是用户理解的"同一个文件已经有一份了"。

## 4. 两个 ACL 过滤函数不能混用（安全隐患）

`permissions/acl.py` 里有两个语义相同的函数：

| 函数 | 写法 | 用在 |
|---|---|---|
| `document_access_filter` | 直接引用 `Document.xxx` 列 | `select(Document)` |
| `confidentiality_filter_for_chunks` | `Chunk.document_id.in_(子查询)` | `select(Chunk)` |

把它们混用不会报错，但结果是错的：

```python
select(Chunk).where(document_access_filter(ctx))
# → SELECT ... FROM chunks, documents WHERE documents.deleted_at IS NULL AND ...
#                              ^^^^^^^^^^^^^^^^^^ 没有 join 条件
```

SQLAlchemy 把 `documents` 加进了 FROM，**没有 join 条件**，形成笛卡尔积。
过滤条件对着"所有文档"求值，而不是"这条 chunk 所属的文档"——
SECRET 内容照样被检索出来。

实测过程：先给 `ChatService._retrieve` 接上 `document_access_filter`，
写测试断言"MEMBER 检索不到 SECRET"，**测试红了**。打印 SQL 才看到
`FROM chunks, documents`。换成 `confidentiality_filter_for_chunks` 后绿。

这个坑不报错、不告警、不影响单条查询的正确性（单文档详情用
`document_access_filter` 是对的），只在 chunk 级查询里爆。
所以 ADR 里要留一条：**chunk 级查询只能用 `confidentiality_filter_for_chunks`**。

## 5. 检索路径此前完全没接 ACL

`ChatService._retrieve` 调 `search_vector` 时**没有传 `access_filter`**：

```python
hits = await self._chunks.search_vector(tenant_id=..., query_embedding=..., limit=...)
#                                                              ↑ 没有 access_filter
```

`ChunkRepository.search_vector` 早就支持这个参数（A10 写的），
`document_access_filter` 也早就实现（A7 写的），但两边从来没接上。

后果：任何租户成员都能通过提问把机密文档的内容问出来。这比"直接
返回文档"更隐蔽——泄漏发生在答案里，用户看不出异常。

`search_keyword` 连这个参数都没有（这次补上）。

**这类缺陷只能靠"逐角色对比检索结果"发现**，所以
`tests/test_acl_retrieval.py` 里有一条**破坏性对照测试**：

```
test_search_vector_filters_by_confidentiality  → MEMBER 看不到 SECRET ✓
test_search_vector_without_filter_leaks_secret → 不传 filter 时 SECRET 泄漏 ✓
```

第二条是故意证明"漏洞存在"。它的价值是：如果哪天有人把过滤
硬编码进仓储（让"传不传都一样"），第二条会红——提醒我们
安全性当前**依赖调用方记得传**，这个约定需要维护。

破坏性验证做过：把 `access_filter=...` 从测试里摘掉，
第一条立刻变红。

## 6. 上传会话的归属校验

同一租户内，任何成员都能猜到别人的 `upload_id`（UUID 但会出现在
URL、日志、浏览器历史里）。原实现只按 `tenant_id` 查会话，于是：

- 可以往别人的上传里塞分片（污染对方文件）；
- 可以 `abort` 别人的上传（已传分片被清理，必须从头再来）。

补了 `UploadService.get_session_owner()` + 路由层的
`ensure_own_resource()`：本人或 owner/admin 放行，其他人 404
（不泄漏会话存在性）。

**为什么不把权限判断放进 service**：service 不认识
`PermissionContext`（那是 web 层概念）。"谁是所有者"（事实）
与"这个人能不能操作"（策略）分开，worker / 脚本复用 service 时
不被 web 权限模型绑住。

## 7. 建文档返回 202 而不是 201

201 表示"资源已完整创建"。这里只写了文档元数据，chunk 与向量
都还在异步管线里。若返回 201，前端会以为文档立刻可检索，
用户提问时检索不到内容——这种"接口说成功但功能没生效"的差异
很难归因。

202 + `task_id` 让前端明确知道"要轮询 `GET /documents/{id}`
直到 `status=ready`"。

## 8. 删除知识库要求库内无文档

`documents.knowledge_base_id` 是 `ON DELETE CASCADE`。物理删库会
连带删掉文档元数据，而 chunk 与向量是异步清理的——中间窗口里
检索会返回"指向已不存在文档"的结果。

所以删除是**归档**（`status=archived`），且要求文档数为 0。
错误信息必须写清"还有 N 个文档"，否则用户只能猜为什么删不掉。

## 9. 契约层的两处刻意收紧

**拒绝 `visibility=department`**：`departments` 表未建模，
`PermissionContext.department_ids` 恒为空，`knowledge_base_visibility_filter`
里该分支显式不匹配。放行会让用户建出一个**自己都看不到的库**——
数据建成功了，但谁都访问不到，且没有任何报错。

**更新只允许 name / description / confidentiality**：
- 改 `visibility` 是权限变更，该走专门接口 + 审计，不能顺手 PATCH；
- 改 chunking 参数会让已入库 chunk 与新配置不一致（同一份文档里
  既有 800 字块又有 400 字块，覆盖率统计与重建行为都无法解释）。

## 10. 一个与 B4 无关但顺手修掉的测试缺陷

`test_gateway_cost_uses_actual_model` 断言 `pricing_tier == "off_peak"`
且 `total_usd == 0.15`，但 Gateway 内部用 `datetime.now(UTC)` 判峰谷。
DeepSeek 峰时价是谷时的 **2 倍**，于是这个测试的结果取决于
"跑测试时是不是峰时段"：

- UTC 01:00-04:00 / 06:00-10:00（峰时）→ 红；
- 其余时间 → 绿。

实测在 09:2x 跑全量时它红了。修法：`Gateway.complete()` 加
`priced_at` 参数（生产传 None 用真实时钟），测试传 `off_peak_time()`。

**时间相关的断言必须固定时间**，否则 CI 会在某些时段随机变红，
最终被人用 `--no-verify` 绕过去。

## 11. 遗留

1. **`dispatch_outbox` handler 仍未实现**。当前只写 `tasks` 表，
   靠 worker 每 2 秒轮询兜底。outbox 是"降低延迟"的优化，
   不是正确性必要条件，但表会一直为空——实装转发后应补。
2. **文档版本回滚接口未做**。`DocumentRepository` 有
   `mark_version_current` / `supersede_older_versions`，但没有
   对外端点。回滚需要"切回旧版本 → 重建索引版本"，属于 A10 的范围。
3. **文档删除没有清理 chunk/向量**。软删只标记 `deleted_at` 并减
   blob 引用，chunk 与向量仍在库里（检索靠 `is_current` 与
   `deleted_at` 过滤掉）。物理清理由 `cleanup_orphan_blobs` 一类
   任务承担，该 handler 尚未实现。
4. **`visibility=department` 仍是死代码**。等 `departments` 表建模后
   才能开放，届时契约层的 validator 要一起改。
5. **前端尚未接入**。本轮只做后端（用户明确要求）。`/knowledge`
   页面仍是 Placeholder。
