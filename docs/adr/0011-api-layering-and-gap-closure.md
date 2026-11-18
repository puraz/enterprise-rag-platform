# 0011 接口分层与补口：`v1/user` vs `v1/admin`，以及补掉的六条断链

> 日期：2026-11-18
> 验证：`uv run pytest tests/test_api_layering.py tests/test_admin_permissions.py -q`

## 1. 为什么重排目录（而不是"只是整理一下"）

重构前 `api/v1/` 是平的六个文件：

```
api/v1/{auth, chat, documents, knowledge_bases, uploads, admin_audit}.py
```

`admin_audit.py` 是唯一的管理面文件，它的前缀写成了 `/admin/audit`，
其余五个是用户面。这个结构的问题不是"不够整齐"，而是**它无法回答
一个必须回答的问题：这个接口属于哪一面**。

判断只能靠读依赖：

```python
# 这个接口是用户面还是管理面？
async def handler(ctx: PermissionContext = Depends(get_permission_context)) -> ...
```

而权限依赖在一个函数签名里可能和业务逻辑混在一起。新加接口时，
"它该用哪套鉴权"只存在于作者脑中；review 时也只能靠肉眼比对。
一旦判断错，管理能力就会被暴露给用户前端，而**接口仍然返回 200**。

重排后：

```
api/v1/user/    ← get_permission_context（租户角色 + ACL）
api/v1/admin/   ← get_admin_context（平台角色 ∪ 租户管理员 + 能力集）
```

**URL 完全不变**。这不是妥协，是硬约束：网关按 `/api/v1/admin` 前缀
禁止用户前端访问管理接口，前端代码、部署文档、第三方集成也都按旧
URL 写。`tests/test_api_layering.py::test_existing_user_endpoints_unchanged`
用精确清单锁死 25 条已有端点。

## 2. 管理面鉴权为什么要做成"能力集"而不是角色 if

A4 只有一个管理接口（审计），权限判定写在 `AuditQueryService` 内部：

```python
can_query_global = platform_role in {SUPER_ADMIN, AUDITOR}
can_query_tenant = ctx.has_any_role(OWNER, ADMIN)
```

接口数涨到九个之后，这段逻辑会以三种方式腐烂，而且**都不报错**：

| 腐烂方式 | 现象 | 后果 |
|---|---|---|
| 新接口忘了判 | 任何登录用户都能调 | 最严重：无信号 |
| 判错方向 | 审计员能改配置 | 审计员本应只读 |
| 口径漂移 | A 能进、B 不能 | 前端无法解释 |

所以把"谁能做什么"收敛成 `AdminCapability` 枚举 + 一张映射表：

```python
_PLATFORM_CAPABILITIES = {
    SUPER_ADMIN: frozenset(AdminCapability),  # 全部
    PLATFORM_OPERATOR: frozenset({READ, OPERATE}),  # 不能查审计
    AUDITOR: frozenset({READ, AUDIT}),  # 不能改任何东西
    NONE: frozenset(),
}
```

**`platform_operator` 不能查审计**是这张表最容易被写错的一格：
运营管运维（重试任务、看成本），但审计日志能暴露"谁在什么时候看了
什么文档"——那是审计员的职责。`test_platform_operator_cannot_audit`
专门钉住它。

每个管理路由用 `dependencies=[Depends(require_admin_capability(X))]`
声明能力，`test_admin_package_declares_capability_on_every_route`
用 AST 扫描保证 9/9 都挂了（漏挂 = 测试红，而不是上线后被探测到）。

## 3. 数据范围：`tenant_id` 与 `scope_tenant_id` 必须分开

`AdminContext` 里有两个看起来相似的字段，语义完全不同：

| 字段 | 含义 | 平台角色 | 租户管理员 |
|---|---|---|---|
| `tenant_id` | token 里选中的租户 | 任意租户 | 自己的租户 |
| `scope_tenant_id` | **能看哪些租户的数据** | `None`（可跨租户） | 自己的租户 |

混用会泄漏：一个平台超管登录在租户 A，他能看 A 和 B；一个租户 owner
登录在 A，他只能看 A。所有查询必须用 `ctx.restrict_to(requested)` 收敛，
而 `test_tenant_admin_cannot_read_other_tenant` 钉住这条。

## 4. 一个真实的静默 bug：管理面用了无租户上下文的会话

这是这次改动中最值得记录的一处。

初版让管理面用 `get_session`（**不设置 `app.tenant_id`**），
理由是"平台角色要看所有租户，不能被 RLS 限制"。

问题：`tasks` / `documents` / `knowledge_bases` / `model_calls` /
`chunks` / `tenant_memberships` **全在 `RLS_TABLES` 里**。
RLS 策略是：

```sql
USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
```

未设置时 `current_setting` 返回空串，`NULLIF` 转成 NULL，
`tenant_id = NULL` 结果为 NULL（非真）→ **拒绝所有行**。

后果，且全部不报错：

- 看板所有计数是 0；
- 任务台账永远空；
- **租户 owner 调管理接口 403**（查不到自己的成员关系）。

为什么集成测试没抓到：本机开发用的角色是超级用户，`BYPASSRLS`
无条件绕过策略。这个 bug 只会在真实 `rag_app` 角色下出现。

修复分两层：

1. `get_admin_context` 用两段**短事务**：平台角色查 `users`（不在 RLS，
   普通会话）；租户角色查 `tenant_memberships`（**在 RLS，必须带
   `tenant_id` 上下文**）；
2. 管理面查询改用两个新依赖：
   - `Depends(get_admin_db)`（`permissions/admin.py` 里的，按
     `data_tenant_id` 开 RLS 上下文）；
   - `scoped_admin_session(ctx, db, requested_tenant_id=...)`
     （支持 `tenant_id` 参数的路由，上下文跟着收敛后的范围走）。

第三种写法（`deps.get_admin_db`，BYPASSRLS）**没有用于普通管理查询**：
它绕过租户隔离，正确性全靠每个 SQL 记得写 `WHERE tenant_id`——
漏一处就是跨租户，而且返回 200。

守卫：`test_admin_routes_use_rls_scoped_session` 静态扫描 admin 包，
出现 `get_session` 且没有 `get_admin_db` / `scoped_admin_session`
就失败。反向验证过：故意改回 `get_session` 会立刻失败。

## 5. 补掉的六条断链（都是有契约、有实现、没有出口）

这些不是"新功能"，是**已有代码引用了一个不存在的接口**。
这类缺口的共同特征：契约文档写着、前端按它写了、联调才发现，
而每个组件单独看都正确。

| # | 断在哪 | 症状 | 补的接口 |
|---|---|---|---|
| 1 | `LoginPage.tsx` 链到 `/api/v1/auth/oidc/login` | 点击"企业账号登录"404 | `GET /auth/oidc/authorize` + `POST /auth/oidc/exchange` |
| 2 | `TaskAccepted.poll_url` 文档写 `GET /api/v1/tasks/{id}` | 上传后进度条永远不动 | `GET /tasks/{task_id}` + `POST /tasks/{task_id}/cancel` |
| 3 | `RefreshTokenStore.revoke_token()` 无调用方 | 登出只清本地，服务端 token 仍有效 30 天 | `POST /auth/logout` |
| 4 | `mark_version_current` / `mark_version_chunks_stale` / `AuditAction.DOC_VERSION_ROLLBACK` 都在，无调用方 | B5 要求的版本回滚无法执行 | `POST /documents/{id}/versions/{vid}/activate` |
| 5 | `StorageBackend.presigned_url` + `AuditAction.DOC_DOWNLOAD` 都在，无调用方 | 无法下载原文（只能看引用片段） | `GET /documents/{id}/download` |
| 6 | `ConversationRepository.rename/archive` 都在，无调用方 | 会话列表全是 30 字截断的自动标题 | `PATCH /conversations/{id}` |

另外两个能力是从零补的：

- **多租户切换**：用户可以有多个 membership，但 access token 只能带一个
  `tenant_id`，且没有列表接口 → 登录后永远停在第一个租户。
  `GET /me/tenants` + `POST /me/tenants/switch`（换发新 token，并校验
  目标租户的 active 成员关系——`tenant_id` 来自请求参数，不可信）。
- **管理面九个接口**：看板的计数/成本拆解、任务台账与死信重试、
  文档台账/版本快照/chunk 预览/索引版本、审计链验证。
  这些的原料全在仓储里写好了（`ModelCallRepository.breakdown`、
  `AuditRepository.verify_chain`、`TaskRepository.list_dead`…），
  只是**从未有任何路由调用它们**。

## 6. 为什么 OIDC 拆成两个端点

标准授权码 + PKCE 必须有浏览器往返：

```
① GET  /auth/oidc/authorize → {authorize_url, state}
② 浏览器跳 IdP，用户登录
③ IdP 302 → 前端回调页（带 code & state）
④ POST /auth/oidc/exchange  → {access_token, refresh_token, ...}
```

合并成一个端点会让服务端必须在 ②③ 期间持有会话状态 → 有状态 API
→ 多副本要粘性会话、重启丢状态。

两个关键设计：

- **`code_verifier` 不在 authorize 响应里**：PKCE 的前提是 verifier
  只在浏览器手里。服务端持有一份就等于"任何人拿到 state 都能完成授权"。
- **`state` 存服务端且一次性消费**（`OIDCStateStore`，Redis + `GETDEL`）：
  前端比对只能防"state 不匹配"，防不了**重放**。服务端消费才能让第二次
  提交失败。

## 7. chunk 预览为什么是管理面唯一读正文的接口

文档台账只给元数据（标题、状态、计数、失败原因）。理由是管理权限管的是
"系统怎么运行"，不是"我能看什么内容"——管理员可能没有该文档的读权限。

唯一例外是切分质量检查（`GET /admin/documents/{id}/versions/{vid}/chunks`），
因为排障必须看正文。它的约束：

1. 能力要求提到 `OPERATE`（不是只读能力）；
2. 每次调用写 `admin.chunk_view` 审计（谁在何时看了哪个版本）；
3. 正文按 4000 字符截断（一次 20 条不至于把浏览器拖死）。

## 8. 后续补齐（原「还没做的」两项，已在本轮完成）

### 8.1 索引重建全链路

原计划分成"只读版本列表（本步）"与"重建触发（下一步）"两段，理由是
重建需要动 `TaskHandler` 的事务边界。现已补齐，做法如下。

#### 8.1.1 扩展 `TaskHandler`：`owns_transaction`

默认 handler（parse / chunk / embed）是**单文档、单事务**：引擎给一个
租户会话，整段包在一个事务里，失败整体回滚。

索引重建做不到。十万 chunk 回填几十分钟，整段一个事务会同时带来：

| 问题 | 现象 |
|---|---|
| 连接被长时间占用 | 阻塞 vacuum；多个重建任务迅速耗尽连接池 |
| 崩溃丢全部进度 | 重跑从头再来 |
| `locked_at` 不更新 | 超 `lock_timeout` 被判卡死 → **两个 worker 同时回填**（付款翻倍） |

所以在协议上加一个布尔属性：

```python
class TaskHandler(Protocol):
    owns_transaction: bool = False

    async def __call__(self, *, session, task, progress, session_factory=None): ...
```

`owns_transaction = True` 时引擎传 `session=None` + `session_factory`，
由 handler 自己按批开事务。代价是**幂等责任完全转移给 handler**。

三个单文档 handler 显式声明 `owns_transaction = False`（而不是依赖默认值）：
默认值在运行期是隐式的，显式写出来让"这个 handler 的事务边界在哪"
成为 review 时看得见的事实。

#### 8.1.2 为什么要"锁续期"

`reclaim_stale` 的判据是 `locked_at < now() - lock_timeout`（默认 600 秒）。
引擎只在认领时写一次 `locked_at`。重建跑 20 分钟，第 10 分钟就会
被另一个 worker 抢走重跑。

新增 `TaskRepository.touch_lock()`，handler 每批调一次。它带
`worker_id` + `status=running` 条件——更新 0 行说明锁已被抢走，
handler **必须立刻停止**（继续跑就是双份费用 + 两份打架的进度日志）。

异常与"0 行"要区分：异常只是这次没刷上（下批再试，不能停），
0 行才是锁真的丢了（必须停）。混为一谈会让一次网络抖动
杀掉一个正常的十分钟任务，然后干等 `lock_timeout` 才被重派。

#### 8.1.3 三条独立接口（不是一个"重建"接口）

| 阶段 | 接口 | 状态码 | 能力 | 理由 |
|---|---|---|---|---|
| 回填 | `POST .../index-rebuild` | 202 | `OPERATE` | 不切换，检索仍走旧版本，风险低 |
| 切换 | `POST .../index-versions/{id}/activate` | 200 | `ADMIN` | 立刻改变所有检索结果 |
| 回滚 | `POST .../index-versions/{id}/rollback` | 200 | `ADMIN` | 同上，方向相反 |

合成一条会让"回填完成自动切换"成为默认——而**回填完成不等于质量合格**
（新模型可能召回更差）。人工验收是刻意的。

切换与回滚要求 `confirm_version_tag` 与目标完全一致。URL 里已有
`version_id`，加 tag 看似冗余，但它防的是"前端把两个版本的 ID 搞反"
（`.map` 里拿错字段很常见），而切换是用户立刻能感知的操作。

切换前**硬校验向量数 ≥ 预期 chunk 数**：未回填完就切换会让检索
静默丢失内容——用户说"查不到"，而文档在库里。这比切换失败严重得多。

#### 8.1.4 版本号与并发

- `version_tag` 由服务端生成 `v{序号}-{模型短名}`。客户端指定会
  在并发发起时撞 `uq_index_versions_kb_tag`，报错是 "duplicate key"，
  根因（并发）完全看不出来。
- 同一知识库同时只允许一个 `building` 版本（`find_building` + 409）。
  并行回填争抢 GPU/API 配额，且两个半成品无法判断该切哪个。
- 回填判据是**旁表 EXISTS**（`chunk_embeddings` 里有没有该版本这一行），
  不是计数器。计数器会漂移，旁表事实不会——这也是幂等的基础。

### 8.2 管理端跨租户汇总

实现了 `GET /admin/dashboard/cross-tenant-cost`，**不做 `all_tenants=true` 开关**。

理由在 8.2 原稿里已经写清，这里补一条实施细节：开关一旦存在，
被误开（前端默认值写错、URL 被复制）时接口返回 200 和"看起来合理"的数据，
没有任何异常信号。独立接口让这件事在代码里显式可见。

三层保护：

1. `AdminCapability.ADMIN` 能力（平台超管 / 租户管理员）；
2. 接口体内再检查 `scope_tenant_id is not None` → 403。
   租户管理员有 `ADMIN` 能力（他能管自己租户的配置），
   但"全平台成本"不属于任何租户，**光靠能力集不够**；
3. 强制审计，且 `tenant_id=None`（全局事件，不进任何租户的哈希链）。

必须用 `transaction_bypass_rls()`：`model_calls` 在 `RLS_TABLES` 里，
带租户上下文的会话会把它压成单租户——返回的数字看起来合理，只是漏了别人。
这是本项目唯一跨租户的**读**接口。

### 8.3 仍未做（诚实划出）

- **旧索引版本回收**：`IndexVersion.status = retired` 的版本及其向量行
  目前不自动删除。回收需要"观察期"策略（切到新版本后 N 天没人回滚才删），
  属于运维策略而非接口能力，接定时任务时再做。
- **重建期间并发写入的版本归属**：回填时若有新文档入库，它的向量会写入
  当前 ACTIVE 版本（旧版本），因此新 chunk 不在 building 版本里。
  `expected_chunk_count` 是发起时快照，这类增量会让完成度校验偏严
  （`actual < expected` 只在期望更大时才触发，这里反而安全）。
  严格的解决是"回填期间双写两个版本"，代价是双倍 embedding 费用，
  当前不做。
