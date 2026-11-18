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

## 8. 还没做的（明确划出去）

- **索引重建触发**（`POST /admin/knowledge-bases/{id}/index-rebuild`）：
  需要新增 `REBUILD_INDEX` worker handler + 分批短事务会话工厂
  （现有 `TaskHandler` 契约只给一个 session，做不到分批提交）。
  这是 `docs/adr/0002` 里记的待改进项，动事务边界，单独一步做更安全。
  本次只补了**只读**的索引版本列表，排障已经够用。
- **管理端跨租户汇总**：`scope_tenant_id=None` 支持，但需要显式的
  跨租户接口 + 审计，不做成 `all_tenants=true` 开关（太容易被顺手打开）。
